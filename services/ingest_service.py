"""文件解析與索引管線。

以 MarkItDown 統一處理所有格式，取代原本規劃的六個獨立解析器。
圖多或純圖片的檔案另走 VLM 路徑，補上掃描件讀不到內容的缺口。
"""

from __future__ import annotations

import hashlib
import os
import io
import json
import zipfile
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from database import (
    ensure_vec_table,
    get_int_setting,
    get_session,
    get_setting,
    raw_connection,
    serialize,
    vec_table_dim,
)
from models import (
    DOC_FAILED,
    DOC_INDEXED,
    Chunk,
    ChunkKeyword,
    Document,
    IngestError,
)
from services import ollama_client

# MarkItDown 支援的副檔名
SUPPORTED = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".msg", ".eml", ".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm",
}

# 走 VLM 路徑的圖片格式
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}

# 使用者還沒指定知識庫資料夾時，上傳會自動建立並記住這一個。
# 放在專案底下而不是使用者家目錄，是為了讓「整包複製給別人」仍然帶得走。
DEFAULT_KB_DIR = Path(__file__).resolve().parent.parent / "sample_knowledge_base"

# Office 檔內嵌圖片的最小尺寸。低於此值多半是圖示、項目符號、logo，
# 送去 VLM 只會得到「一個藍色小圖示」這種對檢索毫無幫助的描述。
MIN_EMBEDDED_IMAGE_BYTES = 20_000

# 視覺解析的提示詞。**任務是「轉錄」不是「描述」**——這個差別是實測踩出來的。
#
# 舊版寫的是「請詳細描述圖中的所有資訊……以文字逐列敘述其內容」。
# 對一般圖片沒問題，但碰到 28 列的分類表時，模型會給出這種東西：
#
#     大類從 01 到 22，每個大類後面跟著其描述。例如：01 CPU, 02 CHIPSET,
#     03 MEMORY 等等。
#
# 這在「描述」的框架下完全合理——摘要本來就是描述的一種。問題是
# **「等等」後面的項目從此不存在於知識庫**，之後再強的檢索也救不回來。
# 同一份 PDF、同一個模型、temperature 0.1，在兩台機器上一台逐列轉錄、
# 一台摘要成「等等」，可見不是模型不行，是任務給錯了。
VLM_PROMPT = (
    "把這張圖片裡的所有文字**逐字轉錄**出來。這是資料保存，不是內容摘要。\n"
    "1. 表格一律輸出成完整的 Markdown 表格，**每一列都要寫**，不可以只寫前幾列。\n"
    "2. **禁止**使用「等」「等等」「以此類推」「例如…等」「其餘省略」這類省略寫法。"
    "寧可輸出很長，也不可以少寫任何一列。\n"
    "3. 流程圖與示意圖：寫出每個節點的文字，以及節點之間的連接方向。\n"
    "4. 圖片若沒有文字（純照片、裝飾圖），簡短說明它畫的是什麼即可。\n"
    "直接輸出結果，不要開場白。"
)

# 偵測到省略後的重試提示詞。把上一次的失敗明講出來，模型改正的機率高很多。
VLM_PROMPT_RETRY = (
    "你剛才描述這張圖片時，用了「等」「等等」之類的省略寫法，導致資料遺失。\n\n"
    + VLM_PROMPT
    + "\n\n再做一次。特別注意：**表格的每一列都必須完整寫出來，一列都不能少。**"
)

# 判定 VLM「摘要掉了」的字樣。
#
# 只抓明確的省略語，不抓單獨的「等」——「無表格等內容」「等待」都含「等」，
# 全抓會讓每一頁都重試一次，時間直接翻倍。因此「如…等」這一類要求
# 前後文成立（列舉起手式 + 80 字內收尾在「等」+ 標點）才算數。
_ELISION = re.compile(
    r"等等|以此類推|依此類推|不一一列[出舉]|以下省略"
    r"|其餘(?:項目|內容|資料|部分)?(?:省略|從略|略)"
    r"|etc\.|and so on"
    r"|[如例][：:][^。\n]{0,80}?等[。，、\n]"
    r"|如[^。\n]{0,80}?等[。，、\n]"
)


def _looks_elided(text: str) -> bool:
    """這段 VLM 輸出是不是把內容摘要掉了。"""
    return bool(_ELISION.search(text))


# 建議安裝的視覺模型。名稱的兩個細節都是實測過才確定的：
#
# 1. **要有 `-vl`。** `qwen3:8b` 沒有視覺能力
#    （`ollama show` 的 capabilities 只有 completion/tools/thinking），
#    拿來當 VLM 完全讀不到圖片。
# 2. **要寫完整的 `-instruct`。** 短標籤 `qwen3-vl:8b` 指向的是 *thinking* 版本
#    （兩者的 manifest 雜湊相同，而 instruct 版是另一個）。
#    本系統一律以 `think: False` 呼叫，且實測 17/17 全對的是 instruct 版，
#    因此建議明確指定，不要用短標籤。
RECOMMENDED_VLM = "qwen3-vl:8b-instruct"

# 標在錯誤訊息開頭，讓上層知道「這不是解析壞了，是缺少視覺模型」。
# 兩者的處理方式完全不同：前者要修檔案，後者只要裝個模型。
NEEDS_VLM_MARKER = "[NEEDS_VLM]"


@dataclass
class IngestStats:
    scanned: int = 0
    new: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    chunks: int = 0
    vlm_used: int = 0
    # 是圖片型、但因為沒有可用的 VLM 而只能略過內容的檔案
    needs_vlm: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def vlm_ready() -> tuple[bool, str]:
    """目前有沒有可用的視覺模型。回傳 (是否可用, 原因)。

    只檢查「有沒有裝、有沒有啟用」，不做辨識品質自檢——
    那個要跑一次推理，不適合放在每次建索引前。
    """
    if get_setting("enable_vlm", "1") != "1":
        return False, "VLM 功能目前為停用狀態"

    model = get_setting("vlm_model", "")
    if not model:
        return False, "尚未指定 VLM 模型"

    status = ollama_client.check_status()
    if not status.alive:
        return False, "Ollama 未啟動"

    installed = {m.split(":")[0]: m for m in status.models}
    if model not in status.models and model.split(":")[0] not in installed:
        return False, f"模型 `{model}` 尚未安裝"
    return True, ""


def count_image_type(root: Path) -> list[Path]:
    """列出資料夾裡需要 VLM 才能讀內容的檔案。

    圖片檔看副檔名就知道；掃描件要解析後才知道，成本太高，
    因此這裡只回報圖片檔，掃描件在解析當下才會被發現。
    """
    if not root.exists() or not root.is_dir():
        return []
    return [p for p in scan_folder(root) if p.suffix.lower() in IMAGE_TYPES]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------ 知識庫（資料夾）
#
# **一個知識庫 = 知識庫根目錄下的一個子資料夾；直接放在根目錄的檔案屬於「通用」。**
# 資料夾是唯一的真相來源：使用者用檔案總管把檔案丟進子資料夾也算歸類，
# 全量重建不會丟失分類，資料夾結構本身就是備份。代價是搬移要真的搬檔案——
# 但搬檔案**不需要重新向量化**，內容沒變，只要把資料庫裡的路徑同步過去。
KB_NAME_MAX = 64
_KB_NAME_BAD = re.compile(r'[\/:*?"<>|\x00-\x1f]')


def valid_kb_name(name: str) -> tuple[bool, str]:
    name = (name or "").strip()
    if not name:
        return False, "名稱不能是空的"
    if len(name) > KB_NAME_MAX:
        return False, f"名稱不能超過 {KB_NAME_MAX} 字"
    if _KB_NAME_BAD.search(name) or name in (".", ".."):
        return False, "名稱不能包含路徑符號或特殊字元"
    if name.startswith("."):
        return False, "名稱不能以「.」開頭"
    if name.lower() in SKIP_DIRS:
        return False, f"「{name}」是系統保留的資料夾名稱"
    return True, ""


def kb_of(path: Path, root: Path) -> str | None:
    """檔案所屬的知識庫：根目錄下第一層子資料夾的名稱；直接在根目錄則為 None（通用）。"""
    try:
        parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        return None
    return parts[0] if len(parts) >= 2 else None


def list_kb_names(root_path: str) -> list[str]:
    root = Path(root_path)
    if not root_path or not root.is_dir():
        return []
    names = [
        p.name for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name.lower() not in SKIP_DIRS
    ]
    return sorted(names, key=str.casefold)


def create_kb(root_path: str, name: str) -> tuple[bool, str]:
    ok, why = valid_kb_name(name)
    if not ok:
        return False, why
    root = Path(root_path)
    if not root_path or not root.is_dir():
        return False, "知識庫根目錄不存在"
    target = root / name.strip()
    if target.exists():
        return False, f"「{name}」已經存在"
    try:
        target.mkdir()
    except OSError as exc:
        return False, f"建立失敗：{exc}"
    return True, f"已建立知識庫「{name}」"


def _rebase_paths(old_prefix: str, new_prefix: str) -> None:
    """把資料庫裡所有以 old_prefix 開頭的絕對路徑改成 new_prefix。

    **四張表都要改，少一張就會留下孤兒**：documents 是索引本體；
    chunk_keywords 是關鍵字／編輯備份，靠路徑套回；doc_options 是強制視覺
    解析的開關；ingest_errors 是解析狀態頁的來源。它們都以絕對路徑當鍵。
    """
    from models import DocOption

    with get_session() as session:
        for model in (Document, ChunkKeyword, DocOption, IngestError):
            for row in session.query(model).filter(model.file_path.like(old_prefix + "%")).all():
                if row.file_path == old_prefix or row.file_path.startswith(old_prefix + os.sep):
                    row.file_path = new_prefix + row.file_path[len(old_prefix):]
        session.commit()


def rename_kb(root_path: str, old: str, new: str) -> tuple[bool, str]:
    ok, why = valid_kb_name(new)
    if not ok:
        return False, why
    root = Path(root_path)
    src, dst = root / old, root / new.strip()
    if not src.is_dir():
        return False, f"「{old}」不存在"
    if dst.exists():
        return False, f"「{new}」已經存在"
    try:
        src.rename(dst)
    except OSError as exc:
        return False, f"改名失敗：{exc}"
    _rebase_paths(str(src.resolve()), str(dst.resolve()))
    with get_session() as session:
        session.query(Document).filter(Document.kb == old).update({"kb": new.strip()})
        session.commit()
    return True, f"已將「{old}」改名為「{new}」"


def delete_kb(root_path: str, name: str) -> tuple[bool, str]:
    """只刪空的知識庫。裡面還有檔案就拒絕——要先搬走或搬到通用。

    比「自動搬到通用」安全：不會有人誤刪一整批文件的分類。
    """
    root = Path(root_path)
    target = root / name
    if not target.is_dir() or kb_of(target / "x", root) != name:
        return False, f"「{name}」不存在"
    if any(p.is_file() for p in target.rglob("*")):
        return False, f"「{name}」裡還有檔案，請先搬走"
    try:
        for sub in sorted(target.rglob("*"), reverse=True):
            sub.rmdir()
        target.rmdir()
    except OSError as exc:
        return False, f"刪除失敗：{exc}"
    return True, f"已刪除知識庫「{name}」"


def move_document(root_path: str, rel_path: str, kb: str | None) -> tuple[bool, str]:
    """把一份文件搬到另一個知識庫（kb=None 代表通用）。**不重新向量化。**

    內容沒變，切片與向量原封不動；要同步的只有路徑（四張表）與 documents.kb。
    """
    root = Path(root_path)
    if not root_path or not root.is_dir():
        return False, "知識庫根目錄不存在"
    try:
        src = (root / rel_path).resolve()
        src.relative_to(root.resolve())
    except (ValueError, OSError):
        return False, "路徑不合法"
    if not src.is_file():
        return False, "檔案不存在"
    if kb:
        ok, why = valid_kb_name(kb)
        if not ok:
            return False, why
        dst_dir = root / kb.strip()
        if not dst_dir.is_dir():
            return False, f"知識庫「{kb}」不存在"
    else:
        dst_dir = root
    dst = dst_dir / src.name
    if dst.resolve() == src:
        return True, "已經在該知識庫裡"
    if dst.exists():
        return False, f"目的地已有同名檔案：{src.name}"
    try:
        src.rename(dst)
    except OSError as exc:
        return False, f"搬移失敗：{exc}"
    _rebase_paths(str(src), str(dst.resolve()))
    with get_session() as session:
        doc = session.query(Document).filter(Document.file_path == str(dst.resolve())).first()
        if doc:
            doc.kb = kb.strip() if kb else None
            session.commit()
    return True, f"已搬到「{kb or '通用'}」"


# 結構標記：每種來源長得不一樣，但作用相同——標示「這裡是一個新單元的開頭」，
# 而且它本身就是這個單元的名字，正好拿來當切片的麵包屑。
#   Markdown 標題    → 規範文件、正規化後的表格記錄
#   <!-- Slide N --> → MarkItDown 解析 pptx 時產生
#   第 N 頁          → VLM 逐頁判讀掃描件時我們自己標的
_MARKERS = (
    (re.compile(r"^(#{1,6})\s+(.+?)\s*$"), "heading"),
    (re.compile(r"^<!--\s*Slide number:\s*(\d+)\s*-->\s*$"), "slide"),
    (re.compile(r"^#*\s*(第\s*\d+\s*頁)\s*$"), "page"),
    # 編號章節（`1.2 Testing Procedure`、`12.2.3 測試設備`）。
    # 規格書、測試計畫這類文件從 PDF 解析出來時**一個 Markdown 標題都沒有**，
    # 但它們有嚴謹的章節編號——那就是它們的標題，只是沒有 `#` 而已。
    # 實測一份 400K 字的測試計畫：`#` 標題 0 個，編號章節 542 個。
    #
    # **必須至少有一個小數點。** 只寫 `^\d+\s+文字` 會把規格數值當成章節標題：
    # 實測誤判出 `7 min`、`30 cycles`、`2 hr 2 hr`、
    # `1 cm diameter, 5kgf, 5Sec…`、`2 Vibration step stress test 5 Grms…` 等 24 種，
    # 其中「2 Vibration step stress test」還一口氣污染了 11 個切片的麵包屑。
    # 真正的章節編號幾乎都是 `N.N` 起跳；單層編號（`1 Introduction`）失去標記
    # 只是退回用空行分段，不會比改版前差。
    (re.compile(r"^(\d+(?:\.\d+){1,3}\s+[A-Za-z一-鿿][^\n]{0,60})$"), "numbered"),
)

# 句尾。硬切前的最後一道防線，至少不要把句子切斷。
#
# **英文句號一定要納入。** 只認全形標點的話，英文文件根本找不到任何句尾，
# 會直接掉到硬切——實測一份英文測試計畫有 36% 的切片從半個句子、
# 甚至半個單字開始（`tery capacity value…`，battery 被剖成兩半）。
#
# 英文句號要求**後面接空白**，否則 `12.2.3` 這種章節編號會被當成三個句子。
_SENTENCE_END = re.compile(r"(?<=[。！？；])\s*|(?<=[.!?;])\s+")

# 小於這個長度的單元不足以獨立成段，會被併進相鄰單元。
MIN_UNIT_CHARS = 120

# 去掉標記與符號後仍少於這麼多字，就是一個沒有資訊的切片（空的 `### Notes:`、
# 只剩管線符號的表格殘骸）。放進索引只會佔用檢索名額。
MIN_MEANINGFUL_CHARS = 15


def _meaningful_len(text: str) -> int:
    """扣掉麵包屑、標題符號、表格符號後，還剩多少實質內容。"""
    body = re.sub(r"(?m)^>.*$", "", text)          # 麵包屑那一行
    body = re.sub(r"(?m)^#{1,6}\s*", "", body)     # 標題符號
    body = re.sub(r"!\[.*?\]\(.*?\)", "", body)    # 圖片佔位
    body = re.sub(r"[|\-:*`\s]", "", body)         # 表格與排版符號
    return len(body)


def _marker_info(line: str) -> tuple[int, str] | None:
    """這一行是不是結構標記？是的話回傳 (層級, 顯示名稱)。

    層級是為了組出**階層路徑**。只取最近的一個標記會丟掉真正有意義的名字：

        12.1 XX 測試              ← 父章節，這才是「這段在講什麼」
        12.1.1 Testing Objective      ← 最近的標記

    實測一份 400K 字的測試計畫，1220 個切片裡**83% 的麵包屑是「N.N.N Testing XXX」
    這種每章都有的通用小節名**——單獨看等於沒有資訊，模型與使用者都無法
    從「12.1.1 Testing Objective」看出那是哪一項測試的目的。
    """
    text = line.strip()
    for pattern, kind in _MARKERS:
        m = pattern.match(text)
        if not m:
            continue
        if kind == "heading":
            return len(m.group(1)), m.group(2)
        if kind == "slide":
            return 1, f"投影片 {m.group(1)}"
        if kind == "numbered":
            name = m.group(1).strip()
            # 「12.1.1 …」是第 3 層、「12.1 …」是第 2 層
            number = name.split()[0]
            return number.count(".") + 1, name
        return 1, m.group(1).replace(" ", "")
    return None


def _marker_of(line: str) -> str | None:
    """相容用：只要名稱不要層級。"""
    info = _marker_info(line)
    return info[1] if info else None


# 麵包屑最多顯示幾層、多長。
# 全路徑可能很深（`### 文件 › 12 章 › 12.1 節 › 12.1.1 小節`），
# 而麵包屑會被前置到續段裡佔用切片預算，取最後兩層已經足夠定位。
CRUMB_MAX_LEVELS = 2
CRUMB_MAX_CHARS = 90


def _render_crumb(stack: list[tuple[int, str]]) -> str:
    if not stack:
        return ""
    names = [name for _, name in stack[-CRUMB_MAX_LEVELS:]]
    crumb = " › ".join(names)
    return crumb if len(crumb) <= CRUMB_MAX_CHARS else crumb[-CRUMB_MAX_CHARS:]


def _split_by_lines(text: str, limit: int, overlap: int) -> list[str]:
    """沒有句尾可切時退到行尾。

    表格、清單、程式碼都是以行為單位的內容，而且**經常整段沒有一個句號**，
    所以會直接掉到最後的硬切。實測的後果：一張 16 列的廠商表被切成
    `| 01012 | ROCKC` 與 `HIP (CPU/SOC) |`，模型讀到前半段之後，
    就把「ROCKC」當成一個真的廠商名稱列進答案裡——它沒有錯，
    它看到的就是那樣。切在行尾就不會有這個問題。
    """
    out: list[str] = []
    buffer = ""
    for line in text.splitlines(keepends=True):
        # 單獨一行就超過上限（超長表格列、整段沒換行的長句），只能硬切這一行
        if len(line) > limit:
            if buffer:
                out.append(buffer)
                buffer = ""
            step = max(limit - overlap, 1)
            for i in range(0, len(line), step):
                out.append(line[i : i + limit])
            continue
        if len(buffer) + len(line) > limit:
            out.append(buffer)
            buffer = ""
        buffer += line
    if buffer:
        out.append(buffer)
    return out


def _split_oversize(block: str, limit: int, overlap: int) -> list[str]:
    """單一區塊超過上限時，依序退讓：句尾 → 行尾 → 硬切。"""
    if len(block) <= limit:
        return [block]

    pieces: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_END.split(block):
        if not sentence:
            continue
        # 連一個句子都超過上限，退到行尾再切
        if len(sentence) > limit:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.extend(_split_by_lines(sentence, limit, overlap))
            continue
        if len(buffer) + len(sentence) > limit:
            pieces.append(buffer)
            buffer = ""
        buffer += sentence
    if buffer:
        pieces.append(buffer)
    return [p.strip() for p in pieces if p.strip()]


def chunk_text(text: str, size: int, overlap: int, base_locator: str = "") -> list[dict]:
    """切片。在最強的可用邊界切開，並讓每一段都帶得到自己的出處。

    **邊界是一道階梯，找得到就用，找不到才往下退**：

        1. 結構標記   ## 標題 ／ <!-- Slide N --> ／ 第 N 頁
        2. 空行段落   ← 改版前唯一的一階
        3. 句尾       。！？；
        4. 硬切       最後手段

    改版前只有第 2 和第 4 階。這對有空行的文件沒問題（知識庫裡 14 份都是），
    但一份「整份沒有任何空行」的文件——例如 Excel 轉出來的長表格——會被當成
    單一巨大段落直接硬切：實測 127 個切片裡只有 1 個帶得到欄位表頭，
    63% 的切片看不出自己在回答什麼問題。

    **麵包屑只在必要時加。** 一個結構單元若被迫拆成多段，第 2 段以後會補上
    `> 出處` 那一行；沒被拆的段落本來就從標題開頭，不需要重複。這讓原本就
    切得好好的文件幾乎零影響，只有真正缺脈絡的續段才付出這 60–80 字的成本。
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []

    # ── 第 1 階：依結構標記切成單元，同時記住每個單元的出處 ──
    units: list[tuple[str, list[str]]] = []   # (麵包屑, 行)
    stack: list[tuple[int, str]] = []         # 標題堆疊，用來組出階層路徑
    crumb = ""
    current: list[str] = []
    for line in text.split("\n"):
        info = _marker_info(line)
        if info is not None:
            level, name = info
            if current:
                units.append((crumb, current))
            # 同層或更淺的標題出現時，把比它深的都收掉
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, name))
            crumb = _render_crumb(stack)
            current = [line]
        else:
            current.append(line)
    if current:
        units.append((crumb, current))

    # 相鄰單元塞得下就併回去。
    # **少了這步，結構標記反而幫倒忙**：一份 374 字的規範有 6 個 `##`，
    # 會被切成 6 個 60 字的碎片，每片都短到沒有脈絡，檢索品質比不切還差。
    # 標記的用途是「提供可切的地方」，不是「非切不可」。
    packed: list[tuple[str, str]] = []
    for unit_crumb, lines in units:
        block = "\n".join(lines).strip()
        if not block:
            continue
        fits = packed and len(packed[-1][1]) + len(block) + 2 <= size
        # 前一個單元小到不足以獨立成段時，**即使併起來會超過上限也要併**。
        # 那多半是一個標題（`## 第 7 頁`）後面幾乎沒有內容，單獨切出來會變成
        # 一個 17 字、沒有任何資訊的向量。標題本來就該跟著它的內容走；
        # 併完超過上限也沒關係，下面的分段與句尾切割會再處理一次。
        starved = packed and len(packed[-1][1]) < MIN_UNIT_CHARS
        if fits or starved:
            # 前一段只是個沒有內容的標題時，麵包屑要換成**進來這一段**的。
            # 一份 Excel 轉出來的文件開頭是工作表名（`## MIL-STD-810H QA`），
            # 沿用它的話，每個續段的麵包屑都是工作表名——那是全文件共通的資訊，
            # 等於沒有脈絡。該帶的是這筆記錄的題目。
            keep = unit_crumb if (starved and unit_crumb) else packed[-1][0]
            packed[-1] = (keep, packed[-1][1] + "\n\n" + block)
        else:
            packed.append((unit_crumb, block))

    # 收尾：最後一個單元太小時往前併，它後面已經沒有東西可以帶它了
    if len(packed) > 1 and len(packed[-1][1]) < MIN_UNIT_CHARS:
        tail = packed.pop()
        packed[-1] = (packed[-1][0], packed[-1][1] + "\n\n" + tail[1])

    chunks: list[dict] = []

    def emit(body: str, unit_crumb: str, continued: bool) -> None:
        body = body.strip()
        # 沒有實質內容的段落不進索引。空的 `### Notes:`、只剩管線符號的表格殘骸
        # 都會產生一個語意近乎隨機的向量，偶爾擠掉真正相關的來源。
        if not body or _meaningful_len(body) < MIN_MEANINGFUL_CHARS:
            return
        if continued and unit_crumb:
            body = f"> {unit_crumb}\n\n{body}"
        locator = f"{base_locator} › {unit_crumb}" if unit_crumb else base_locator
        chunks.append({"content": body, "locator": locator})

    for unit_crumb, block in packed:
        # 麵包屑會佔掉預算，先扣起來。**這個上限要套用在所有分支**，
        # 只扣在硬切那一支的話，靠空行併出來的段落照樣會在補上麵包屑後超標。
        prefix_cost = len(unit_crumb) + 6 if unit_crumb else 0
        budget = max(size - prefix_cost, size // 2)

        # ── 第 2 階：單元內以空行分段，塞得下就併在一起 ──
        paragraphs = [p.strip() for p in block.split("\n\n") if p.strip()]
        buffer = ""
        first = True
        for para in paragraphs:
            if len(para) > budget:
                # **緩衝區要併進去一起切，不能單獨送出。**
                # 記錄的標題與內文之間有空行，若在這裡把緩衝區直接 emit，
                # 標題就會變成一個 52 字、只有題目沒有答案的獨立切片，
                # 而它的答案在下一段裡失去題目——正好是這次要修的毛病。
                if buffer:
                    para = buffer.strip() + "\n\n" + para
                    buffer = ""
                # ── 第 3、4 階：句尾 → 硬切 ──
                for piece in _split_oversize(para, budget, overlap):
                    emit(piece, unit_crumb, not first)
                    first = False
                continue
            if buffer and len(buffer) + len(para) + 2 > budget:
                emit(buffer, unit_crumb, not first)
                first, buffer = False, ""
            buffer += para + "\n\n"
        if buffer:
            emit(buffer, unit_crumb, not first)

    return chunks


# 掃描件判定門檻：平均每頁少於這麼多字就當作沒有文字層。
# 一般文件動輒每頁上千字，實測本機樣本是 136 與 2049 字/頁；
# 真正的掃描件是 0 字/頁。取 30 留很大的餘裕，避免誤判正常的稀疏文件。
SCANNED_CHARS_PER_PAGE = 30

# 每頁 VLM 約需 9 秒。上限訂在 40 頁（最壞約 6 分鐘，含重試約 12 分鐘）。
#
# 原本是 10。實際踩到的狀況是一份 16 頁的教育訓練教材被砍掉後 6 頁，
# 而**唯一的提示只寫在解析後的內容裡**——使用者要點進「原始文件」才看得到，
# 等於靜靜地少收了三分之一的內容。10 頁對簡報型文件太小，投影片動輒數十頁。
#
# 超過上限時除了標注在內容裡，也會經由 progress 打進上傳日誌。
MAX_VLM_PAGES = 40

# 送進 VLM 前的算圖倍率。2 倍對掃描的中文字已足夠辨識，再高只是變慢。
PDF_RENDER_SCALE = 2


def _looks_scanned(path: Path, content: str) -> bool:
    """判斷 PDF 是不是沒有文字層的掃描件。"""
    if len(content) < 100:
        return True
    try:
        import pypdfium2

        pdf = pypdfium2.PdfDocument(str(path))
        pages = len(pdf)
        pdf.close()
    except Exception:  # noqa: BLE001
        return False
    return bool(pages) and len(content) / pages < SCANNED_CHARS_PER_PAGE


# 表格區塊裡「分隔列 ÷ 資料列」超過這個值就視為碎掉了。
#
# 正常的 Markdown 表格是 1 條分隔線配 N 列資料，比值接近 1/N。實測全庫：
#   一般 Markdown 0.14–0.18、正常文字 PDF 0.27、視覺解析輸出 0.11
#   簡報轉 PDF 的文字層 **0.75** ← 每一兩列就重開一個表
# 0.5 對兩群都留有餘裕。
BROKEN_TABLE_RATIO = 0.5

_TABLE_SEP = re.compile(r"^\s*\|[\s\-|:]+\|\s*$")


def _strip_broken_tables(md: str) -> str:
    """把文字層裡碎掉的表格區塊整段拿掉。**只在視覺解析成功時呼叫。**

    簡報轉出的 PDF，表格的每個儲存格是獨立文字框，pdfminer 依座標順序讀，
    欄列關係整個瓦解——名稱被踢成孤立行、代號沒有名字、每列自己重開一個表。
    開了視覺解析之後，同一份文件同時有這份殘骸和 VLM 轉錄的完整表格，
    兩者都被切片索引；實測檢索撈到殘骸那份，模型忠實照抄了一坨垃圾進答案。

    只在有 VLM 版本時才過濾，這樣永遠不會丟掉唯一的一份內容。
    以空行切區塊：殘骸表格是一整串沒有空行的表列與孤立文字交錯，會落在同一區塊。
    """
    kept = []
    for block in md.split("\n\n"):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        seps = sum(1 for ln in lines if _TABLE_SEP.match(ln))
        rows = sum(1 for ln in lines
                   if ln.lstrip().startswith("|") and not _TABLE_SEP.match(ln))
        # 至少要像個表（有分隔線、有資料列）才進入判斷；純文字區塊一律保留
        if seps >= 2 and rows and seps / rows > BROKEN_TABLE_RATIO:
            continue
        kept.append(block)
    return "\n\n".join(kept)

def _describe_once(data: bytes) -> tuple[str, str, bool]:
    """描述一張圖，發現被摘要就用更嚴格的提示詞重試一次。

    回傳 `(內容, 錯誤, 仍然省略)`。第三個值是給呼叫端示警用的——
    **重試不保證成功，所以必須讓使用者知道哪幾頁可能不完整**，
    而不是重試完就當作沒事。

    只重試一次：實測第二次改不掉的，第三次也改不掉，多跑只是浪費時間。
    """
    text, error = ollama_client.describe_image(data, VLM_PROMPT)
    if error:
        return "", error, False
    text = text.strip()
    if not _looks_elided(text):
        return text, "", False

    retry, retry_error = ollama_client.describe_image(data, VLM_PROMPT_RETRY)
    retry = retry.strip()
    if retry_error or not retry:
        return text, "", True
    # 重試同樣省略而且沒有更長，代表沒改善——留原本的，別拿更差的換掉
    if _looks_elided(retry) and len(retry) <= len(text):
        return text, "", True
    return retry, "", _looks_elided(retry)


def _vlm_read_pdf(path: Path, progress=None,
                  *, scanned: bool = True) -> tuple[str, str, bool]:
    """把每一頁算成圖片後交給 VLM。回傳 (內容, 錯誤, 是否完整轉錄)。

    `pypdfium2` 與 `Pillow` 都是 MarkItDown 已經帶進來的相依套件，
    不需要為了這條路徑額外安裝東西。

    第三個回傳值 `complete` 表示「每一頁都成功轉錄了」。**只有它為真時，
    呼叫端才可以丟掉文字層**——頁數超過 `MAX_VLM_PAGES`、或某頁解析失敗
    被跳過時，那些頁只存在於文字層，丟掉等於永久遺失。

    `scanned` 只影響開頭那段說明文字：掃描件是「沒有文字層可用」，強制
    視覺解析是「有文字層但選擇不用」，兩者不能共用同一句話。
    """
    try:
        import pypdfium2
    except ImportError as exc:
        return "", f"缺少 PDF 算圖套件：{exc}", False

    try:
        pdf = pypdfium2.PdfDocument(str(path))
    except Exception as exc:  # noqa: BLE001
        return "", f"無法開啟 PDF：{type(exc).__name__}: {str(exc)[:120]}", False

    total = len(pdf)
    blocks: list[str] = []
    errors: list[str] = []
    elided: list[int] = []
    try:
        pages_to_do = min(total, MAX_VLM_PAGES)
        for index in range(pages_to_do):
            if progress:
                progress(f"      視覺解析第 {index + 1}/{pages_to_do} 頁…")
            image = pdf[index].render(scale=PDF_RENDER_SCALE).to_pil()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            description, error, still_short = _describe_once(buffer.getvalue())
            if error:
                errors.append(f"第 {index + 1} 頁：{error}")
                continue
            if still_short:
                elided.append(index + 1)
                if progress:
                    progress(f"      ⚠ 第 {index + 1} 頁重試後仍有省略，內容可能不完整")
            if description:
                blocks.append(f"## 第 {index + 1} 頁\n\n{description}")
    finally:
        pdf.close()

    if not blocks:
        return "", "；".join(errors) or "VLM 未產生任何內容", False

    # 每一頁都轉錄成功才算完整：頁數沒被上限砍掉，而且沒有任何一頁出錯跳過。
    complete = total <= MAX_VLM_PAGES and len(blocks) == total

    if scanned:
        header = f"（本檔案無文字層，以下內容由 VLM 從掃描影像判讀，共 {total} 頁）"
    else:
        header = f"（本檔案指定強制視覺解析，以下內容由 VLM 逐頁判讀，共 {total} 頁）"
    # 這兩個警告以前只寫進內容裡，使用者要點開「原始文件」才看得到。
    # 一併打進 progress，讓它出現在上傳當下的日誌。
    if total > MAX_VLM_PAGES:
        note = f"只解析了前 {MAX_VLM_PAGES} 頁，其餘 {total - MAX_VLM_PAGES} 頁未納入索引"
        header += f"\n\n**注意：{note}。**"
        if progress:
            progress(f"    ⚠ {path.name}：{note}")
    if elided:
        pages = "、".join(f"第 {p} 頁" for p in elided)
        header += f"\n\n**注意：{pages} 的表格可能被模型摘要，內容未必完整。**"
        if progress:
            progress(f"    ⚠ {path.name}：{pages} 可能不完整，建議換用 {RECOMMENDED_VLM} 後重新索引")
    return header + "\n\n" + "\n\n".join(blocks), "；".join(errors), complete


def get_doc_option(file_path: str) -> bool:
    """這個檔案有沒有被指定強制視覺解析。"""
    from models import DocOption

    with get_session() as session:
        row = session.query(DocOption).filter(DocOption.file_path == file_path).first()
        return bool(row and row.force_vlm)


def set_doc_option(file_path: str, force_vlm: bool) -> None:
    """設定檔案的解析選項。關閉時直接刪列，不留無意義的紀錄。"""
    from models import DocOption

    with get_session() as session:
        row = session.query(DocOption).filter(DocOption.file_path == file_path).first()
        if not force_vlm:
            if row:
                session.delete(row)
                session.commit()
            return
        if row:
            row.force_vlm = True
            row.updated_at = datetime.now()
        else:
            session.add(DocOption(file_path=file_path, force_vlm=True))
        session.commit()


def _office_images(path: Path) -> list[tuple[str, bytes]]:
    """從 Office 檔（docx/pptx/xlsx）取出內嵌圖片。

    這些格式本質是 ZIP，圖片一律放在 `word/media/`、`ppt/media/`、`xl/media/`，
    **用標準庫的 `zipfile` 就能取，不需要額外套件**。

    只取點陣圖：SVG 是向量圖，VLM 讀不了；而且簡報裡的 SVG 多半是圖示與裝飾，
    描述它們只會產生噪音。
    """
    raster = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}
    out: list[tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(path) as zf:
            names = sorted(n for n in zf.namelist() if "/media/" in n)
            for name in names:
                if Path(name).suffix.lower() not in raster:
                    continue
                info = zf.getinfo(name)
                # 太小的多半是圖示、項目符號、公司 logo，描述它們沒有價值
                if info.file_size < MIN_EMBEDDED_IMAGE_BYTES:
                    continue
                out.append((Path(name).name, zf.read(name)))
    except Exception:  # noqa: BLE001
        return []
    return out


def _describe_images(items: list[tuple[str, bytes]], label: str,
                     progress=None) -> tuple[str, list[str]]:
    """把一組圖片交給 VLM 描述。回傳 (markdown 區塊, 錯誤清單)。

    **每張圖都要回報進度。** 一張圖約需數秒，32 張就是好幾分鐘；
    中間完全沒有輸出時，使用者會以為系統當掉而直接關掉視窗。
    """
    blocks: list[str] = []
    errors: list[str] = []
    total = min(len(items), MAX_VLM_PAGES)
    for index, (name, data) in enumerate(items[:MAX_VLM_PAGES], start=1):
        if progress:
            progress(f"      視覺解析 {label} {index}/{total}（{name}）…")
        description, error, still_short = _describe_once(data)
        if error:
            errors.append(f"{name}：{error}")
            continue
        if still_short and progress:
            progress(f"      ⚠ {label} {index} 重試後仍有省略，內容可能不完整")
        if description:
            blocks.append(f"### {label} {index}（{name}）\n\n{description}")
    return "\n\n".join(blocks), errors


def extract_markdown(path: Path, enable_vlm: bool, force_vlm: bool = False,
                     progress=None) -> tuple[str, bool, str]:
    """以 MarkItDown 抽出 Markdown，並整理成適合切片的形狀。

    正規化獨立成一層而不是散在各個 return：`_extract_raw()` 有六個出口
    （一般檔、強制 VLM、掃描件、純圖片…），逐一處理遲早會漏掉一個。
    """
    content, used_vlm, error = _extract_raw(path, enable_vlm, force_vlm, progress)
    if content:
        content = normalize_wide_tables(content)
    return content, used_vlm, error


# 儲存格超過這個長度就視為「長文欄位」，整張表改寫成逐筆記錄。
# 200 字是這樣抓的：一般表格的欄位（人名、日期、料號、狀態）遠低於它；
# 而長文欄位動輒上千字。中間地帶很少，門檻不敏感。
WIDE_CELL_CHARS = 200

# 標題行只放短欄位。超過這個長度的欄位放進內文，否則標題會變成一整段文章。
TITLE_CELL_CHARS = 80


def _unescape_cell(text: str) -> str:
    r"""還原 MarkItDown 為了塞進表格而做的轉義。

    Markdown 表格的一列**必須在同一行**，所以儲存格裡的真換行會被寫成字面
    `\n`，`**粗體**` 會被轉義成 `\*\*`。用 openpyxl 讀原始檔驗證過：
    儲存格裡本來就是真換行與真的 `**`，所以這個還原是可逆的，沒有資訊損失。

    不還原的話，AI 讀到的是一整片 `\n` 與 `\*` 雜訊，畫面上也是。
    """
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    # 只還原 Markdown 會轉義的那幾個字元，避免動到內容裡本來就有的反斜線
    return re.sub(r"\\([*_\[\]()#+\-.!`>~|])", r"\1", text)


def _split_table_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def normalize_wide_tables(md: str) -> str:
    r"""把「儲存格塞了長文」的 Markdown 表格改寫成逐筆記錄。

    這種表格（例如一份問答紀錄 Excel，每列的「完整答案」欄有數千字）
    在 Markdown 表格形式下有三個問題，而且互相加乘：

      1. **一列就是一行**，整份文件可能一個空行都沒有，切片器找不到任何邊界，
         只能硬切；實測 127 個切片裡只有 1 個帶得到欄位表頭。
      2. 儲存格內的換行與粗體被轉義成字面 `\n`、`\*`，變成雜訊。
      3. 表格渲染出來也沒有用——沒有人能讀一個裡面塞了三千字的儲存格。

    改寫成「一列一筆記錄」同時解掉三個：產生了 `###` 標題當切片邊界、
    順手還原轉義、閱讀時也正常。短欄位放進標題（切片的麵包屑就有東西可帶），
    長欄位獨立成段。

    **欄位短的表格不動**（人員一覽、里程碑表這種），它們本來就該保持表格樣子。
    """
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        # 表格 = 標題列 + 分隔列（| --- | --- |）+ 資料列
        if (i + 1 < len(lines) and lines[i].strip().startswith("|")
                and re.fullmatch(r"\|[\s:|-]+\|", lines[i + 1].strip() or "")):
            header = _split_table_row(lines[i])
            j = i + 2
            rows = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(_split_table_row(lines[j]))
                j += 1

            widest = max((len(_unescape_cell(c)) for r in rows for c in r), default=0)
            if widest > WIDE_CELL_CHARS and rows:
                for row in rows:
                    out.extend(_row_as_record(header, row))
                i = j
                continue

        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _row_as_record(header: list[str], row: list[str]) -> list[str]:
    """一列表格 → 一筆帶標題的記錄。"""
    cells = [(header[n] if n < len(header) else f"欄{n + 1}", _unescape_cell(v))
             for n, v in enumerate(row)]
    short = [(k, v) for k, v in cells if v and len(v) <= TITLE_CELL_CHARS]
    long_ = [(k, v) for k, v in cells if v and len(v) > TITLE_CELL_CHARS]

    block = ["", "### " + " · ".join(v for _, v in short[:5]) if short else "### （無標題）"]
    if len(short) > 5:
        block.append("")
        block.append(" ".join(f"**{k}**：{v}" for k, v in short[5:]))
    for key, value in long_:
        block += ["", f"**{key}**：", "", value]
    block.append("")
    return block


def _extract_raw(path: Path, enable_vlm: bool, force_vlm: bool = False,
                 progress=None) -> tuple[str, bool, str]:
    """以 MarkItDown 抽出 Markdown。回傳 (內容, 是否用了 VLM, 錯誤)。"""
    suffix = path.suffix.lower()

    # 純圖片直接走 VLM
    if suffix in IMAGE_TYPES:
        if not enable_vlm:
            return "", False, NEEDS_VLM_MARKER + "這是圖片檔，沒有可讀取的文字層。"
        try:
            description, error, still_short = _describe_once(path.read_bytes())
        except Exception as exc:  # noqa: BLE001
            return "", False, f"讀取圖片失敗：{exc}"
        if error:
            return "", False, f"VLM 解析失敗：{error}"
        if still_short:
            if progress:
                progress(f"    ⚠ {path.name}：重試後仍有省略，內容可能不完整")
            description += "\n\n**注意：本圖的表格可能被模型摘要，內容未必完整。**"
        return description, True, ""

    try:
        from markitdown import MarkItDown

        converter = MarkItDown()
        result = converter.convert(str(path))
        content = (result.text_content or "").strip()
    except Exception as exc:  # noqa: BLE001
        return "", False, f"{type(exc).__name__}: {str(exc)[:200]}"

    # 強制視覺解析。這條路徑處理的是「有文字也有大量圖表」的文件（簡報、
    # 含流程圖的規範書）。既有的掃描件路徑是「沒有文字層才補救」，會直接
    # 略過這種檔案——圖裡的架構圖、流程圖、表格截圖全部丟失，而那往往才是重點。
    #
    # **PDF 與 Office 的取捨不同，因為兩者的 VLM 涵蓋範圍根本不同：**
    #   * PDF   走 `_vlm_read_pdf()`，**逐頁算圖轉錄整份文件** → 文字層是重複的
    #   * Office 走 `_office_images()`，只抽**嵌入的圖片**    → 文字層是唯一正文
    # 所以只有 PDF 能丟文字層，Office 丟了等於把正文刪光。
    if force_vlm and enable_vlm and suffix not in IMAGE_TYPES:
        extra, errors = "", []
        full_transcript = False  # VLM 是否已完整涵蓋整份文件
        if suffix == ".pdf":
            rendered, error, complete = _vlm_read_pdf(path, progress, scanned=False)
            extra = rendered
            full_transcript = complete
            if error:
                errors.append(error)
        else:
            images = _office_images(path)
            if progress:
                progress(f"    找到 {len(images)} 張可解析的圖片")
            if images:
                extra, errors = _describe_images(images, "圖片", progress)
                if len(images) > MAX_VLM_PAGES:
                    note = (f"本檔共 {len(images)} 張圖，"
                            f"只解析了前 {MAX_VLM_PAGES} 張")
                    extra += f"\n\n**注意：{note}。**"
                    if progress:
                        progress(f"    ⚠ {path.name}：{note}")
        if extra:
            # 視覺解析成功了，文字層裡碎掉的表格就沒有保留的理由——
            # VLM 那份才是完整的，殘骸留著只會被檢索撈到、混進答案裡。
            if content:
                content = _strip_broken_tables(content)
            if full_transcript:
                # 每一頁都轉錄成功時，文字層講的是同一件事，而且是品質較差的
                # 那一份：抽取雙欄表格會錯行。實測「原物料總表」被拆成「18」、
                # 「DISPLAY」、「(LCD/OLED/EPD)」三段散落各行，檢索時與正確的
                # VLM 版一起被撈出來，模型分不出哪份可信，答案就把代號與名稱
                # 配錯（出現「MEMORY (ADAPTER/POWER)」這種不存在的類別）。
                merged = extra
            else:
                # 沒有完整涵蓋（超過頁數上限、或有頁面解析失敗）就必須留著
                # 文字層——那些頁的內容只存在於文字層，丟掉就永久遺失了。
                merged = (content + "\n\n" if content else "") + \
                         "---\n\n## 圖片內容（由視覺模型判讀）\n\n" + extra
            return merged, True, "；".join(errors)
        # 沒抽到圖也不算失敗——文字層照樣收下，只是把原因講清楚
        if content:
            return content, False, ""
        return "", False, "；".join(errors) or "強制視覺解析沒有取得任何內容"

    # 內容過少的 PDF 多半是掃描件，把每一頁算圖送 VLM 補救
    if suffix == ".pdf" and _looks_scanned(path, content):
        if not enable_vlm:
            # 有一點文字就先收下，總比什麼都沒有好；沒有的話就明確標記。
            if content.strip():
                return (
                    content
                    + "\n\n（註：本檔案為掃描件，僅擷取到少量文字層內容，"
                    "影像中的文字未被解析。）"
                ), False, ""
            return "", False, NEEDS_VLM_MARKER + "這是掃描件，沒有可讀取的文字層。"

        # 掃描件本來就沒有可用的文字層，完整與否不影響取捨，直接忽略旗標。
        rescued, error, _ = _vlm_read_pdf(path, progress)
        if rescued:
            return rescued, True, ""
        return content, False, error or "掃描件解析失敗"

    if not content:
        return "", False, "解析後沒有任何文字內容"

    return content, False, ""


def _uploaded_name(uploaded) -> str:
    """取得上傳檔案的原始檔名。

    **這裡要同時吃兩種物件。** V1 是 Streamlit 的 `UploadedFile`（屬性叫 `.name`），
    V2 是 FastAPI 的 `UploadFile`（屬性叫 `.filename`，`.name` 根本不存在）。
    這個 service 是從 V1 原封複製過來的，只認 `.name` 的話，
    V2 上傳一律 500 —— 而且錯誤被前端吞掉，看起來像「按了沒反應」。
    """
    return getattr(uploaded, "filename", None) or getattr(uploaded, "name", "") or "未命名"


def _uploaded_bytes(uploaded) -> bytes:
    """讀出上傳檔案的內容，同樣要相容兩種物件。

    Streamlit 用 `.getbuffer()`；FastAPI 的 `UploadFile` 要讀底層的 `.file`
    （同步讀，避免把整條 save_uploads 改成 async）。
    """
    if hasattr(uploaded, "getbuffer"):
        return bytes(uploaded.getbuffer())
    handle = getattr(uploaded, "file", None)
    if handle is not None:
        handle.seek(0)
        return handle.read()
    return uploaded.read()


def save_uploads(root_path: str, files, kb: str | None = None) -> tuple[list[str], list[str]]:
    """把使用者上傳的檔案存進知識庫資料夾。

    指定知識庫時存入該子資料夾，這樣後續的 kb_of() 就能自動歸類；沒指定就放根目錄（通用）。
    回傳 (成功的檔名, 錯誤訊息)。
    """
    root = Path(root_path)
    if not root_path or not root.exists():
        return [], [f"知識庫資料夾不存在：{root_path or '（未設定）'}"]

    if kb:
        ok, why = valid_kb_name(kb)
        if not ok:
            return [], [why]
        if not (root / kb.strip()).is_dir():
            return [], [f"知識庫「{kb}」不存在"]
    target_dir = root / kb.strip() if kb else root
    target_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    errors: list[str] = []

    for uploaded in files or []:
        name = _uploaded_name(uploaded)
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED and suffix not in IMAGE_TYPES:
            errors.append(f"{name}：不支援的格式 {suffix or '（無副檔名）'}")
            continue

        target = target_dir / Path(name).name  # 去掉路徑，只留檔名，避免目錄穿越
        # 同名檔案加序號，不直接覆蓋
        if target.exists():
            stem, counter = target.stem, 2
            while target.exists():
                target = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        try:
            data = _uploaded_bytes(uploaded)
            if not data:
                errors.append(f"{name}：檔案是空的")
                continue
            target.write_bytes(data)
            saved.append(str(target.relative_to(root)).replace("\\", "/"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}：寫入失敗 {exc}")

    return saved, errors


# 掃描時一律略過的資料夾。
#
# **這是必要的防護，不是潔癖。** 使用者很可能把知識庫指到專案根目錄，
# 或不小心把文件放在專案裡——那樣 `venv/` 底下數萬個 `.txt`／`.json`
# 就會被當成知識文件吃進索引（實測掃出 272 個 `top_level.txt`、
# `entry_points.txt`、`LICENSE.txt` 這類套件中繼資料）。
# 這些檔案的副檔名確實在支援清單裡，光靠副檔名擋不掉。
SKIP_DIRS = {
    "venv", ".venv", "env", "node_modules", "__pycache__", ".git", ".idea",
    ".vscode", "site-packages", "dist", "build", ".pytest_cache", ".mypy_cache",
    "models",
}


def scan_folder(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("~$") or path.name.startswith("."):
            continue
        if SKIP_DIRS & set(path.parts):
            continue
        if path.suffix.lower() in SUPPORTED or path.suffix.lower() in IMAGE_TYPES:
            files.append(path)
    return files


def list_library_files(root_path: str) -> list[dict]:
    """列出知識庫資料夾裡的所有檔案，並帶上索引狀態。

    **以磁碟為準，不是以 `documents` 資料表為準。**
    上傳完、還沒建索引之前是沒有 Document 列的，若只查資料表，
    使用者剛上傳的檔案會看不見——那正是「檔案到底存進去沒有」最需要確認的時刻。
    """
    root = Path(root_path)
    if not root_path or not root.exists():
        return []

    with get_session() as session:
        indexed = {
            doc.file_path: doc
            for doc in session.query(Document).all()
        }
        rows = []
        for path in sorted(scan_folder(root)):
            doc = indexed.get(str(path))
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            rows.append({
                # 相對路徑同時是顯示用的識別與刪除用的鍵：
                # 未索引的檔案沒有 doc_id，用 id 當鍵就少一半檔案刪不掉。
                "rel_path": str(path.relative_to(root)).replace("\\", "/"),
                "file_name": path.name,
                "file_type": path.suffix.lower().lstrip("."),
                "file_size": size,
                "kb": doc.kb if doc else kb_of(path, root),
                "indexed": doc is not None and doc.status == DOC_INDEXED,
                "status": doc.status if doc else "未索引",
                "chunk_count": doc.chunk_count if doc else 0,
                "used_vlm": bool(doc.used_vlm) if doc else False,
                "force_vlm": get_doc_option(str(path)),
                "indexed_at": doc.indexed_at.strftime("%Y-%m-%d %H:%M") if doc and doc.indexed_at else "",
            })
    return rows


def delete_library_file(root_path: str, rel_path: str) -> tuple[bool, str]:
    """刪除一個知識庫檔案，連同它在資料庫裡的所有痕跡。

    要清乾淨的有五處，少一處都會留下問題：
      1. `chunks`          —— 切片本體
      2. `vec_chunks`      —— sqlite-vec 虛擬表，**不會因為 chunks 被刪就跟著消失**
      3. `chunk_keywords`  —— 關鍵字備份。留著的話，日後上傳同檔名的新檔案時，
                              `restore_keywords()` 會把舊關鍵字套到新內容上
      4. `ingest_errors`   —— 否則「解析狀態」頁會一直顯示一個已經不存在的檔案
      5. 磁碟上的檔案本身 —— 只刪資料庫的話，下次建索引又會被掃回來
    """
    root = Path(root_path)
    if not root_path or not root.exists():
        return False, "知識庫資料夾不存在"

    # 路徑遍歷防護：rel_path 來自前端，不可信。
    # 必須確認解析後仍落在知識庫資料夾內，否則 `../../` 就能刪掉專案外的檔案。
    try:
        target = (root / rel_path).resolve()
        target.relative_to(root.resolve())
    except (ValueError, OSError):
        return False, "路徑不合法"

    if not target.is_file():
        return False, "檔案不存在"

    with get_session() as session:
        doc = session.query(Document).filter(Document.file_path == str(target)).first()
        doc_id = doc.id if doc else None

    if doc_id is not None:
        _delete_chunks(doc_id)  # 同時處理 chunks 與 vec_chunks
        with get_session() as session:
            session.query(Document).filter(Document.id == doc_id).delete()
            session.commit()

    with get_session() as session:
        session.query(ChunkKeyword).filter(ChunkKeyword.file_path == str(target)).delete()
        session.query(IngestError).filter(IngestError.file_path == str(target)).delete()
        session.commit()

    try:
        target.unlink()
    except OSError as exc:
        # 資料庫已經清掉了，檔案沒刪成要講清楚——
        # 否則使用者會以為刪好了，下次建索引它又冒出來。
        return False, f"資料庫紀錄已清除，但檔案刪除失敗：{exc}"

    return True, f"已刪除 {rel_path}"


def embed_text(content: str, keywords: str = "") -> str:
    """組出真正送進 embedding 的文字。

    關鍵字是管理員為了調整檢索命中率而補的（同義詞、口語說法、代號），
    **只影響向量，不影響顯示與作答**——來源卡片與 LLM 看到的仍是原文。
    """
    keywords = (keywords or "").strip()
    if not keywords:
        return content
    return f"{content}\n\n[檢索關鍵字] {keywords}"


def save_keywords(file_path: str, seq: int, content: str, keywords: str) -> None:
    """把關鍵字另存一份，讓它能在重新索引後被套回去。"""
    from models import ChunkKeyword

    head = content[:120]
    with get_session() as session:
        row = (
            session.query(ChunkKeyword)
            .filter(ChunkKeyword.file_path == file_path, ChunkKeyword.seq == seq)
            .first()
        )
        if not keywords.strip():
            if row:
                session.delete(row)
                session.commit()
            return
        if row:
            row.keywords, row.content_head = keywords, head
            row.updated_at = datetime.now()
        else:
            session.add(ChunkKeyword(file_path=file_path, seq=seq,
                                     content_head=head, keywords=keywords))
        session.commit()


def save_chunk_override(file_path: str, seq: int, original: str,
                        keywords: str, edited: str) -> None:
    """把管理員編輯過的切片內容另存一份，讓它在重新索引後能被套回去。

    指紋用**原始解析內容**的前 120 字，不是編輯後的內容——
    重新索引時拿到的是重新解析的結果（等同原文），
    用編輯後的內容當指紋會永遠對不上，等於還原機制完全失效。
    """
    from models import ChunkKeyword

    head = original[:120]
    with get_session() as session:
        row = (
            session.query(ChunkKeyword)
            .filter(ChunkKeyword.file_path == file_path, ChunkKeyword.seq == seq)
            .first()
        )
        if not edited.strip() and not keywords.strip():
            if row:
                session.delete(row)
                session.commit()
            return
        if row:
            row.edited_content, row.content_head = edited, head
            row.keywords = keywords
            row.updated_at = datetime.now()
        else:
            session.add(ChunkKeyword(file_path=file_path, seq=seq, content_head=head,
                                     keywords=keywords, edited_content=edited))
        session.commit()


def restore_keywords(file_path: str, pieces: list[dict]) -> dict[int, tuple[str, str]]:
    """重新索引後把先前存下的關鍵字**與編輯過的內容**套回去。

    回傳 `{seq: (keywords, edited_content)}`，`edited_content` 為空字串代表沒編輯過。

    切片每次重建都會換新 id，人工調整若不另存就會無聲消失。
    以 `content_head` 當指紋：**文件內容變了就不套用**，
    避免把關鍵字或編輯結果接到不相干的段落上。
    """
    from models import ChunkKeyword

    with get_session() as session:
        rows = (
            session.query(ChunkKeyword)
            .filter(ChunkKeyword.file_path == file_path)
            .all()
        )
        saved = {r.seq: (r.keywords, r.content_head, r.edited_content or "") for r in rows}

    restored: dict[int, tuple[str, str]] = {}
    for i, piece in enumerate(pieces, start=1):
        if i in saved:
            keywords, head, edited = saved[i]
            if piece["content"][:120] == head:
                restored[i] = (keywords, edited)
    return restored


# 分桶用的開頭字數。重複迴圈吐出來的區塊開頭必然相同，用它分桶就能把
# 比對限制在同一桶內，不必 n² 兩兩比。
DEDUPE_HEAD_CHARS = 40


def _dedupe_pieces(pieces: list[dict], progress=None, file_name: str = "") -> list[dict]:
    """丟掉內容被其他切片完全包含的重複切片。

    VLM 逐頁轉錄時會卡進**重複迴圈**，把同一個區塊連續吐出好幾份。實測一份
    16 頁 PDF 的第 10 頁產生 6 份「下載紀錄」切片（5 份逐字節相同、第 6 份是
    前綴），2222 字，佔整個索引的 20.7%。

    重複內容對檢索是毒藥：同一段存 6 份，被撈中的機率就是單份的 6 倍，會把
    真正有用的段落擠出候選名單——實測「料件有哪些種類」的來源清單裡有 4 個
    名額被同一份下載紀錄佔走。

    既有的 `_looks_elided()` 抓的是「等等」式的**省略**，抓不到相反方向的
    **重複**，兩者是同一類問題的兩面。

    判準用**包含**而不是相等：A 的內容若完全出現在 B 裡面，丟掉 A 不會有
    任何資訊消失，所以是安全的。順序不保證長的先出現（截斷的那份可能落在
    前面），因此兩個方向都要處理。

    **涵蓋範圍**：抓的是「開頭相同的重複區塊」，不是任意的子字串包含——
    後者要 n² 兩兩比，而重複迴圈產生的區塊開頭一定相同，不值得為此付代價。
    漏掉只是少去一份重複，不會誤刪。
    """
    records: list[tuple[dict, str]] = []
    buckets: dict[str, list[int]] = {}
    shorts: list[int] = []
    dead: set[int] = set()

    for piece in pieces:
        # 正規化空白再比，避免只差一個換行就被當成不同內容
        body = " ".join((piece.get("content") or "").split())
        index = len(records)
        records.append((piece, body))
        if not body:
            continue

        if len(body) >= DEDUPE_HEAD_CHARS:
            bucket = buckets.setdefault(body[:DEDUPE_HEAD_CHARS], [])
            # 短段落沒有可靠的桶可待，長段落要另外跟它們比一次
            pool = bucket + shorts
        else:
            # 比分桶鍵還短的段落，它的全長比鍵還小，切不出對得上的鍵——
            # 只能跟全部比。這種段落多半是被切斷的尾巴，數量少，成本可接受。
            bucket = None
            pool = list(range(index))

        alive = [i for i in pool if i not in dead and records[i][1]]
        if any(body in records[i][1] for i in alive):
            dead.add(index)          # 這段已被留下的某段涵蓋
            continue
        for i in alive:
            if records[i][1] in body:
                dead.add(i)          # 反過來，這段更完整，換掉舊的
        (shorts if bucket is None else bucket).append(index)

    if not dead:
        return pieces

    kept = [p for i, (p, _) in enumerate(records) if i not in dead]
    if progress:
        progress(f"    ⚠ {file_name}：去除 {len(dead)} 個重複切片"
                 f"（{len(pieces)} → {len(kept)}），多半是視覺模型卡在重複迴圈")
    return kept


def _delete_chunks(doc_id: int) -> None:
    """刪除文件的所有切片與其向量。"""
    with get_session() as session:
        ids = [c.id for c in session.query(Chunk).filter(Chunk.doc_id == doc_id).all()]
        session.query(Chunk).filter(Chunk.doc_id == doc_id).delete()
        session.commit()

    if ids:
        conn = raw_connection()
        try:
            conn.executemany("DELETE FROM vec_chunks WHERE chunk_id = ?", [(i,) for i in ids])
            conn.commit()
        finally:
            conn.close()


def _index_document(path: Path, root: Path, stats: IngestStats, enable_vlm: bool,
                    progress=None) -> None:
    size = path.stat().st_size
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    digest = file_hash(path)

    force_vlm = get_doc_option(str(path))

    with get_session() as session:
        doc = session.query(Document).filter(Document.file_path == str(path)).first()

        # 檔案沒變就跳過——但**「剛被指定強制視覺解析」也算變更**。
        # 只比對 sha256 的話，使用者對既有文件勾了「強制視覺解析」再按增量更新，
        # 會因為檔案內容沒動而被跳過，看起來像那個開關完全沒作用。
        #
        # `doc.content` 為空也不能跳過：那是舊版索引留下的資料（該欄位是後來才加的）。
        # 少了這條，閱讀頁永遠讀不到存下來的內容，只能回退去重新解析。
        if (doc and doc.sha256 == digest and doc.status == DOC_INDEXED
                and bool(doc.used_vlm) == force_vlm and doc.content):
            # 內容沒變可以跳過解析，但**分類要跟著資料夾走**：使用者用檔案總管
            # 把檔案搬到另一個知識庫時 sha256 不會變，少了這一行 DB 會一直
            # 記著舊分類，檢索範圍就過濾錯了。資料夾是唯一的真相來源。
            actual_kb = kb_of(path, root)
            if doc.kb != actual_kb:
                doc.kb = actual_kb
                session.commit()
            stats.skipped += 1
            return

        is_update = doc is not None
        if not doc:
            doc = Document(file_path=str(path))
            session.add(doc)

        doc.file_name = path.name
        doc.file_type = path.suffix.lower().lstrip(".")
        doc.file_size = size
        doc.sha256 = digest
        doc.modified_at = mtime
        doc.kb = kb_of(path, root)
        session.commit()
        doc_id = doc.id

    if is_update:
        _delete_chunks(doc_id)

    if force_vlm and enable_vlm and progress:
        progress(f"  {path.name}：強制視覺解析中，這一份會比較久…")
    content, used_vlm, error = extract_markdown(path, enable_vlm, force_vlm, progress)

    # **有內容就收下，即使過程中出過錯。**
    #
    # 視覺解析是一張圖一張圖跑的，32 張裡有一張逾時是常態。原本的
    # `if error:` 不看有沒有內容，一律把整份文件標成 failed——結果是
    # 31 張成功的描述加上完整文字層全部丟掉，使用者看到的是「解析失敗」，
    # 完全不知道其實只差一張圖。
    #
    # 這個 bug 一直都在，只是 MAX_VLM_PAGES 還是 10 的時候跑不到第 27 張圖，
    # 所以碰不到。真正的失敗條件是「什麼都沒抽到」，不是「出過錯」。
    if error and content:
        with get_session() as session:
            session.add(
                IngestError(
                    file_path=str(path), file_name=path.name,
                    error_type="partial",
                    message=f"部分內容未解析成功，其餘已納入索引：{error}",
                )
            )
            session.commit()
        if progress:
            progress(f"    ⚠ {path.name}：部分內容未解析成功，其餘已納入索引")
        error = ""

    if error:
        # 「缺少視覺模型」跟「檔案解析失敗」是兩回事，分開記錄。
        # 前者裝個模型重跑就好，不該混在解析錯誤清單裡讓人以為檔案有問題。
        needs_vlm = error.startswith(NEEDS_VLM_MARKER)
        with get_session() as session:
            doc = session.get(Document, doc_id)
            doc.status = DOC_FAILED
            session.add(
                IngestError(
                    file_path=str(path),
                    file_name=path.name,
                    error_type="needs_vlm" if needs_vlm else "parse_failed",
                    message=(
                        error.replace(NEEDS_VLM_MARKER, "")
                        + f"　安裝視覺模型後重新索引即可讀取："
                        f"ollama pull {RECOMMENDED_VLM}"
                        if needs_vlm else error
                    ),
                )
            )
            session.commit()
        if needs_vlm:
            stats.needs_vlm.append(path.name)
        else:
            stats.failed += 1
        return

    size_setting = get_int_setting("chunk_size", 500)
    overlap = get_int_setting("chunk_overlap", 80)
    pieces = chunk_text(content, size_setting, overlap, base_locator=path.name)
    # 去重要放在 restore_keywords 之前：切片編號會因為去重而位移，而
    # restore_keywords 本來就用 content_head 指紋擋住對不上的情況。
    pieces = _dedupe_pieces(pieces, progress, path.name)

    if not pieces:
        with get_session() as session:
            doc = session.get(Document, doc_id)
            doc.status = DOC_FAILED
            session.add(
                IngestError(
                    file_path=str(path), file_name=path.name,
                    error_type="empty", message="切片後沒有任何內容",
                )
            )
            session.commit()
        stats.failed += 1
        return

    # 把先前存下的檢索關鍵字**與編輯過的內容**套回來。
    # 切片每次重建都換新 id，不還原的話管理員的調校會無聲消失。
    restored = restore_keywords(str(path), pieces)
    edited_count = sum(1 for _, edited in restored.values() if edited)
    if restored:
        parts = [f"還原 {len(restored)} 組檢索關鍵字"] if any(k for k, _ in restored.values()) else []
        if edited_count:
            parts.append(f"{edited_count} 段編輯過的內容")
        if parts:
            stats.messages.append(f"  ↳ {path.name}：{'、'.join(parts)}")

    # 寫入切片
    chunk_ids: list[int] = []
    with get_session() as session:
        for seq, piece in enumerate(pieces, start=1):
            keywords, edited = restored.get(seq, ("", ""))
            # 編輯過的內容要蓋掉重新解析的結果，同時把原文留在 original_content，
            # 否則「還原」按鈕會沒有東西可以還原。
            body = edited or piece["content"]
            chunk = Chunk(
                doc_id=doc_id,
                seq=seq,
                content=body,
                original_content=piece["content"] if edited else "",
                locator=f"{piece['locator']} #{seq}",
                char_count=len(body),
                keywords=keywords,
            )
            session.add(chunk)
            session.flush()
            chunk_ids.append(chunk.id)
        session.commit()

    # 批次產生向量，一次 16 筆避免記憶體壓力
    conn = raw_connection()
    try:
        for start in range(0, len(pieces), 16):
            batch = pieces[start : start + 16]
            ids = chunk_ids[start : start + 16]
            # **向量要用編輯後的內容算。**
            # 用 piece["content"]（重新解析的原文）算，檢索仍會命中已經被
            # 管理員清掉的噪音，等於編輯白做。
            vectors, error = ollama_client.embed([
                embed_text(
                    restored.get(start + n + 1, ("", ""))[1] or p["content"],
                    restored.get(start + n + 1, ("", ""))[0],
                )
                for n, p in enumerate(batch)
            ])
            if error:
                raise RuntimeError(f"產生向量失敗：{error}")
            conn.executemany(
                "INSERT OR REPLACE INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                [(cid, serialize(vec)) for cid, vec in zip(ids, vectors)],
            )
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        with get_session() as session:
            doc = session.get(Document, doc_id)
            doc.status = DOC_FAILED
            session.add(
                IngestError(
                    file_path=str(path), file_name=path.name,
                    error_type="embed_failed", message=str(exc)[:300],
                )
            )
            session.commit()
        stats.failed += 1
        return
    finally:
        conn.close()

    with get_session() as session:
        doc = session.get(Document, doc_id)
        doc.status = DOC_INDEXED
        doc.indexed_at = datetime.now()
        doc.chunk_count = len(pieces)
        doc.used_vlm = used_vlm
        # 存下這次解析的結果供閱讀頁使用。
        # 閱讀時重新解析會拿到不同的東西——解析結果取決於當下的 VLM 設定，
        # 而那份設定隨時可能被改。存下來才能保證「看到的」就是「AI 讀到的」。
        doc.content = content
        session.commit()

    stats.chunks += len(pieces)
    if used_vlm:
        stats.vlm_used += 1
    if is_update:
        stats.updated += 1
    else:
        stats.new += 1


def ingest(root_path: str, full_rebuild: bool = False, progress=None) -> IngestStats:
    """執行索引。full_rebuild=True 時先清空既有索引。"""
    stats = IngestStats()

    def report(message: str) -> None:
        stats.messages.append(message)
        if progress:
            progress(message)

    # **空字串一定要先擋掉。** `Path("")` 在 Windows 等同當前目錄，
    # `.exists()` 與 `.is_dir()` 都回 True，所以下面那道檢查完全擋不住——
    # 實測結果是把整個專案資料夾（含 venv 的 3 萬個檔案）當成知識庫掃進去，
    # 索引出 272 個 `top_level.txt`、`entry_points.txt` 這類套件中繼資料。
    if not root_path.strip():
        report("尚未設定知識庫資料夾。請先在上方指定要索引的資料夾路徑。")
        return stats

    root = Path(root_path)
    if not root.exists() or not root.is_dir():
        report(f"資料夾不存在：{root}")
        return stats

    status = ollama_client.check_status()
    if not status.alive:
        report(f"無法建立索引：{status.message}")
        return stats

    # 先確認向量表的維度跟目前的 embedding 模型對得上。
    #
    # 維度在建表時就固定了，換模型若維度不同，寫入會全數失敗。
    # 而全量重建是「先清空、再寫入」，不先檢查的話會留下一個空知識庫——
    # 實測換成 4096 維的 qwen3-embedding:8b 時 12/12 全失敗、切片數歸零。
    probe, probe_error = ollama_client.embed(["維度探測"])
    if probe_error or not probe:
        report(f"無法取得 embedding 模型的向量維度：{probe_error or '回傳為空'}")
        return stats
    dim = len(probe[0])

    if vec_table_dim() != dim:
        if not full_rebuild:
            report(
                f"embedding 模型的維度（{dim}）與現有索引（{vec_table_dim()}）不符，"
                f"無法做增量更新。請改用「全量重建」。"
            )
            return stats
        report(f"embedding 模型維度為 {dim}，正在重建向量表...")
        ensure_vec_table(dim)

    if full_rebuild:
        report("正在清空既有索引...")
        with get_session() as session:
            session.query(Chunk).delete()
            session.query(Document).delete()
            session.query(IngestError).delete()
            session.commit()
        conn = raw_connection()
        try:
            conn.execute("DELETE FROM vec_chunks")
            conn.commit()
        finally:
            conn.close()

    files = scan_folder(root)
    stats.scanned = len(files)
    report(f"找到 {len(files)} 個可解析的檔案")

    enable_vlm = get_setting("enable_vlm", "1") == "1"

    for i, path in enumerate(files, start=1):
        report(f"[{i}/{len(files)}] {path.name}")
        try:
            _index_document(path, root, stats, enable_vlm, report)
        except Exception as exc:  # noqa: BLE001 - 單一檔案失敗不中斷整批
            stats.failed += 1
            with get_session() as session:
                session.add(
                    IngestError(
                        file_path=str(path), file_name=path.name,
                        error_type="unexpected", message=f"{type(exc).__name__}: {exc}"[:300],
                    )
                )
                session.commit()

    # 清理已不存在的檔案。
    #
    # **這一段的前提是 `files` 涵蓋整個資料夾。** 曾經為了「只重建單一檔案」
    # 把 files 過濾成一個元素，結果「不在 files 裡」等於「其他每一份文件」，
    # 一次清掉 12 份文件的索引。若日後要加單檔模式，這裡必須一併跳過。
    with get_session() as session:
        existing = {str(p) for p in files}
        stale = [d for d in session.query(Document).all() if d.file_path not in existing]
        for doc in stale:
            _delete_chunks(doc.id)
            session.delete(doc)
        if stale:
            session.commit()
            report(f"清理了 {len(stale)} 筆已不存在的檔案索引")

    report(
        f"完成：新增 {stats.new}、更新 {stats.updated}、"
        f"略過 {stats.skipped}、失敗 {stats.failed}，共 {stats.chunks} 個切片"
    )
    if stats.needs_vlm:
        report(
            f"另有 {len(stats.needs_vlm)} 個圖片型檔案只保留了檔名與階段資訊，"
            f"內容未解析（需要視覺模型）"
        )
    return stats
