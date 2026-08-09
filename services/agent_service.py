"""以工具調用驅動的問答流程。

與傳統 RAG 的差別在於**誰決定要查什麼**：

    傳統：程式先檢索 → 把結果塞給 LLM → LLM 只能就這批資料作答。
    這裡：LLM 自己決定檢索什麼、看完覺得不夠就換關鍵詞再查，
          真的查不到才說沒有。

這同時解決了三件事：

1. **追問**。「還有嗎」的語意全在前文裡，模型看得到對話紀錄，
   自然會把 query 補成「常見失誤有哪些」再去查，不需要另外做問題改寫。
2. **關鍵詞不對**。使用者的口語措辭常與文件用語對不上
   （「踩過什麼雷」vs「Lesson Learnt」），模型第一次查不到會自己換詞重試。
3. **誠實的沒有**。「查無資訊」是在模型嘗試過數種問法之後才說出口的，
   而不是第一次檢索距離偏高就放棄。

模型不支援工具調用時，`chat.py` 會退回 `rag_service` 的傳統單次檢索路徑。
"""

from __future__ import annotations

import json
import re

from services import ollama_client, rag_service

# 即使帶了 think=False，gemma4 偶爾仍會把推理通道的標記漏進正文，
# 開頭出現 "thought" 加上 "<channel|>" 之類的字樣。
# 出現機率不高但看得很清楚，逐字輸出時會直接秀在使用者面前。
_LEAK_PREFIX = re.compile(r"^\s*(thought)?\s*<channel\|?[^>]*>\s*", re.IGNORECASE)
_LEAK_INLINE = re.compile(r"<\|?/?channel\|?[^>]*>", re.IGNORECASE)

# 先攔住這麼多字元再開始逐字輸出，讓上面的標記有機會被完整看到並清掉。
# 標記約 20 字元，取 48 留餘裕；對觀感的影響是首字晚零點幾秒出現。
_LEAK_GUARD_CHARS = 48


def _strip_leak(text: str) -> str:
    return _LEAK_INLINE.sub("", _LEAK_PREFIX.sub("", text)).lstrip()

# 一個問題最多讓模型查幾次。
#
# 設 3 是實測折衷：多數問題一次就夠，換過一次關鍵詞仍找不到的，
# 再查第三次幾乎不會有新東西，只是讓使用者多等。
# 用完額度後最後一輪會把 tools 拿掉，逼模型用現有資料作答或誠實說沒有。
MAX_SEARCHES = 3

# 列舉型問題（「有哪些」「列出全部」）的檢索額度。
#
# **這類問題用 top_k=6 的單次檢索必然不完整。** 實測一份 400K 字的測試計畫：
# 同一主題底下有 6 個獨立章節、每章又有 7 個小節，
# 6 個切片連章節數都涵蓋不完——答案漏了其中 2 個章節。
# 而單獨問那 2 個章節的名稱時它們都是第 1 名，證明撈得到，只是沒被撈。
#
# 提高額度只是必要條件——但**光靠額度與提示沒有用**。
# 實測：把額度提到 5 次、在工具回傳裡明確要求「請再換關鍵詞查 1–2 次」、
# 系統指示也加了規則，gemma4:12b 仍然只查一次就作答。
# 靠提示要求模型多查是不可靠的，所以真正的修法是下面的 ENUMERATION_TOP_K。
MAX_SEARCHES_ENUMERATION = 5

# 列舉型問題改用較大的檢索量。**這是不依賴模型配合的做法。**
#
# 實測「Pressure test 有哪些種類」對上 6 個壓力測試章節的涵蓋率：
#     top_k=6 → 2/6　　top_k=12 → 2/6　　top_k=20 → 4/6　　top_k=30 → **6/6**
#
# 30 段約 15,000 字，對 gemma4 的 131k context 沒有壓力。
# 代價是精準度會被稀釋（撈進較多沾邊的段落），所以只用在列舉型問題上。
ENUMERATION_TOP_K = 30

# 判斷是不是列舉型問題。寧可誤判成列舉（多查幾次只是慢一點），
# 也不要漏判（漏判的代價是答案不完整，而使用者看不出來少了什麼）。
_ENUMERATION_HINTS = re.compile(
    r"有哪些|哪些|列出|列舉|全部|所有|各項|清單|一覽|完整|總共|幾種|幾個|"
    r"\ball\b|\blist\b|\bevery\b|\bwhat kinds\b",
    re.IGNORECASE,
)


def _is_enumeration(question: str) -> bool:
    return bool(_ENUMERATION_HINTS.search(question or ""))


_ENUMERATION_NUDGE = (
    "\n\n**這是列舉型問題，目前這批結果很可能不完整。**\n"
    "一次檢索只會回傳最相關的數段，不足以窮盡所有項目。"
    "請至少再用**不同的關鍵詞**檢索 1–2 次（例如換成同義詞、上位詞、"
    "或你在上面看到的相關名詞），確認沒有遺漏，再開始作答。\n"
    "作答時若仍不確定是否完整，明講「可能還有其他項目未列出」，不要假裝窮盡。"
)

# 對話紀錄帶幾輪。助理的回答動輒上千字，全帶會把檢索結果擠掉。
HISTORY_TURNS = 3
HISTORY_ANSWER_CHARS = 400

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "在這個知識庫中做語意檢索，"
            "回傳最相關的文件片段。這是取得公司流程資訊的唯一管道。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "檢索用的問題或關鍵詞。必須能獨立理解——"
                        "把對話中省略的主詞補回去，不要直接送出「還有嗎」這種句子。"
                    ),
                },
                "stage_code": {
                    "type": "string",
                    "description": (
                        "限定檢索的階段代碼，可用值："
                        "Concept、Plan、EVT、DVT、PVT、MP。不確定就不要填。"
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

SYSTEM_INSTRUCTION = (
    "你是這個知識庫的助理。\n\n"
    "**你對知識庫的主題沒有任何內建知識。** 任何與知識庫內容有關的問題，"
    "都必須先呼叫 search_knowledge_base 查詢，"
    "絕對不可以憑自己的印象作答。\n\n"
    "規則：\n"
    "1. 檢索用的 query 要能獨立理解。對話中省略的主詞要補回去——\n"
    "   例如前面在談「常見失誤」，使用者問「還有嗎」，\n"
    "   query 應該是「常見失誤 Lesson Learnt」而不是「還有嗎」。\n"
    "2. 檢索結果不足以回答時，**換個說法再查一次**。\n"
    "   使用者的口語常與文件用語對不上，例如「踩過什麼雷」在文件裡\n"
    "   可能寫成「Lesson Learnt」或「常見疏漏」。也可以把問題拆小再查。\n"
    "2-1. **列舉型問題（「有哪些」「列出全部」）一定要查多次。**\n"
    "   一次檢索只回傳最相關的幾段，涵蓋不了所有項目。要換不同關鍵詞\n"
    "   （同義詞、上位詞、你在結果裡看到的相關名詞）多查幾輪再作答。\n"
    "   若無法確定已窮盡，明講「可能還有其他項目未列出」，不要假裝完整。\n"
    "3. **先前的對話只能用來理解使用者在問什麼，不是事實來源。**\n"
    "   即使上一則回答裡已經有相關內容，這一輪仍然必須重新檢索；\n"
    "   沒有本輪的檢索結果就不准標註任何來源編號。\n"
    "4. 只依據本輪檢索到的內容作答，每個結論後面標註來源編號 [1]、[2]。\n"
    "   來源編號沿用檢索結果給的編號，不要自己重新編、也不要沿用上一輪的編號。\n"
    "4-1. 檢索結果依文件分組呈現。**不可把不同文件的數值或條件合併成同一組結論**——\n"
    "   不同文件可能是不同的標準、版本或適用範圍。若各文件說法不一致，\n"
    "   分別說明各自出自哪份文件；與問題無關的文件直接忽略，不必勉強使用。\n"
    "4-1-1. **同一份文件內也依章節分組，章節之間同樣不可互相借用數值。**\n"
    "   某個章節底下沒有列出規格數值時，就說那個章節未提供，\n"
    "   **不可以拿隔壁章節的數值填上去**。條件、規格、判定標準都適用這一條。\n"
    "4-2. **編號要標在該內容真正出處的那一段上。** 檢索結果中標了 [n] 的段落才是來源；\n"
    "   標示為「前後文」的段落只是為了讓你讀懂上下文，**不可標註為來源**。\n"
    "   若某個數值出自標了 [6] 的段落，就標 [6]，不要因為它在 [2] 附近就標成 [2]。\n"
    "5. 換過幾種問法仍然查不到時，明確說「知識庫中查無足夠資訊」，"
    "不要用自己的知識補足。\n"
    "6. 與知識庫主題明顯無關的問題（閒聊、寫作、投資…）直接婉拒，不需要檢索。\n"
    "7. 以繁體中文回答，條列說明，語氣專業簡潔。\n"
    "8. **不要使用 LaTeX 數學語法。** 溫度直接寫 25°C，不要寫 $25^\\circ\\text{C}$；"
    "專有名詞直接寫 Tjmax，不要寫 $\\text{Tjmax}$。這些內容是規格與數值，不是數學式。"
)

FORCE_SEARCH_NUDGE = (
    "你還沒有檢索知識庫。請先呼叫 search_knowledge_base 查詢，"
    "再依照檢索結果回答；若查完確實沒有相關內容，再說查無資訊。"
)


def _history_messages(history: list[dict] | None) -> list[dict]:
    if not history:
        return []
    messages = []
    for record in history[-(HISTORY_TURNS * 2):]:
        content = record["content"]
        if record["role"] != "user" and len(content) > HISTORY_ANSWER_CHARS:
            content = content[:HISTORY_ANSWER_CHARS] + "…（略）"
        messages.append({"role": record["role"], "content": content})
    return messages


class _Citations:
    """跨多次檢索維持穩定的來源編號。

    同一個切片在第一次與第三次檢索都出現時必須是同一個編號，
    否則模型標的 [2] 會指向不同的東西。
    """

    def __init__(self) -> None:
        self._index: dict[int, int] = {}

    def number(self, chunk_id: int) -> int:
        if chunk_id not in self._index:
            self._index[chunk_id] = len(self._index) + 1
        return self._index[chunk_id]

    @property
    def chunk_ids(self) -> list[int]:
        return sorted(self._index, key=self._index.get)


def _run_search(query: str, stage_code: str | None, citations: _Citations,
                wide: bool = False, enumeration: bool = False,
                remaining: int = 0) -> tuple[str, int]:
    """執行檢索並排版成模型讀得懂的文字。回傳 (內容, 命中數)。"""
    chunks, error = rag_service.retrieve(
        query, stage_code, top_k=ENUMERATION_TOP_K if enumeration else None
    )
    if error:
        return f"檢索失敗：{error}", 0
    if not chunks:
        return "沒有找到任何內容。請換不同的關鍵詞再查一次。", 0

    # 引註編號一定要在擴展前先配好：`citations.number()` 是跨輪次累積的，
    # 而來源面板是靠這個編號對應回切片的。順序也不能因為分組而改變。
    numbers = {c.chunk_id: citations.number(c.chunk_id) for c in chunks}

    # 命中段前後各補一段，讓模型看得到完整的上下文。
    # 相鄰的命中會合併成同一個區段，不會重複餵。
    # 列舉型問題已經撈了 30 段，再補前後文會讓上下文暴漲且稀釋重點，
    # 除非使用者明確按了「更多脈絡」。
    context_blocks = rag_service.build_context_blocks(
        chunks, wide=wide, expand=(wide or not enumeration)
    )
    number_of = {c.chunk_id: numbers[c.chunk_id] for c in chunks}
    seq_index = {i + 1: c.chunk_id for i, c in enumerate(chunks)}

    # 依文件分組。**平鋪呈現時模型會把不同文件的內容融成一份看似完整的規格**——
    # 實測發生過：一份公司內部測試計畫與一份 MIL-STD 問答紀錄被合成同一組參數，
    # 而那組參數在任何一份文件裡都不存在。
    grouped: dict[str, list] = {}
    for block in context_blocks:
        grouped.setdefault(block.file_name, []).append(block)

    sections = []
    for file_name, items in grouped.items():
        stage = f"（{items[0].stage_code} 階段）" if items[0].stage_code else ""

        # 文件底下再依**章節**分組。
        #
        # 只依文件分組時，同一份大型 PDF 的 6 個章節會平鋪成一串，模型得自己
        # 從 `28.2.6` 這種編號回推「這是哪個測試的規格」——實測它對錯了：
        # 把某章節的規格數值掛到相鄰章節頭上。
        # 答案通順、標了來源，數字卻是錯的。
        #
        # 分組後歸屬由版面決定，不需要模型推論。分組鍵完全從 locator 推導
        # （見 `rag_service.section_of`），對所有格式一致，沒有麵包屑就退回單一組。
        by_section: dict[str, list] = {}
        for block in items:
            by_section.setdefault(rag_service.section_of(block.locator), []).append(block)

        chunks_of_section = []
        for section, blocks in by_section.items():
            body_parts = []
            for block in blocks:
                # **逐段標號，不是整塊掛一個編號。**
                # 整塊只掛一個編號的話，模型會把鄰居段的內容也標成那個來源——
                # 實測發生過：壓力測試機型號出自 28.1.2，卻被標成 28.1.1 的編號。
                for idx, locator, text in block.segments:
                    if idx is not None and idx in seq_index:
                        body_parts.append(f"[{number_of[seq_index[idx]]}] {locator}\n{text}")
                    else:
                        body_parts.append("（前後文，僅供理解，不可標註為來源）\n" + text)
            head = f"#### 章節：{section}\n（以下內容全部屬於這個章節，不可用於其他章節）\n\n" if section else ""
            chunks_of_section.append(head + "\n\n".join(body_parts))

        sections.append(
            f"### 文件：{file_name}{stage}\n\n" + "\n\n".join(chunks_of_section)
        )
    body = "\n\n---\n\n".join(sections)

    warning = ""
    if len(grouped) > 1:
        warning = (
            f"\n\n**以上內容來自 {len(grouped)} 份不同文件。**"
            "不同文件可能是不同的標準、版本或適用範圍，"
            "**不可把不同文件的數值或條件合併成同一組結論**。"
            "若說法不一致，分別說明各自的出處；與問題無關的文件直接忽略即可。"
        )

    # 列舉型問題還有額度時，明確要求續查。額度用完就不催了——
    # 催了也查不了，只會讓模型多說一句沒有意義的話。
    tail = (
        _ENUMERATION_NUDGE if (enumeration and remaining > 0)
        else "\n\n（若以上內容不足以回答使用者的問題，請換個關鍵詞再查一次。）"
    )
    return f"檢索到 {len(chunks)} 段內容：\n\n{body}{warning}{tail}", len(chunks)


# 逐章節作答時最多處理幾個章節。
#
# 每個章節一次 LLM 呼叫（約 10–20 秒），8 個就要兩分鐘。超過的章節會在
# 結尾明確列出「還有這些相關章節未展開」，而不是靜靜丟掉。
MAX_SECTIONS = 8

# 章節少於這個數就不值得拆——一次問完更快，而且拆了也沒有混淆可言。
MIN_SECTIONS_TO_SPLIT = 3

SECTION_SYSTEM = (
    "你是知識庫助理。**只依據使用者提供的這一個章節的內容作答。**\n"
    "規則：\n"
    "1. 這段內容全部屬於同一個章節。不要提到、不要推測其他章節的內容。\n"
    "2. 問題問到的項目若本章節沒有提供，明講「本章節未提供」，"
    "**絕對不可以從別處推測或填補數值**。\n"
    "3. 每個結論後面標註來源編號 [n]，沿用內容裡給的編號，不要自己編。\n"
    "4. 回答要短。條列，只講與問題直接相關的內容，不要重述整個章節。\n"
    "5. 繁體中文。不要用 LaTeX。"
)


def _answer_by_section(question: str, stage_code: str | None,
                       citations: _Citations, wide: bool):
    """列舉／彙整型問題：**依章節拆開，一節問一次，程式負責組裝。**

    為什麼要這樣做：這類問題的失敗模式是「模型讀了 30 段卻只列出 3 個」。
    今天在提示層試過六種修法全部無效——提高檢索額度、工具回傳催促、
    系統指示加規則、開啟 think、`top_k` 提到 30、程式跑 6 輪查詢。
    最後一項最能說明問題：**第 1 輪就已經涵蓋 6/6，多跑 5 輪一點都沒改善**，
    證明瓶頸不在檢索而在生成——模型自己不會把讀到的東西全部列出來。

    因此改成不讓模型做彙整：
      * **清單由程式產生**（依 `section_of` 分組，確定性）
      * 模型只負責「描述單一章節的內容」——那是它做得好的事
      * 每次呼叫只餵一個章節，歸屬在結構上不可能出錯

    代價是 N 次 LLM 呼叫。但每次的上下文小很多，實際上不見得比
    「一次餵 30 段再生成長答案」慢。

    **這保證的是「檢索到的章節都會被列出」，不是「知識庫裡所有的都會被列出」**——
    後者取決於檢索，程式無法保證，所以結尾會誠實說明涵蓋範圍。
    """
    chunks, error = rag_service.retrieve(
        question, stage_code, top_k=ENUMERATION_TOP_K
    )
    if error:
        yield {"type": "error", "message": f"檢索失敗：{error}"}
        return
    if not chunks:
        yield {"type": "text", "piece": "知識庫中查無足夠資訊。"}
        return

    # 編號先配好，來源面板才對得上
    for chunk in chunks:
        citations.number(chunk.chunk_id)
    yield {"type": "search", "query": question, "stage": stage_code,
           "hits": len(chunks), "broad": True}

    # 依（文件, 章節）分組，保留最佳名次當排序依據
    groups: dict[tuple[str, str], list] = {}
    for chunk in chunks:
        key = (chunk.file_name, rag_service.section_of(chunk.locator))
        groups.setdefault(key, []).append(chunk)
    # 用重排序分數挑章節，不要用向量距離——距離的鑑別力太差，
    # 實測會把真正切題的章節排到第 12 名而選進不相關的。
    ordered = sorted(groups.items(), key=lambda kv: -rag_service.best_score(kv[1]))

    picked, dropped = ordered[:MAX_SECTIONS], ordered[MAX_SECTIONS:]

    header = f"檢索到 **{len(ordered)} 個相關章節**"
    if dropped:
        header += f"（以下展開最相關的 {len(picked)} 個）"
    yield {"type": "text", "piece": header + "：\n\n"}

    parts = [header + "：\n"]
    for (file_name, section) in [k for k, _ in picked]:
        members = groups[(file_name, section)]
        title = section or file_name
        block = "\n\n".join(
            f"[{citations.number(c.chunk_id)}] {c.locator}\n{c.content}" for c in members
        )
        prompt = (
            f"章節：{title}（出自 {file_name}）\n\n"
            f"內容：\n{block}\n\n---\n\n問題：{question}\n\n"
            f"請只根據上面這個章節的內容回答。"
        )
        # 400 不夠。這條路徑**專門處理列舉型問題**，而列舉正是最會撞上限的：
        # 實測一份分類表的章節輸出到第 29 項就被硬切在「ROCKC」中間——
        # 前面費了一番功夫才讓模型願意逐項列出來，最後卻被 token 上限砍掉。
        text, err = ollama_client.generate(prompt, system=SECTION_SYSTEM, num_predict=1200)
        body = rag_service.apply_blocklist(_strip_leak(text).strip()) or "（本章節未取得內容）"
        if err:
            body = f"（此章節處理失敗：{err[:80]}）"
        piece = f"### {title}\n\n{body}\n\n"
        parts.append(piece)
        yield {"type": "text", "piece": piece}

    if dropped:
        names = "、".join(s or f for (f, s) in [k for k, _ in dropped])
        tail = (
            f"\n> 另有 {len(dropped)} 個相關章節未展開：{names}。"
            f"若需要它們的細節，請針對該章節單獨提問。\n"
        )
        parts.append(tail)
        yield {"type": "text", "piece": tail}

    note = (
        "\n> 以上涵蓋的是**本次檢索到的章節**。知識庫中可能還有未被檢索到的相關內容，"
        "若懷疑有遺漏，可換關鍵詞再問一次。\n"
    )
    parts.append(note)
    yield {"type": "text", "piece": note}

    yield {
        "type": "done",
        "answer": "".join(parts),
        "chunk_ids": citations.chunk_ids,
        "searches": 1,
    }


def answer(question: str, history: list[dict] | None = None,
           stage_code: str | None = None, wide: bool = False,
           broad: bool = False):
    """執行一次問答。逐一 yield 事件字典：

        {"type": "search", "query": str, "stage": str|None, "hits": int, "broad": bool}
        {"type": "text", "piece": str}
        {"type": "error", "message": str}
        {"type": "done", "answer": str, "chunk_ids": [int], "searches": int}

    `stage_code` 是 UI 上選定的檢索範圍，會覆寫模型自己填的階段——
    使用者明確指定範圍時，那是指令而不是建議。
    """
    citations = _Citations()
    messages = (
        [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        + _history_messages(history)
        + [{"role": "user", "content": question}]
    )

    answer_text = ""
    searches = 0
    nudged = False

    # 列舉／彙整型問題改走「逐章節提問」——由程式保證清單完整，
    # 模型只描述單一章節。先探一次檢索結果，章節夠多才值得拆。
    if broad or _is_enumeration(question):
        probe, probe_error = rag_service.retrieve(
            question, stage_code, top_k=ENUMERATION_TOP_K
        )
        if not probe_error and probe:
            sections = {
                (c.file_name, rag_service.section_of(c.locator)) for c in probe
            }
            if len(sections) >= MIN_SECTIONS_TO_SPLIT:
                yield from _answer_by_section(question, stage_code, citations, wide)
                return

    # 列舉型問題撈更多段。**自動偵測 + 使用者可強制開啟**：
    # 自動偵測讓它開箱即用（否則問「有哪些」的人拿到 6 段，涵蓋率只有 2/6），
    # `broad` 則讓關鍵詞沒命中時使用者能自己補上，不必去猜要用哪個詞。
    enumeration = broad or _is_enumeration(question)
    max_searches = MAX_SEARCHES_ENUMERATION if enumeration else MAX_SEARCHES

    for round_no in range(max_searches + 2):
        # 額度用完的最後一輪不給工具，逼模型用手上的資料作答或誠實說沒有。
        tools = [SEARCH_TOOL] if round_no < max_searches else None

        # 一次都還沒查就吐出來的文字不能直接顯示給使用者。
        #
        # 實測過的失敗案例：問完「產出物有哪些」之後追問「還有嗎」，
        # 模型看到前一則回答裡已經有內容，就完全不檢索直接續寫，
        # 還編出 [8] 這種當輪根本不存在的來源編號。
        # 因此先緩衝，確認這一輪真的有檢索過（或已經催過一次）才敢即時串流。
        #
        # 正常路徑不會因此失去逐字輸出：第一輪模型是去呼叫工具的，本來就沒有文字，
        # 真正的答案出現在檢索之後的那一輪。
        #
        # 判斷條件只看「查過沒有」，不看「催過沒有」——催了不一定會照做，
        # 若以催過為準，模型第二次仍不查時，那段沒有依據的文字already 進了畫面。
        stream_live = searches > 0

        piece_buffer = ""
        emitted = 0  # 已經送出去幾個字元，用來銜接前置攔截與後續逐字輸出
        calls: list[dict] = []
        for event in ollama_client.chat_stream(messages, tools=tools):
            if event["type"] == "error":
                yield event
                return
            if event["type"] == "text":
                piece_buffer += event["piece"]
                if not stream_live or len(piece_buffer) < _LEAK_GUARD_CHARS:
                    continue
                clean = _strip_leak(piece_buffer)
                if len(clean) > emitted:
                    yield {
                        "type": "text",
                        "piece": rag_service.apply_blocklist(clean[emitted:]),
                    }
                    emitted = len(clean)
            elif event["type"] == "tool_calls":
                calls = event["calls"]

        # 內容短到還沒突破攔截門檻就結束了，補送剩下的。
        if stream_live:
            clean = _strip_leak(piece_buffer)
            if len(clean) > emitted:
                yield {"type": "text", "piece": rag_service.apply_blocklist(clean[emitted:])}
        piece_buffer = _strip_leak(piece_buffer)

        if not calls:
            # 完全沒查就想作答 —— 擋下來。
            #
            # 這在追問時特別容易發生：模型看到前一則回答裡已經有內容，
            # 就直接續寫，還會編出當輪根本不存在的來源編號。
            if searches == 0:
                if not nudged:
                    # 先好好講一次。多數情況這樣就會乖乖去查。
                    nudged = True
                    messages.append({"role": "assistant", "content": piece_buffer})
                    messages.append({"role": "user", "content": FORCE_SEARCH_NUDGE})
                    continue

                # 催過還是不查，就別再指望它了 —— 自己查，把結果塞回去。
                # 「答案必須有依據」是這個系統的存在前提，不能交給模型自律。
                #
                # **這裡刻意用原始問題，不做依前文的改寫。**
                # 曾經在這裡呼叫 condense_question()，結果是：使用者問「今天午餐吃什麼」，
                # 模型本來已經正確拒答，卻因為前文在談 PVT，
                # 改寫把問題變成「PVT 階段的產出物」，查回一堆 PVT 內容，
                # 模型就照著答了——一個合法的拒答被硬生生變成答非所問。
                #
                # 走到這一步代表模型已經不配合，此時再加一層猜測只會放大錯誤。
                # 用原始問題最壞的情況是查不到而回「查無資訊」，方向是安全的。
                content, hits = _run_search(question, stage_code, citations, wide,
                                            enumeration, max_searches - searches - 1)
                query = question
                searches += 1
                yield {"type": "search", "query": query, "stage": stage_code, "hits": hits,
                       "broad": enumeration}
                # 檢索結果放在 user 訊息裡，不用 tool 訊息。
                #
                # `tool` 角色的訊息在對話模板中必須跟在帶 tool_calls 的 assistant
                # 訊息後面才會被算進去；這裡是我們自己補查的，沒有對應的 tool_call，
                # 模型會完全看不到內容，然後回你「您尚未提供檢索結果」。
                messages.append({"role": "assistant", "content": piece_buffer})
                messages.append({
                    "role": "user",
                    "content": (
                        f"以下是知識庫的檢索結果：\n\n{content}\n\n"
                        f"請依據以上內容回答我的問題「{question}」並標註來源編號；"
                        f"若內容確實無法回答，就說知識庫中查無足夠資訊。"
                    ),
                })
                continue

            answer_text = rag_service.apply_blocklist(piece_buffer)

            # 模型有可能什麼都不回就結束（實測 gemma4:e2b 在追問時常這樣），
            # 直接存下去會在對話紀錄裡留一則空白訊息，使用者也不知道發生什麼事。
            if not answer_text.strip():
                answer_text = (
                    "（模型未產生回覆）已完成檢索但沒有得到答案，請換個說法再問一次。"
                    "若持續發生，請管理員到「模型設定」確認目前的生成模型是否適用。"
                )
                yield {"type": "text", "piece": answer_text}
            elif not stream_live:  # 緩衝下來的內容還沒送出去，補上
                yield {"type": "text", "piece": answer_text}
            break

        # 模型偶爾會在呼叫工具前先講一句話，那不是最終答案，不保留。
        messages.append({"role": "assistant", "content": piece_buffer, "tool_calls": calls})

        for call in calls:
            function = call.get("function", {})
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):  # 有些版本會回字串
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            query = (arguments.get("query") or question).strip()
            stage = stage_code or (arguments.get("stage_code") or None)
            content, hits = _run_search(query, stage, citations, wide,
                                        enumeration, max_searches - searches - 1)
            searches += 1
            yield {"type": "search", "query": query, "stage": stage, "hits": hits,
                   "broad": enumeration}
            messages.append({"role": "tool", "content": content})

    else:
        # 迴圈跑完仍沒有定案（模型一直在呼叫工具）。用最後一輪的文字，
        # 沒有就給一句明確的說明，不要回空白。
        answer_text = rag_service.apply_blocklist(piece_buffer) or (
            "知識庫中查無足夠資訊。已嘗試多種關鍵詞檢索，仍找不到能回答此問題的內容。"
        )
        yield {"type": "text", "piece": answer_text}

    yield {
        "type": "done",
        "answer": answer_text,
        "chunk_ids": citations.chunk_ids,
        "searches": searches,
    }
