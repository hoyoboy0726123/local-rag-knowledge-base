"""知識庫文件查詢、切片維護、文件閱讀與下載。"""

from __future__ import annotations

import mimetypes
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from database import get_session, get_setting
from models import DOC_INDEXED, Document
from services import ingest_service


def get_kb_documents(kb: str | None = None) -> pd.DataFrame:
    """某個知識庫的已索引文件。

    `kb` 是 None → 全部；`""`（空字串）→ 通用（根目錄檔案，kb 為 NULL）；
    其他 → 該名稱的知識庫。空字串當「通用」的鍵是刻意的：它不是資料夾，
    沒有名稱可用，而 None 已經被「全部」佔走。
    """
    with get_session() as session:
        query = session.query(Document).filter(Document.status == DOC_INDEXED)
        if kb == "":
            query = query.filter(Document.kb.is_(None))
        elif kb:
            query = query.filter(Document.kb == kb)
        rows = query.order_by(Document.file_name).all()

    return pd.DataFrame(
        [
            {
                "文件名稱": d.file_name,
                "類型": d.file_type.upper(),
                "知識庫": d.kb or "通用",
                "切片數": d.chunk_count,
                "VLM 解析": "✅" if d.used_vlm else "",
                "索引時間": d.indexed_at.strftime("%Y-%m-%d %H:%M") if d.indexed_at else "-",
                "路徑": d.file_path,
            }
            for d in rows
        ]
    )


def kb_doc_counts() -> dict[str | None, int]:
    """各知識庫的已索引文件數。鍵 None 代表通用。"""
    from sqlalchemy import func

    with get_session() as session:
        rows = (
            session.query(Document.kb, func.count(Document.id))
            .filter(Document.status == DOC_INDEXED)
            .group_by(Document.kb)
            .all()
        )
    return {kb: count for kb, count in rows}


def list_chunks(
    doc_id: int | None = None, keyword: str = "", limit: int = 200
) -> pd.DataFrame:
    """列出已索引的切片內容，供管理員確認索引結果是否正確。

    這是最直接的除錯工具——AI 答不出來時，先看它到底存進去了什麼。
    """
    from models import Chunk

    with get_session() as session:
        query = (
            session.query(Chunk, Document)
            .join(Document, Chunk.doc_id == Document.id)
        )
        if doc_id:
            query = query.filter(Chunk.doc_id == doc_id)
        if keyword.strip():
            query = query.filter(Chunk.content.like(f"%{keyword.strip()}%"))
        rows = query.order_by(Chunk.doc_id, Chunk.seq).limit(limit).all()

    return pd.DataFrame(
        [
            {
                "切片 ID": chunk.id,
                # 前端要靠這個把切片依文件分組。用檔名分組會在同名不同知識庫時併錯組。
                "文件 ID": chunk.doc_id,
                "文件": doc.file_name,
                "知識庫": doc.kb or "通用",
                "位置": chunk.locator,
                "字數": chunk.char_count,
                "內容": chunk.content,
            }
            for chunk, doc in rows
        ]
    )


def _revector(chunk_id: int, content: str, keywords: str) -> tuple[bool, str]:
    """用給定的內容重算單一切片的向量。

    向量是索引當下算出來的，**只改資料庫欄位不會讓檢索有任何變化**。
    少了這一步，管理員會以為調整生效了，實際上什麼都沒動到。
    """
    from database import raw_connection, serialize
    from services import ingest_service, ollama_client

    vectors, error = ollama_client.embed([ingest_service.embed_text(content, keywords)])
    if error or not vectors:
        return False, f"重算向量失敗：{error or '回傳為空'}"

    conn = raw_connection()
    try:
        # sqlite-vec 虛擬表不支援 INSERT OR REPLACE，必須先刪再插
        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,))
        conn.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, serialize(vectors[0])),
        )
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        return False, f"寫入向量失敗：{type(exc).__name__}: {str(exc)[:120]}"
    finally:
        conn.close()
    return True, ""


def set_chunk_content(chunk_id: int, content: str) -> tuple[bool, str]:
    """編輯切片內容，並立刻重算向量。

    用途是清掉解析出來的噪音——郵件的收件人清單、簽名檔、免責聲明、
    頁首頁尾重複的表頭。這些文字會佔用檢索名額卻答不出任何東西。

    **為什麼原本不開放、現在可以開放：**

    當初反對有兩個理由，其中一個是真的、一個是可以解掉的：

    1. 「向量不會跟著變」—— 這個**可以解**，就是下面 `_revector()` 做的事，
       跟關鍵字編輯用的是同一套機制。
    2. 「破壞可追溯性」—— 這才是真問題：`content` 同時是使用者看到的原文
       與 LLM 作答的依據，改了它，來源卡片就會顯示原始文件裡沒有的文字。
       解法是**保留 `original_content`**：第一次編輯時把解析結果原樣存下來，
       之後隨時能比對與還原，介面上也標示「已編輯」。
       traceability 從「內容必定等於原文」變成「內容的來歷必定查得到」——
       後者才是真正需要的保證。

    還有第三件當初沒講、但更容易讓人踩到的事：切片是**衍生資料**，
    重新索引會整批重建。所以編輯必須跟關鍵字一樣另存一份
    （`chunk_keywords.edited_content`），否則下次全量重建就無聲消失。
    """
    from models import Chunk
    from services import ingest_service

    content = (content or "").strip()
    if not content:
        return False, "內容不能是空的（要整段拿掉請用刪除切片）"

    with get_session() as session:
        result = (
            session.query(Chunk, Document)
            .join(Document, Chunk.doc_id == Document.id)
            .filter(Chunk.id == chunk_id)
            .first()
        )
        if not result:
            return False, "找不到該切片"
        chunk, doc = result
        if chunk.content == content:
            return True, "內容沒有變動"
        # 只在第一次編輯時保存原文，之後不再覆蓋——
        # 否則編輯兩次就再也回不到檔案真正解析出來的樣子。
        original = chunk.original_content or chunk.content
        keywords, seq, file_path = chunk.keywords, chunk.seq, doc.file_path

    ok, error = _revector(chunk_id, content, keywords)
    if not ok:
        return False, error

    with get_session() as session:
        chunk = session.get(Chunk, chunk_id)
        chunk.original_content = original
        chunk.content = content
        chunk.char_count = len(content)
        session.commit()

    ingest_service.save_chunk_override(file_path, seq, original, keywords, content)
    saved = len(original) - len(content)
    return True, f"已更新並重算向量（少了 {saved} 字）" if saved > 0 else "已更新並重算向量"


def revert_chunk_content(chunk_id: int) -> tuple[bool, str]:
    """把切片還原成檔案原本解析出來的內容。"""
    from models import Chunk
    from services import ingest_service

    with get_session() as session:
        result = (
            session.query(Chunk, Document)
            .join(Document, Chunk.doc_id == Document.id)
            .filter(Chunk.id == chunk_id)
            .first()
        )
        if not result:
            return False, "找不到該切片"
        chunk, doc = result
        if not chunk.original_content:
            return False, "這個切片沒有被編輯過"
        original, keywords = chunk.original_content, chunk.keywords
        seq, file_path = chunk.seq, doc.file_path

    ok, error = _revector(chunk_id, original, keywords)
    if not ok:
        return False, error

    with get_session() as session:
        chunk = session.get(Chunk, chunk_id)
        chunk.content = original
        chunk.char_count = len(original)
        chunk.original_content = ""
        session.commit()

    ingest_service.save_chunk_override(file_path, seq, original, keywords, "")
    return True, "已還原為原始解析內容"


def delete_chunk(chunk_id: int) -> tuple[bool, str]:
    """刪除單一切片與它的向量。

    用途是把噪音切片踢出檢索範圍——例如 `.msg` 郵件裡的收件人清單、
    簽名檔、免責聲明，這些內容會佔用檢索名額卻答不出任何東西。

    **這個刪除不是永久的，要講清楚：** 切片是從檔案解析出來的衍生資料。
    「增量更新」會比對 sha256 跳過未變更的檔案，所以刪掉的切片不會回來；
    但「全量重建」會重新解析所有檔案，被刪的切片會**原樣長回來**。
    要永久排除，只能改動或刪除來源檔案本身。
    """
    from models import Chunk

    with get_session() as session:
        chunk = session.get(Chunk, chunk_id)
        if not chunk:
            return False, "切片不存在"
        doc_id = chunk.doc_id
        session.delete(chunk)
        # chunk_count 是給管理端顯示用的統計，不同步會讓文件清單的數字對不上
        doc = session.get(Document, doc_id)
        if doc and doc.chunk_count > 0:
            doc.chunk_count -= 1
        session.commit()

    # sqlite-vec 的虛擬表不會因為 chunks 被刪就跟著消失，必須自己清
    from database import raw_connection

    conn = raw_connection()
    try:
        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,))
        conn.commit()
    finally:
        conn.close()

    return True, f"已刪除切片 #{chunk_id}"


def get_chunk(chunk_id: int) -> dict | None:
    from models import Chunk

    with get_session() as session:
        result = (
            session.query(Chunk, Document)
            .join(Document, Chunk.doc_id == Document.id)
            .filter(Chunk.id == chunk_id)
            .first()
        )
        if not result:
            return None
        chunk, doc = result
        return {
            "id": chunk.id, "file_name": doc.file_name,
            "kb": doc.kb, "locator": chunk.locator,
            "content": chunk.content, "char_count": chunk.char_count,
            "keywords": chunk.keywords or "",
            # 有原文備份就代表被編輯過。前端要標示出來，
            # 否則管理員看到的內容跟檔案不一樣卻無從得知。
            "edited": bool(chunk.original_content),
            "original_content": chunk.original_content or "",
        }


@lru_cache(maxsize=32)
def _parse_cached(file_path: str, mtime: float, enable_vlm: bool, force_vlm: bool = False) -> tuple[str, str]:
    """解析結果快取。以 mtime 當快取鍵，檔案更新後自動失效。

    `force_vlm` 也必須是快取鍵的一部分：同一個檔案在開／關強制視覺解析下
    會解析出完全不同的內容，少了它會回傳上一次設定的結果。
    """
    content, _used_vlm, error = ingest_service.extract_markdown(
        Path(file_path), enable_vlm, force_vlm
    )
    return content, error


def is_readable_document(file_path: str) -> bool:
    """這個路徑是否為知識庫裡真正被索引過的文件。

    **這是一道必要的安全檢查，不是防呆。** 這些讀檔函式的 `file_path`
    全部來自前端查詢參數，完全由使用者控制。少了這道檢查，任何登入者
    只要把 path 換成 `C:\\...\\.env` 或系統任意檔案就能整份讀走。

    用資料庫白名單而不是「檢查是否在知識庫資料夾底下」：
    後者得處理 `..`、符號連結、大小寫、磁碟機代號等一堆邊界情況，
    而且知識庫資料夾裡本來就可能有沒被索引的雜項檔案。
    白名單直接對到 `documents` 表，能讀的就只有真的進過索引的那些。

    路徑比對前先 `abspath` + `normcase`：資料庫存的是絕對路徑，
    但前端可能傳來大小寫不同或含 `/` 的等價寫法。
    """
    if not file_path:
        return False
    target = os.path.normcase(os.path.abspath(file_path))
    with get_session() as session:
        rows = session.query(Document.file_path).all()
    return any(os.path.normcase(os.path.abspath(r[0])) == target for r in rows if r[0])


def read_document(file_path: str) -> tuple[str, str]:
    """讀取單一文件並回傳可直接渲染的 Markdown。回傳 (內容, 錯誤)。

    **優先用索引時存下來的內容**（`documents.content`），而不是當場重新解析。

    重新解析看起來比較「新鮮」，實際上是錯的：解析結果取決於解析當下的
    VLM 設定——全域開關，以及該檔案有沒有被勾「強制視覺解析」。閱讀時用
    另一組設定重跑，拿到的就不是 AI 讀到的那份東西。實測踩過：某份 PDF
    以強制視覺解析建索引，切片裡有 VLM 產生的圖片描述，閱讀頁卻只剩文字層，
    兩邊對不起來。

    而且重跑很貴——開了視覺解析的檔案每頁 3–11 秒，看一份文件不該等好幾分鐘。

    只有舊資料庫（欄位剛加、尚未重新索引）才回退到重新解析，
    這時**必須帶上該檔案的強制 VLM 設定**，否則會重現上面那個不一致。

    **不把切片拼回去**：切片之間有 80 字重疊，拼接會產生重複段落；
    而且切片是為了檢索而切的，不是為了閱讀。
    """
    if not is_readable_document(file_path):
        return "", "找不到這份文件，或它不在知識庫索引中。"

    target = os.path.normcase(os.path.abspath(file_path))
    with get_session() as session:
        for doc in session.query(Document).all():
            if doc.file_path and os.path.normcase(os.path.abspath(doc.file_path)) == target:
                if doc.content:
                    return doc.content, ""
                break

    path = Path(file_path)
    if not path.exists():
        return "", "檔案已不存在，請聯繫管理員重新建立索引。"
    try:
        mtime = path.stat().st_mtime
    except OSError as exc:
        return "", f"無法讀取檔案：{exc}"

    enable_vlm = get_setting("enable_vlm", "1") == "1"
    force_vlm = ingest_service.get_doc_option(str(path))
    try:
        return _parse_cached(str(path), mtime, enable_vlm, force_vlm)
    except Exception as exc:  # noqa: BLE001
        return "", f"解析失敗：{type(exc).__name__}: {str(exc)[:150]}"


def read_document_bytes(file_path: str) -> bytes | None:
    """讀出原始檔位元組，供下載按鈕使用（遠端使用者才拿得到檔案）。"""
    if not is_readable_document(file_path):
        return None
    path = Path(file_path)
    if not path.exists():
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


# 允許以 inline 送出的格式。這是白名單，不是黑名單，**兩個理由都是必要的**：
#
#   1. **安全**：前端用 blob URL 開新分頁，而 blob URL 會繼承建立它的頁面來源。
#      因此 `.svg` 或 `.html` 若以真實 MIME 送出，其中的 <script> 會在本應用的
#      origin 下執行，讀得到 sessionStorage 裡的 JWT ——上傳一個檔案就能盜用
#      其他人的登入狀態。這兩種格式一律不給 inline。
#   2. **編碼**：純文字類（.md/.txt）在 blob URL 下瀏覽器常忽略 charset，
#      改用系統預設編碼解碼，中文會整片變成亂碼。這類檔案本來就該走
#      「開啟完整文件」——那條路徑有正確渲染，效果比原始碼好得多。
#
# 剩下的都是二進位且不可執行的格式，交給瀏覽器內建檢視器最安全。
INLINE_VIEWABLE = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def can_inline(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in INLINE_VIEWABLE


def media_type_of(file_path: str) -> str:
    """猜出 MIME type，供「在新分頁開啟」使用。

    下載走 `application/octet-stream` 是對的（強制存檔），但要讓瀏覽器
    **顯示** PDF 就必須給 `application/pdf`，否則它一樣會當成檔案下載。
    """
    guessed, _ = mimetypes.guess_type(file_path)
    return guessed or "application/octet-stream"


def open_with_local_app(file_path: str) -> tuple[bool, str]:
    """用伺服器本機的預設程式開啟檔案。

    **呼叫端必須先確認請求來自 localhost。** 這個函式會在「執行後端的那台機器」
    上開檔；區網部署時等於在伺服器上開，遠端使用者什麼也看不到。
    這裡不做來源判斷是因為服務層拿不到請求資訊，由 API 層把關。
    """
    if not is_readable_document(file_path):
        return False, "找不到這份文件，或它不在知識庫索引中。"
    path = Path(file_path)
    if not path.exists():
        return False, "檔案已不存在。"
    opener = getattr(os, "startfile", None)
    if opener is None:
        return False, "此作業系統不支援以本機程式開啟。"
    try:
        opener(str(path))
        return True, ""
    except OSError as exc:
        return False, f"無法開啟：{exc}"


def set_chunk_keywords(chunk_id: int, keywords: str) -> tuple[bool, str]:
    """更新切片的檢索關鍵字，並**立刻重算該切片的向量**。

    重算是關鍵：向量是索引當下算出來的，只改資料庫欄位不會讓檢索有任何變化。
    若省略這一步，管理員會以為調校生效了，實際上完全沒動到。

    同時把關鍵字另存一份（`chunk_keywords` 表），
    讓它在下一次重新索引後能被套回去，不會無聲消失。
    """
    from database import raw_connection, serialize
    from models import Chunk
    from services import ingest_service, ollama_client

    keywords = (keywords or "").strip()

    with get_session() as session:
        result = (
            session.query(Chunk, Document)
            .join(Document, Chunk.doc_id == Document.id)
            .filter(Chunk.id == chunk_id)
            .first()
        )
        if not result:
            return False, "找不到該切片"
        chunk, doc = result
        content, seq, file_path = chunk.content, chunk.seq, doc.file_path

    vectors, error = ollama_client.embed([ingest_service.embed_text(content, keywords)])
    if error or not vectors:
        return False, f"重算向量失敗：{error or '回傳為空'}"

    conn = raw_connection()
    try:
        # sqlite-vec 的虛擬表**不支援 `INSERT OR REPLACE`**——
        # 對已存在的 chunk_id 會噴 `UNIQUE constraint failed on primary key`。
        # 必須先刪再插。（索引流程的 INSERT OR REPLACE 之所以沒事，
        # 是因為那裡的切片剛建立、鍵一定不存在。）
        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,))
        conn.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, serialize(vectors[0])),
        )
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        return False, f"寫入向量失敗：{type(exc).__name__}: {str(exc)[:120]}"
    finally:
        conn.close()

    with get_session() as session:
        chunk_now = session.get(Chunk, chunk_id)
        chunk_now.keywords = keywords
        # 這一段若曾被編輯過，備份的指紋要用原文，備份的內容要保住編輯結果——
        # 否則改個關鍵字就會把編輯過的內容洗掉。
        original = chunk_now.original_content or content
        edited = content if chunk_now.original_content else ""
        session.commit()

    ingest_service.save_chunk_override(file_path, seq, original, keywords, edited)
    return True, "已更新並重算向量"


def document_options() -> list[tuple[int, str]]:
    with get_session() as session:
        rows = (
            session.query(Document)
            .filter(Document.status == DOC_INDEXED)
            .order_by(Document.file_name)
            .all()
        )
        return [(d.id, f"{d.file_name}（{d.chunk_count} 切片）") for d in rows]


def library_stats() -> dict:
    from sqlalchemy import func

    from models import Chunk, IngestError

    with get_session() as session:
        docs = session.query(func.count(Document.id)).filter(
            Document.status == DOC_INDEXED
        ).scalar()
        chunks = session.query(func.count(Chunk.id)).scalar()
        errors = session.query(func.count(IngestError.id)).scalar()
        last = session.query(func.max(Document.indexed_at)).scalar()

    return {
        "documents": docs or 0,
        "chunks": chunks or 0,
        "errors": errors or 0,
        "last_indexed": last.strftime("%Y-%m-%d %H:%M") if last else "尚未建立索引",
    }
