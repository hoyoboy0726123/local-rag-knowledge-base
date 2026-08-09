"""管理端：索引、切片、關鍵字、模型、帳號。全部僅限管理員。"""

from __future__ import annotations

import json
import queue
import threading

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.deps import require_admin
from database import get_session, get_setting, set_setting
from models import IngestError
from services import (auth_service, ingest_service, ollama_client, rag_service,
                      reranker, stage_service)

router = APIRouter(prefix="/api/admin", tags=["admin"])


class KeywordsBody(BaseModel):
    keywords: str


class ContentBody(BaseModel):
    content: str


class SearchTestBody(BaseModel):
    query: str
    stage_code: str | None = None


class ModelsBody(BaseModel):
    embed_model: str | None = None
    llm_model: str | None = None
    vlm_model: str | None = None


# ------------------------------------------------------------------ 切片
@router.get("/chunks")
def chunks(doc_id: int | None = None, keyword: str = "", limit: int = 3000,
           _: dict = Depends(require_admin)) -> dict:
    """切片清單。

    上限拉到 3000：一封 `.msg` 郵件就可能切出上百段，
    原本的 200 會在管理員完全不知情的狀況下把後面的切片藏起來——
    而這一頁的用途正是「確認到底存進去了什麼」，靜默截斷會直接誤導判斷。
    """
    df = stage_service.list_chunks(doc_id, keyword, limit)
    rows = [] if df.empty else df.to_dict("records")
    return {"chunks": rows, "truncated": len(rows) >= limit, "limit": limit}


@router.put("/chunks/{chunk_id}/content")
def set_content(chunk_id: int, body: ContentBody,
                _: dict = Depends(require_admin)) -> dict:
    """編輯切片內容並立刻重算向量。

    可追溯性靠 `original_content` 保住：第一次編輯時把解析結果原樣存下來，
    介面標示「已編輯」、隨時可比對與還原。
    編輯結果會另存進 `chunk_keywords.edited_content`，重新索引後自動套回。
    """
    ok, message = stage_service.set_chunk_content(chunk_id, body.content)
    if not ok:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, message)
    return {"ok": True, "message": message}


@router.post("/chunks/{chunk_id}/revert")
def revert_content(chunk_id: int, _: dict = Depends(require_admin)) -> dict:
    """還原成檔案原本解析出來的內容。"""
    ok, message = stage_service.revert_chunk_content(chunk_id)
    if not ok:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, message)
    return {"ok": True, "message": message}


@router.delete("/chunks/{chunk_id}")
def delete_chunk(chunk_id: int, _: dict = Depends(require_admin)) -> dict:
    """刪除單一切片與其向量。

    **不是永久的**：切片是從檔案解析出來的衍生資料，
    「全量重建」會讓它原樣長回來（增量更新則因為 sha256 未變而跳過該檔）。
    前端必須把這件事講清楚，否則管理員會以為噪音已經永久排除。
    """
    ok, message = stage_service.delete_chunk(chunk_id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, message)
    return {"deleted": True, "message": message}


@router.get("/chunks/{chunk_id}")
def chunk_detail(chunk_id: int, _: dict = Depends(require_admin)) -> dict:
    detail = stage_service.get_chunk(chunk_id)
    if not detail:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "找不到該切片")
    return detail


@router.put("/chunks/{chunk_id}/keywords")
def set_keywords(chunk_id: int, body: KeywordsBody,
                 _: dict = Depends(require_admin)) -> dict:
    """更新檢索關鍵字並**立即重算該切片的向量**。

    少了重算這一步，整個功能就只是個沒作用的輸入框——
    向量是索引當下算的，只改資料庫欄位檢索結果完全不變。
    """
    ok, message = stage_service.set_chunk_keywords(chunk_id, body.keywords)
    if not ok:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, message)
    return {"message": message}


# ------------------------------------------------------------------ 檢索測試
@router.post("/search-test")
def search_test(body: SearchTestBody, _: dict = Depends(require_admin)) -> dict:
    hits, error = rag_service.retrieve(body.query, body.stage_code)
    if error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, error)
    return {
        "hits": [
            {
                "chunk_id": h.chunk_id, "file_name": h.file_name,
                "locator": h.locator, "stage_code": h.stage_code,
                "distance": round(h.distance, 4),
                "content": h.content[:300],
            }
            for h in hits
        ],
        "threshold": rag_service.DISTANCE_THRESHOLD,
    }


# ------------------------------------------------------------------ 索引
@router.get("/index-status")
def index_status(_: dict = Depends(require_admin)) -> dict:
    root = get_setting("knowledge_root", "")
    ready, reason = ingest_service.vlm_ready()
    from pathlib import Path
    images = ingest_service.count_image_type(Path(root)) if root else []
    return {
        "root": root,
        "stats": stage_service.library_stats(),
        "vlm_ready": ready,
        "vlm_reason": reason,
        "image_files": [p.name for p in images],
        "recommended_vlm": ingest_service.RECOMMENDED_VLM,
        # 支援格式從 service 的常數直接送出，不在前端寫死。
        # 寫死的清單改了副檔名不會同步，使用者會照著上傳一個其實不支援的檔案。
        "supported": sorted(ingest_service.SUPPORTED),
        "image_types": sorted(ingest_service.IMAGE_TYPES),
    }


@router.post("/index")
def run_index(full: bool = False, _: dict = Depends(require_admin)) -> StreamingResponse:
    """執行索引，以 SSE 逐則回報進度。"""
    root = get_setting("knowledge_root", "")

    def stream():
        """**索引必須跑在另一條執行緒，訊息透過 queue 即時送出。**

        原本的寫法是把訊息累積進 list、等 `ingest()` 整個跑完才一次 yield ——
        那樣 SSE 形同虛設，畫面會一直停在「索引中…」直到全部結束。
        開了強制視覺解析之後這個問題更嚴重：一份簡報可能跑好幾分鐘，
        使用者會以為系統當掉。

        SQLite 的 engine 已設 `check_same_thread: False`，可以跨執行緒使用。
        """
        messages: queue.Queue = queue.Queue()
        box: dict = {}

        def progress(message: str) -> None:
            messages.put(message)

        def work() -> None:
            try:
                box["result"] = ingest_service.ingest(root, full_rebuild=full,
                                                      progress=progress)
            except Exception as exc:  # noqa: BLE001
                box["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                messages.put(None)  # 結束訊號

        worker = threading.Thread(target=work, daemon=True)
        worker.start()

        while True:
            try:
                line = messages.get(timeout=1)
            except queue.Empty:
                # 心跳：長時間沒有訊息時（例如單張圖跑很久）送一則註解，
                # 讓連線不被中間層判定為閒置而切斷。
                yield ": keepalive\n\n"
                continue
            if line is None:
                break
            yield f"event: log\ndata: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"

        worker.join(timeout=5)
        if "error" in box:
            yield (f"event: log\ndata: "
                   f"{json.dumps({'line': '索引失敗：' + box['error']}, ensure_ascii=False)}\n\n")
        result = box.get("result")
        payload = {
            "new": result.new if result else 0,
            "updated": result.updated if result else 0,
            "skipped": result.skipped if result else 0,
            "failed": result.failed if result else 0,
            "chunks": result.chunks if result else 0,
            "needs_vlm": result.needs_vlm if result else [],
        }
        yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no"})


@router.post("/upload")
async def upload(stage_code: str | None = None, force_vlm: bool = False,
                 files: list[UploadFile] = File(...),
                 _: dict = Depends(require_admin)) -> dict:
    # 還沒設定知識庫資料夾時，**自動建立預設資料夾並記住它**。
    #
    # 「上傳一份文件」是新使用者最自然的第一個動作，不該因為少設一個路徑就失敗。
    # 先前會回「知識庫資料夾不存在：（未設定）」，使用者看不出要去哪裡設定，
    # 接著按下「全量重建」——而空路徑在 Windows 等同當前目錄，
    # 於是整個專案（含 venv）被當成知識庫掃進索引。
    root = get_setting("knowledge_root", "")
    if not root.strip():
        root = str(ingest_service.DEFAULT_KB_DIR)
        ingest_service.DEFAULT_KB_DIR.mkdir(parents=True, exist_ok=True)
        set_setting("knowledge_root", root)

    saved, errors = ingest_service.save_uploads(root, files, stage_code)
    # 「要不要用視覺模型解析」在上傳當下就該決定，所以選項在這裡一併寫入。
    # 存的是絕對路徑，跟索引流程查的鍵一致。
    if force_vlm:
        from pathlib import Path as _P
        for rel in saved:
            ingest_service.set_doc_option(str(_P(root) / rel), True)
    return {"saved": saved, "errors": errors}


@router.get("/documents")
def documents(_: dict = Depends(require_admin)) -> dict:
    """知識庫資料夾裡的所有檔案（含尚未索引的）。"""
    root = get_setting("knowledge_root", "")
    return {"root": root, "documents": ingest_service.list_library_files(root)}


@router.put("/documents/options")
def set_doc_options(path: str, force_vlm: bool,
                    _: dict = Depends(require_admin)) -> dict:
    """切換單一檔案的「強制視覺解析」。

    改完要重新索引才會生效——索引流程的跳過條件已把這個旗標算進去，
    所以按一次「增量更新」就會重解析這一份（不必全量重建）。
    """
    from pathlib import Path as _P

    root = get_setting("knowledge_root", "")
    try:
        target = (_P(root) / path).resolve()
        target.relative_to(_P(root).resolve())
    except (ValueError, OSError):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "路徑不合法")
    if not target.is_file():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "檔案不存在")

    ingest_service.set_doc_option(str(target), force_vlm)
    return {"ok": True, "force_vlm": force_vlm,
            "message": ("已設為強制視覺解析，執行「增量更新」後生效"
                        if force_vlm else "已取消強制視覺解析")}


@router.delete("/documents")
def delete_document(path: str, _: dict = Depends(require_admin)) -> dict:
    """刪除檔案本身與它在資料庫裡的所有痕跡（含向量）。

    走 query string 的 `path` 而不是 `doc_id`：尚未索引的檔案沒有 doc_id，
    用 id 當鍵會有一半的檔案刪不掉。路徑遍歷的防護在 service 層。
    """
    root = get_setting("knowledge_root", "")
    ok, message = ingest_service.delete_library_file(root, path)
    if not ok:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, message)
    return {"deleted": True, "message": message}


@router.get("/errors")
def errors(_: dict = Depends(require_admin)) -> dict:
    """**分三類回傳。**

    「缺少視覺模型」「部分內容未解析」「檔案解析失敗」對管理員意味著
    完全不同的動作，混在一起會讓人做錯判斷：
      * needs_vlm —— 檔案沒問題，裝個模型重跑就好
      * partial  —— **文件已經在索引裡**，只是有幾張圖沒讀到，可查但可能不全
      * failures —— 真的整份沒進去

    partial 特別不能混進 failures：使用者看到「失敗」會以為要重傳，
    實際上該做的是判斷缺的那部分重不重要。
    """
    with get_session() as session:
        rows = session.query(IngestError).order_by(IngestError.occurred_at.desc()).all()
        items = [
            {
                "file_name": e.file_name, "error_type": e.error_type,
                "message": e.message,
                "occurred_at": e.occurred_at.strftime("%Y-%m-%d %H:%M"),
            }
            for e in rows
        ]
    return {
        "needs_vlm": [i for i in items if i["error_type"] == "needs_vlm"],
        "partial": [i for i in items if i["error_type"] == "partial"],
        "failures": [i for i in items
                     if i["error_type"] not in ("needs_vlm", "partial")],
    }


# ------------------------------------------------------------------ 模型
@router.get("/models")
def models(_: dict = Depends(require_admin)) -> dict:
    status_ = ollama_client.check_status()
    return {
        "alive": status_.alive,
        "available": status_.models,
        "current": {
            "embed_model": get_setting("embed_model"),
            "llm_model": get_setting("llm_model"),
            "vlm_model": get_setting("vlm_model"),
        },
        "supports_tools": ollama_client.supports_tools() if status_.alive else False,
    }


@router.put("/models")
def update_models(body: ModelsBody, _: dict = Depends(require_admin)) -> dict:
    """只寫入前端明確送來的欄位。

    **不可以因為「清單裡沒有目前的值」就自動改成第一個**——
    V1 就是這樣把 embed_model 靜默換成生成模型，導致每次檢索都 501。
    """
    changed = []
    for key in ("embed_model", "llm_model", "vlm_model"):
        value = getattr(body, key)
        if value and value != get_setting(key):
            set_setting(key, value)
            changed.append(key)
    return {"changed": changed}


def _log(line: str) -> str:
    payload = json.dumps({"line": line}, ensure_ascii=False)
    return f"event: log\ndata: {payload}\n\n"


# ------------------------------------------------------------------ 重排序
@router.get("/reranker")
def reranker_status(_: dict = Depends(require_admin)) -> dict:
    return {**reranker.status(), "enabled": get_setting("enable_rerank", "1") == "1"}


class RerankBody(BaseModel):
    enabled: bool


@router.put("/reranker")
def reranker_toggle(body: RerankBody, _: dict = Depends(require_admin)) -> dict:
    set_setting("enable_rerank", "1" if body.enabled else "0")
    return {"enabled": body.enabled}


@router.post("/reranker/download")
def reranker_download(_: dict = Depends(require_admin)) -> StreamingResponse:
    """下載重排序模型（571 MB），以 SSE 回報進度。

    下載完**不需要重啟**：`reranker._load()` 是延遲載入，
    下一次檢索呼叫時才會去讀模型檔，屆時檔案已經在了。
    """
    def stream():
        messages: queue.Queue = queue.Queue()
        box: dict = {}

        def work() -> None:
            try:
                box["result"] = reranker.download(progress=messages.put)
            except Exception as exc:  # noqa: BLE001
                box["result"] = (False, f"{type(exc).__name__}: {exc}")
            finally:
                messages.put(None)

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        yield _log("開始下載（約 571 MB，視網路可能要數分鐘）…")
        while True:
            try:
                line = messages.get(timeout=1)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if line is None:
                break
            yield _log(line)
        ok, message = box.get("result", (False, "未取得結果"))
        payload = {"ok": ok, "message": message, **reranker.status()}
        body = json.dumps(payload, ensure_ascii=False)
        yield f"event: done\ndata: {body}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# ------------------------------------------------------------------ 帳號
@router.get("/users")
def users(_: dict = Depends(require_admin)) -> dict:
    df = auth_service.list_users()
    return {"users": [] if df.empty else df.to_dict("records")}
