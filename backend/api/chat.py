"""問答：SSE 串流 + 對話紀錄。

這是 V2 唯一需要新寫的核心邏輯，而且只是把 `agent_service.answer()`
既有的 generator 轉成 SSE——**Agent 的三層強制檢索防護與緩衝規則完全不動**。
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.deps import current_user
from services import agent_service, chat_service, ollama_client, rag_service

router = APIRouter(prefix="/api/chat", tags=["chat"])


class AskBody(BaseModel):
    session_id: int
    question: str
    stage_code: str | None = None
    # 使用者對答案不滿意時可勾選，改用整個結構單元當脈絡重問一次
    wide: bool = False
    # 強制撈更多來源（列舉型問題本來就會自動開啟，這是手動補開的開關）
    broad: bool = False


@router.get("/sessions")
def sessions(user: dict = Depends(current_user)) -> dict:
    df = chat_service.list_sessions(user["id"])
    return {"sessions": [] if df.empty else df.to_dict("records")}


@router.post("/sessions")
def create_session(user: dict = Depends(current_user)) -> dict:
    return {"session_id": chat_service.create_session(user["id"])}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: int, user: dict = Depends(current_user)) -> dict:
    if not chat_service.delete_session(session_id, user["id"]):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "找不到該對話")
    return {"deleted": True}


@router.get("/sessions/{session_id}/messages")
def messages(session_id: int, user: dict = Depends(current_user)) -> dict:
    """**強制以 user_id 過濾**，拿不到別人的對話——管理員也不例外。"""
    rows = chat_service.get_messages(session_id, user["id"])
    for row in rows:
        row["sources"] = chat_service.resolve_sources(row["source_chunk_ids"])
        row["created_at"] = row["created_at"].isoformat()
    return {"messages": rows}


@router.get("/engine")
def engine(_: dict = Depends(current_user)) -> dict:
    status_ = ollama_client.check_status()
    return {
        "alive": status_.alive,
        "message": status_.message,
        "supports_tools": ollama_client.supports_tools() if status_.alive else False,
    }


def _sse(event: str, payload: dict) -> str:
    """組出一則 SSE 事件。

    **結尾的兩個換行不能少**——少一個，整條串流就不會觸發，
    瀏覽器會一直等下去而不報錯，非常難查。
    """
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/ask")
def ask(body: AskBody, user: dict = Depends(current_user)) -> StreamingResponse:
    history = chat_service.get_messages(body.session_id, user["id"])
    chat_service.add_message(body.session_id, user["id"], "user", body.question)

    def stream():
        answer, chunk_ids = "", []
        try:
            for event in agent_service.answer(body.question, history, body.stage_code,
                                              body.wide, body.broad):
                kind = event["type"]
                if kind == "search":
                    yield _sse("search", {
                        "query": event["query"],
                        "stage": event["stage"],
                        "hits": event["hits"],
                        "broad": event.get("broad", False),
                    })
                elif kind == "text":
                    yield _sse("text", {"piece": event["piece"]})
                elif kind == "error":
                    yield _sse("error", {"message": event["message"]})
                    return
                elif kind == "done":
                    # **一律以 done 事件的 answer 為準，不要用串流累積的 buffer。**
                    # 未檢索前產生的文字不會經由 text 事件送出（Agent 的防護），
                    # 用 buffer 當最終答案等於繞過那層防護。
                    answer = event["answer"]
                    chunk_ids = event["chunk_ids"]
        finally:
            if answer:
                chat_service.add_message(
                    body.session_id, user["id"], "assistant", answer, chunk_ids
                )
        yield _sse("done", {
            "answer": answer,
            "sources": chat_service.resolve_sources(chunk_ids),
        })

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 沒有這一行，反向代理會把整條串流緩衝到結束才一次送出，
            # 逐字輸出的效果就完全消失了。
            "X-Accel-Buffering": "no",
        },
    )
