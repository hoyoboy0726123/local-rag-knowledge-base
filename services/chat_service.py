"""對話紀錄。

**每個人只看得到自己的對話。** 所有查詢一律以 user_id 過濾，
連管理員也不例外——對話可能包含 PM 對專案的個人判斷，屬於私人工作紀錄。
管理員能看到的是使用量統計，不是對話內文。
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
from sqlalchemy import func

from database import get_session
from models import ChatMessage, ChatSession, Chunk, Document, User


def create_session(user_id: int, kb_filter: list[str] | None = None) -> int:
    with get_session() as session:
        record = ChatSession(user_id=user_id,
                             kb_filter=json.dumps(kb_filter, ensure_ascii=False) if kb_filter else None)
        session.add(record)
        session.commit()
        return record.id


def list_sessions(user_id: int, limit: int = 50) -> pd.DataFrame:
    """只回傳該使用者自己的會話。"""
    with get_session() as session:
        rows = (
            session.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .limit(limit)
            .all()
        )
        counts = dict(
            session.query(ChatMessage.session_id, func.count(ChatMessage.id))
            .filter(ChatMessage.user_id == user_id)
            .group_by(ChatMessage.session_id)
            .all()
        )

    return pd.DataFrame(
        [
            {
                "ID": s.id,
                "標題": s.title,
                "範圍": "、".join(k or "通用" for k in json.loads(s.kb_filter)) if s.kb_filter else "全部",
                "訊息數": counts.get(s.id, 0),
                "最後更新": s.updated_at.strftime("%Y-%m-%d %H:%M"),
            }
            for s in rows
        ]
    )


def get_messages(session_id: int, user_id: int) -> list[dict]:
    """取得訊息。**強制以 user_id 過濾**，拿不到別人的對話。"""
    with get_session() as session:
        owner = session.get(ChatSession, session_id)
        if not owner or owner.user_id != user_id:
            return []

        rows = (
            session.query(ChatMessage)
            .filter(
                ChatMessage.session_id == session_id,
                ChatMessage.user_id == user_id,
            )
            .order_by(ChatMessage.id)
            .all()
        )

    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "source_chunk_ids": json.loads(m.source_chunk_ids or "[]"),
            "created_at": m.created_at,
        }
        for m in rows
    ]


def add_message(
    session_id: int,
    user_id: int,
    role: str,
    content: str,
    source_chunk_ids: list[int] | None = None,
) -> None:
    with get_session() as session:
        owner = session.get(ChatSession, session_id)
        if not owner or owner.user_id != user_id:
            return  # 不是自己的會話，不寫入

        session.add(
            ChatMessage(
                session_id=session_id,
                user_id=user_id,
                role=role,
                content=content,
                source_chunk_ids=json.dumps(source_chunk_ids or []),
            )
        )
        # 以第一句提問作為會話標題
        if role == "user" and owner.title == "新對話":
            owner.title = content[:40] + ("..." if len(content) > 40 else "")
        owner.updated_at = datetime.now()
        session.commit()


def delete_session(session_id: int, user_id: int) -> bool:
    with get_session() as session:
        owner = session.get(ChatSession, session_id)
        if not owner or owner.user_id != user_id:
            return False
        session.query(ChatMessage).filter(
            ChatMessage.session_id == session_id,
            ChatMessage.user_id == user_id,
        ).delete()
        session.delete(owner)
        session.commit()
        return True


def clear_all(user_id: int) -> int:
    """清除該使用者的所有對話紀錄。"""
    with get_session() as session:
        ids = [
            s.id for s in session.query(ChatSession).filter(ChatSession.user_id == user_id).all()
        ]
        if ids:
            session.query(ChatMessage).filter(ChatMessage.user_id == user_id).delete()
            session.query(ChatSession).filter(ChatSession.user_id == user_id).delete()
            session.commit()
        return len(ids)


def resolve_sources(chunk_ids: list[int]) -> list[dict]:
    """把切片 ID 還原為可顯示的來源卡片。"""
    if not chunk_ids:
        return []

    with get_session() as session:
        rows = (
            session.query(Chunk, Document)
            .join(Document, Chunk.doc_id == Document.id)
            .filter(Chunk.id.in_(chunk_ids))
            .all()
        )

    order = {cid: i for i, cid in enumerate(chunk_ids)}
    results = [
        {
            "index": order.get(chunk.id, 999) + 1,
            "file_name": doc.file_name,
            "file_path": doc.file_path,
            "kb": doc.kb,
            "locator": chunk.locator,
            "content": chunk.content,
        }
        for chunk, doc in rows
    ]
    return sorted(results, key=lambda r: r["index"])


def usage_stats() -> pd.DataFrame:
    """管理員可見的使用量統計 —— 只有次數，沒有對話內容。"""
    with get_session() as session:
        rows = (
            session.query(
                User.display_name,
                User.username,
                func.count(func.distinct(ChatSession.id)),
                func.count(ChatMessage.id),
                func.max(ChatMessage.created_at),
            )
            .outerjoin(ChatSession, ChatSession.user_id == User.id)
            .outerjoin(ChatMessage, ChatMessage.user_id == User.id)
            .group_by(User.id)
            .all()
        )

    return pd.DataFrame(
        [
            {
                "使用者": name,
                "帳號": username,
                "對話數": sessions or 0,
                "訊息數": messages or 0,
                "最後使用": last.strftime("%Y-%m-%d %H:%M") if last else "從未使用",
            }
            for name, username, sessions, messages, last in rows
        ]
    )
