"""資料庫連線、sqlite-vec 載入與初始化。

向量與一般資料表放在**同一個 .db 檔**，這是選用 sqlite-vec 而非 ChromaDB
的主要理由：檢索時可以直接 JOIN documents 做 stage 過濾，
備份也只要複製一個檔案。
"""

from __future__ import annotations

import re
import sqlite3
import struct
from pathlib import Path

import sqlite_vec
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from models import DEFAULT_SETTINGS, EMBED_DIM, AppSetting, Base

DB_PATH = Path(__file__).parent / "knowledge.db"
_engine = None
_SessionFactory = None


def _load_vec_extension(dbapi_conn) -> None:
    """每條新連線都要重新載入擴充，這是 sqlite-vec 唯一的設置成本。"""
    dbapi_conn.enable_load_extension(True)
    sqlite_vec.load(dbapi_conn)
    dbapi_conn.enable_load_extension(False)


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            f"sqlite:///{DB_PATH}",
            future=True,
            connect_args={"check_same_thread": False, "timeout": 15},
        )

        @event.listens_for(_engine, "connect")
        def _on_connect(dbapi_conn, _record):
            _load_vec_extension(dbapi_conn)
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=15000")
            cursor.close()

    return _engine


def get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), future=True, expire_on_commit=False)
    return _SessionFactory()


def raw_connection() -> sqlite3.Connection:
    """取得已載入 sqlite-vec 的原生連線，供向量操作使用。"""
    conn = sqlite3.connect(DB_PATH, timeout=15)
    _load_vec_extension(conn)
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def vec_table_dim() -> int | None:
    """現有 vec_chunks 表是以幾維建立的。表不存在時回傳 None。"""
    with get_engine().begin() as conn:
        row = conn.execute(
            text("SELECT sql FROM sqlite_master WHERE name='vec_chunks'")
        ).fetchone()
    if not row or not row[0]:
        return None
    match = re.search(r"FLOAT\[(\d+)\]", row[0])
    return int(match.group(1)) if match else None


def ensure_vec_table(dim: int) -> bool:
    """確保 vec_chunks 表的維度是 `dim`。維度不符時重建，回傳是否重建過。

    向量表的維度在建表時就固定了，換 embedding 模型若維度不同就塞不進去。
    這裡在建立索引前先對齊，避免出現「先清空舊索引、再全部寫入失敗」
    而留下一個空知識庫的情況。

    重建會清掉所有向量，但呼叫端本來就是要重新產生向量，所以是安全的。
    """
    current = vec_table_dim()
    with get_engine().begin() as conn:
        if current == dim:
            return False
        if current is not None:
            conn.execute(text("DROP TABLE vec_chunks"))
        conn.execute(
            text(
                f"CREATE VIRTUAL TABLE vec_chunks "
                f"USING vec0(chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{dim}])"
            )
        )
    return current is not None


def _migrate_columns() -> None:
    """補上既有資料庫缺少的欄位。

    `create_all()` 只會建立不存在的**資料表**，不會替既有資料表加欄位。
    已經在跑的知識庫不該為了一個新欄位就得重建整份索引，
    因此這裡用 `ALTER TABLE` 逐項補齊。
    """
    wanted = {
        ("chunks", "keywords"): "TEXT DEFAULT ''",
        ("chunks", "original_content"): "TEXT DEFAULT ''",
        ("chunk_keywords", "edited_content"): "TEXT DEFAULT ''",
        ("documents", "content"): "TEXT DEFAULT ''",
    }
    with get_engine().begin() as conn:
        for (table, column), ddl in wanted.items():
            existing = {
                row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")
            }
            if column not in existing:
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db() -> None:
    Base.metadata.create_all(get_engine())
    _migrate_columns()

    with get_engine().begin() as conn:
        conn.execute(
            text(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks "
                f"USING vec0(chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{EMBED_DIM}])"
            )
        )

    with get_session() as session:
        for key, value in DEFAULT_SETTINGS.items():
            if not session.get(AppSetting, key):
                session.add(AppSetting(key=key, value=value))
        session.commit()


def serialize(vector: list[float]) -> bytes:
    """把向量打包成 sqlite-vec 需要的二進位格式。"""
    return struct.pack(f"{len(vector)}f", *vector)


def get_setting(key: str, default: str = "") -> str:
    with get_session() as session:
        setting = session.get(AppSetting, key)
        return setting.value if setting else default


def get_int_setting(key: str, default: int) -> int:
    try:
        return int(get_setting(key, str(default)) or default)
    except (TypeError, ValueError):
        return default


def set_setting(key: str, value: str) -> None:
    from datetime import datetime

    with get_session() as session:
        setting = session.get(AppSetting, key)
        if setting:
            setting.value = str(value)
            setting.updated_at = datetime.now()
        else:
            session.add(AppSetting(key=key, value=str(value)))
        session.commit()
