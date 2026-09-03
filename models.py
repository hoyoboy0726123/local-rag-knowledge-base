"""ORM 模型定義。

部署型態為區網共用，因此：
  - 使用者需登入，區分 ADMIN（可管理文件）與 USER（僅查詢）
  - 對話紀錄按 user_id 分開保存，任何人都只看得到自己的
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


ROLE_ADMIN = "ADMIN"
ROLE_USER = "USER"

ROLE_LABELS = {
    ROLE_ADMIN: "知識庫管理員",
    ROLE_USER: "一般使用者",
}

# NUC Stage-Gate 六大階段
STAGES = [
    ("Concept", "概念階段", 1),
    ("Plan", "規劃階段", 2),
    ("EVT", "工程驗證", 3),
    ("DVT", "設計驗證", 4),
    ("PVT", "量產驗證", 5),
    ("MP", "量產階段", 6),
]

# 文件解析狀態
DOC_INDEXED = "indexed"
DOC_FAILED = "failed"
DOC_PENDING = "pending"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    salt: Mapped[str] = mapped_column(String(64))
    display_name: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(16), default=ROLE_USER)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    must_change_pwd: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Stage(Base):
    __tablename__ = "stages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True)
    name_zh: Mapped[str] = mapped_column(String(64))
    seq: Mapped[int] = mapped_column(Integer)
    description: Mapped[str] = mapped_column(Text, default="")
    deliverables: Mapped[str] = mapped_column(Text, default="[]")  # JSON 陣列


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_path: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    file_name: Mapped[str] = mapped_column(String(512))
    file_type: Mapped[str] = mapped_column(String(16))
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64), default="", index=True)
    modified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    stage_code: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default=DOC_PENDING)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    used_vlm: Mapped[bool] = mapped_column(Boolean, default=False)

    # 索引當下解析出來的完整 Markdown。
    #
    # **存下來而不是閱讀時重新解析**，兩個理由：
    #   1. 正確性：解析結果取決於「當時」的 VLM 設定（全域開關 + 該檔的強制視覺解析）。
    #      閱讀時用另一組設定重跑，看到的就不是 AI 讀到的那份東西——
    #      實測就發生過：切片裡有 VLM 產生的圖片描述，閱讀頁卻只有文字層。
    #   2. 速度：開了視覺解析的檔案每頁要 3–11 秒，重跑一次可能要好幾分鐘。
    #      閱讀一份文件不該付這個代價。
    content: Mapped[str] = mapped_column(Text, default="")


class Chunk(Base):
    """文本切片。id 同時作為 sqlite-vec 虛擬表的鍵。"""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text)
    locator: Mapped[str] = mapped_column(String(128), default="")
    char_count: Mapped[int] = mapped_column(Integer, default=0)

    # 管理員補的檢索關鍵字。**只影響向量，不影響顯示與作答。**
    #
    # 為什麼不讓管理員直接改 content：
    #   1. content 同時是給使用者看的「原文」與給 LLM 的證據，
    #      改了它，來源卡片就會顯示原始文件裡沒有的文字，破壞可追溯性。
    #   2. 向量是索引當下用 content 算的，事後改 content 不會重算向量——
    #      管理員以為調好了，其實檢索完全沒變。
    # 拆成獨立欄位後，改動會明確觸發「只重算這一個切片的向量」。
    keywords: Mapped[str] = mapped_column(Text, default="")

    # 管理員編輯 content 前的原始解析結果。**只在第一次編輯時寫入，之後不再變動。**
    #
    # 有了它，「編輯切片內容」才不會破壞可追溯性：
    # 隨時能比對「檔案原本解析出什麼」與「管理員改成什麼」，也能一鍵還原。
    # 空字串代表從未被編輯過。
    original_content: Mapped[str] = mapped_column(Text, default="")


class DocOption(Base):
    """單一檔案的解析選項。

    **以 `file_path` 為鍵而不是 `documents.id`**，因為選項必須在建索引**之前**
    就能設定——檔案剛上傳時還沒有 Document 列，而「要不要用視覺模型解析」
    正是上傳當下最該決定的事。
    """

    __tablename__ = "doc_options"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_path: Mapped[str] = mapped_column(String(1024), unique=True, index=True)

    # 強制以視覺模型補讀圖片內容。
    # 預設關閉：VLM 每張圖都要跑一次推論，對純文字文件是純浪費。
    force_vlm: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class ChunkKeyword(Base):
    """切片人工調整的持久化備份（關鍵字 **與編輯過的內容**）。

    切片在每次重新索引時都會被刪除重建（換新 id），
    因此關鍵字必須另存一份，索引完成後再套回去，
    否則管理員調校的成果會在下一次重建時無聲消失。

    以 `(file_path, seq)` 對應，另存 `content_head` 作為指紋：
    文件內容變了就不套用，避免把關鍵字接到不相干的段落上。
    """

    __tablename__ = "chunk_keywords"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_path: Mapped[str] = mapped_column(String(512), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    content_head: Mapped[str] = mapped_column(String(120), default="")
    keywords: Mapped[str] = mapped_column(Text, default="")

    # 管理員編輯後的切片內容。空字串代表沒編輯過、照解析結果走。
    #
    # 跟關鍵字放同一張表是刻意的：兩者的鍵完全相同（file_path + seq + content_head 指紋），
    # 拆成兩張表只會讓「重新索引後套回去」的邏輯要跑兩遍。
    edited_content: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class ChatSession(Base):
    """對話會話。按使用者分開，任何人只看得到自己的。"""

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(255), default="新對話")
    stage_filter: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user / assistant
    content: Mapped[str] = mapped_column(Text)
    source_chunk_ids: Mapped[str] = mapped_column(Text, default="[]")  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class IngestError(Base):
    __tablename__ = "ingest_errors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_path: Mapped[str] = mapped_column(String(1024))
    file_name: Mapped[str] = mapped_column(String(512), default="")
    error_type: Mapped[str] = mapped_column(String(64))
    message: Mapped[str] = mapped_column(Text, default="")
    occurred_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


DEFAULT_SETTINGS = {
    "knowledge_root": "",
    # 用 127.0.0.1 而非 localhost：Windows 會先解析到 IPv6 ::1，
    # 但 Ollama 只監聽 IPv4，每個請求都要卡滿逾時才退回，固定多付約 2 秒。
    "ollama_host": "http://127.0.0.1:11434",
    "embed_model": "quentinz/bge-large-zh-v1.5:latest",
    "llm_model": "gemma4:12b",
    "vlm_model": "qwen3-vl:8b-instruct",
    "enable_vlm": "1",
    "chunk_size": "500",
    "chunk_overlap": "80",
    "top_k": "6",
    # 送進 Ollama 的上下文視窗。**預設值必須跟 ollama_client.DEFAULT_NUM_CTX
    # 一致**——兩邊分開放著遲早會對不上，而不一致的後果是安靜的：提示詞超過
    # 視窗時 Ollama 直接截掉前面的內容，不會報錯，只是答案突然變得不對。
    "num_ctx": "8192",
    "blocklist": "",
    "db_version": "1",
}

# 預設 embedding 模型（bge-large-zh-v1.5）的向量維度。
#
# **只是預設值，不是固定值。** 實際維度取決於當前使用的 embedding 模型，
# 由 `database.ensure_vec_table()` 在建立索引前向模型問出來，
# 與現有向量表不符時會重建該表。
#
# 曾經把這個值寫死並直接用來建表，結果管理員在 UI 換成 4096 維的
# qwen3-embedding:8b 後，全量重建會**先清空索引再全部失敗**
# （Dimension mismatch，12/12），留下一個空的知識庫。
EMBED_DIM = 1024
