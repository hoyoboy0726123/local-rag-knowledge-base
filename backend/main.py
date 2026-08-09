"""FastAPI 進入點。

**單一行程**：前端在安裝階段建置一次，之後由這裡提供 `frontend/dist`
的靜態檔。不要在正式啟動時同時跑 `npm run dev` 與 uvicorn——
兩個行程還要處理跨埠 CORS，沒有任何好處。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # 讓相對路徑（knowledge.db、sample_knowledge_base）一致

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from backend.api import admin, auth, chat, stages  # noqa: E402
from database import init_db  # noqa: E402

app = FastAPI(
    title="本機知識庫 API",
    description="React + FastAPI 版。所有 AI 運算都在本機完成。",
    version="2.0.0",
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(stages.router)
app.include_router(admin.router)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": "2.0.0"}


DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    # index.html **必須每次重新驗證**。
    # 檔名沒有雜湊，而它裡面寫死了帶雜湊的 JS/CSS 檔名；
    # 瀏覽器一旦把它快取起來，重新建置後使用者仍然載入舊的 bundle，
    # 而且看不出任何異狀——只會覺得「改的東西沒生效」。
    # /assets 底下的檔名本身帶雜湊，可以放心長期快取。
    NO_STORE = {"Cache-Control": "no-cache, must-revalidate"}

    @app.get("/{path:path}")
    def spa(path: str):
        """SPA fallback：非 API 路徑一律回 index.html，交給前端路由處理。"""
        candidate = DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST / "index.html", headers=NO_STORE)
