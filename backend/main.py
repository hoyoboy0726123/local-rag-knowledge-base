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

from backend.api import admin, auth, chat, kbs  # noqa: E402
from database import get_session, init_db  # noqa: E402

app = FastAPI(
    title="本機知識庫 API",
    description="React + FastAPI 版。所有 AI 運算都在本機完成。",
    version="2.0.0",
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(kbs.router)
app.include_router(admin.router)


# 預設密碼。與 `seed_data.DEMO_PASSWORD` 相同，但**不從那裡 import**——
# seed_data 會拉進整套建立範例文件的相依，只為了比對一個字串不值得。
DEMO_PASSWORD = "demo1234"


def _warn_default_passwords() -> None:
    """啟動時檢查還有誰在用預設密碼，直接印在主控台。

    只警告不強制。強制改密碼會擋住「發下去讓大家先試用」這個情境，
    而這正是這套系統目前的用途。但**預設密碼寫在公開的 README 裡**，
    掛上區網之後等於門是開的——所以警告要夠明顯，不能只寫在文件裡。
    """
    from services.auth_service import hash_password
    from models import User

    with get_session() as session:
        weak = [
            u.username for u in session.query(User).filter(User.is_active).all()
            if hash_password(DEMO_PASSWORD, u.salt) == u.password_hash
        ]
    if not weak:
        return
    # **全部用 ASCII，而且包在 try 裡。** 這段曾經印了中文，stderr 被導向檔案時
    # Python 走 cp1252 就 UnicodeEncodeError，整個 startup 失敗、服務起不來——
    # 一則警告把主程式拖垮，本末倒置。警告再重要也不能有讓啟動失敗的可能。
    try:
        bar = "!" * 72
        print(bar, flush=True)
        print(f"  WARNING: {len(weak)} account(s) still use the default password "
              f"'{DEMO_PASSWORD}'", flush=True)
        print(f"    {', '.join(weak)}", flush=True)
        print("  This password is published in README.md. Anyone on the LAN can "
              "log in.", flush=True)
        print("  Change it in the app: Profile page -> Change password", flush=True)
        print(bar, flush=True)
    except Exception:  # noqa: BLE001
        pass


@app.on_event("startup")
def startup() -> None:
    init_db()
    _warn_default_passwords()


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
