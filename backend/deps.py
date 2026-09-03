"""共用依賴：JWT 簽發／驗證與角色守衛。

**權限檢查一律在這一層，前端隱藏選單只是體驗不是安全機制。**
前端可以被繞過——把 URL 改掉就能直接打端點。
"""

from __future__ import annotations

import os
import secrets
import sys
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_session, get_setting, set_setting  # noqa: E402
from models import ROLE_ADMIN, User  # noqa: E402

ALGORITHM = "HS256"
TOKEN_HOURS = 12

# 金鑰在資料庫裡的鍵名。**不會出現在任何 API 回應中**——
# `/api/admin/models` 只回傳指名的那幾個鍵，沒有端點會傾印整張 app_settings。
_SECRET_KEY = "jwt_secret"
_secret_cache = ""


def _secret() -> str:
    """取得 JWT 簽章金鑰。順序：環境變數 → 資料庫 → 首次啟動時自動產生。

    **這裡原本是一個寫死在原始碼裡的固定字串**，註解寫著「正式部署務必改為由
    環境變數提供」。但那是寫給人看的提醒，不是機制：`start.bat` 沒有設那個變數，
    而原始碼是公開的 repo——等於那串金鑰是公開資訊，任何人都能自己簽一張
    role=ADMIN 的 token，不需要帳號密碼。

    改成第一次啟動時自動產生並存進資料庫：
      * 每台機器的金鑰都不同，公開的原始碼不再是通行證
      * 存在資料庫所以重啟不會把所有人踢出去（那正是當初用固定值的理由）
      * 使用者什麼都不用做——要求大家去設環境變數，實際上沒有人會做

    仍然保留環境變數，讓需要多台機器共用金鑰的部署可以覆寫。
    延遲到第一次使用才解析：模組載入時 `init_db()` 還沒跑，資料表不存在。
    """
    global _secret_cache
    if _secret_cache:
        return _secret_cache

    from_env = os.environ.get("KB_JWT_SECRET", "").strip()
    if from_env:
        _secret_cache = from_env
        return _secret_cache

    stored = get_setting(_SECRET_KEY)
    if not stored:
        stored = secrets.token_urlsafe(48)
        set_setting(_SECRET_KEY, stored)
    _secret_cache = stored
    return _secret_cache

_bearer = HTTPBearer(auto_error=False)


def make_token(user_id: int, username: str, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS),
    }
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM)


def current_user(
    cred: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    if cred is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "尚未登入")
    try:
        payload = jwt.decode(cred.credentials, _secret(), algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "登入已過期，請重新登入")

    # 每次都回資料庫確認帳號仍存在且啟用——
    # token 本身無法反映「管理員剛把這個帳號停用」這件事。
    with get_session() as session:
        user = session.get(User, int(payload["sub"]))
        if not user or not user.is_active:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "帳號不存在或已停用")
        return {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role,
        }


def require_admin(user: dict = Depends(current_user)) -> dict:
    if user["role"] != ROLE_ADMIN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "此操作僅限管理員")
    return user
