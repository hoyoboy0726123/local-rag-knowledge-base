"""共用依賴：JWT 簽發／驗證與角色守衛。

**權限檢查一律在這一層，前端隱藏選單只是體驗不是安全機制。**
前端可以被繞過——把 URL 改掉就能直接打端點。
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_session  # noqa: E402
from models import ROLE_ADMIN, User  # noqa: E402

# 展示用專案：金鑰以環境變數覆寫，未設定時用固定值以免每次重啟就把所有人踢出去。
# 正式部署務必改為由環境變數提供。
SECRET = os.environ.get("KB_JWT_SECRET", "leslie-v2-dev-secret-change-in-production")
ALGORITHM = "HS256"
TOKEN_HOURS = 12

_bearer = HTTPBearer(auto_error=False)


def make_token(user_id: int, username: str, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS),
    }
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)


def current_user(
    cred: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    if cred is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "尚未登入")
    try:
        payload = jwt.decode(cred.credentials, SECRET, algorithms=[ALGORITHM])
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
