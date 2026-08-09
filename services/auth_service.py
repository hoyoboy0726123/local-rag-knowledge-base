"""登入驗證與角色權限。

區網共用部署，因此權限判斷一律放在 service 層，
UI 隱藏按鈕只是輔助，不能當作唯一防線。
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime

import pandas as pd

from database import get_session
from models import ROLE_ADMIN, ROLE_LABELS, ROLE_USER, User


class PermissionDeniedError(Exception):
    pass


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()


def new_salt() -> str:
    return os.urandom(16).hex()


def verify_user(username: str, password: str) -> dict | None:
    with get_session() as session:
        user = session.query(User).filter(
            User.username == username.strip().lower()
        ).first()
        if not user or not user.is_active:
            return None
        if hash_password(password, user.salt) != user.password_hash:
            return None
        user.last_login_at = datetime.now()
        session.commit()
        return {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role,
            "must_change_pwd": user.must_change_pwd,
        }


# V2 與 V1 在這一段的差別：
#
# V1 是 Streamlit，登入狀態存在 `st.session_state`，
# `require_login()` 直接 `st.stop()` 中斷頁面。
# V2 是前後端分離，登入狀態由 **JWT** 攜帶，
# 權限守衛改用 FastAPI 的依賴注入（見 `backend/deps.py`）。
#
# 因此本模組只保留**與框架無關**的部分：雜湊、驗證帳密、帳號管理。
# 「目前是誰」與「擋不擋下來」交給呼叫端決定。


def is_admin(user: dict | None) -> bool:
    return bool(user and user["role"] == ROLE_ADMIN)



def assert_admin(user: dict) -> None:
    """service 層用的權限斷言，不依賴 Streamlit。"""
    if not user or user.get("role") != ROLE_ADMIN:
        raise PermissionDeniedError("此操作僅限知識庫管理員")


# ================================================================ 帳號管理
def list_users() -> pd.DataFrame:
    with get_session() as session:
        rows = session.query(User).order_by(User.role, User.username).all()
    return pd.DataFrame(
        [
            {
                "ID": u.id,
                "帳號": u.username,
                "顯示名稱": u.display_name,
                "角色": ROLE_LABELS.get(u.role, u.role),
                "啟用": u.is_active,
                "最後登入": u.last_login_at.strftime("%Y-%m-%d %H:%M")
                if u.last_login_at else "從未登入",
            }
            for u in rows
        ]
    )


def create_user(
    username: str, password: str, display_name: str, role: str, operator: dict
) -> tuple[bool, str]:
    assert_admin(operator)

    username = username.strip().lower()
    if not username:
        return False, "帳號不可為空"
    if len(password) < 8:
        return False, "密碼長度至少 8 碼"
    if role not in (ROLE_ADMIN, ROLE_USER):
        return False, "角色不正確"

    with get_session() as session:
        if session.query(User).filter(User.username == username).first():
            return False, f"帳號 {username} 已存在"
        salt = new_salt()
        session.add(
            User(
                username=username,
                password_hash=hash_password(password, salt),
                salt=salt,
                display_name=display_name.strip() or username,
                role=role,
                must_change_pwd=True,
            )
        )
        session.commit()
    return True, f"帳號 {username} 已建立（首次登入需修改密碼）"


def toggle_active(user_id: int, operator: dict) -> tuple[bool, str]:
    assert_admin(operator)
    with get_session() as session:
        user = session.get(User, user_id)
        if not user:
            return False, "找不到使用者"
        if user.id == operator["id"] and user.is_active:
            return False, "不能停用自己的帳號"
        user.is_active = not user.is_active
        session.commit()
        return True, f"{user.username} 已{'啟用' if user.is_active else '停用'}"


def change_password(user_id: int, old: str, new: str) -> tuple[bool, str]:
    if len(new) < 8:
        return False, "新密碼長度至少 8 碼"
    with get_session() as session:
        user = session.get(User, user_id)
        if not user:
            return False, "找不到使用者"
        if hash_password(old, user.salt) != user.password_hash:
            return False, "目前密碼不正確"
        user.salt = new_salt()
        user.password_hash = hash_password(new, user.salt)
        user.must_change_pwd = False
        session.commit()
    return True, "密碼已更新"


def reset_password(user_id: int, new_password: str, operator: dict) -> tuple[bool, str]:
    assert_admin(operator)
    if len(new_password) < 8:
        return False, "密碼長度至少 8 碼"
    with get_session() as session:
        user = session.get(User, user_id)
        if not user:
            return False, "找不到使用者"
        user.salt = new_salt()
        user.password_hash = hash_password(new_password, user.salt)
        user.must_change_pwd = True
        session.commit()
        return True, f"{user.username} 的密碼已重設（下次登入需修改）"
