"""認證端點。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.deps import current_user, make_token
from services import auth_service

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginBody(BaseModel):
    username: str
    password: str


class PasswordBody(BaseModel):
    old_password: str
    new_password: str


@router.post("/login")
def login(body: LoginBody) -> dict:
    user = auth_service.verify_user(body.username, body.password)
    if not user:
        # 不要區分「帳號不存在」與「密碼錯誤」——那等於幫人確認帳號存不存在。
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "帳號或密碼錯誤")
    return {
        "token": make_token(user["id"], user["username"], user["role"]),
        "user": user,
    }


@router.get("/me")
def me(user: dict = Depends(current_user)) -> dict:
    return user


@router.post("/password")
def change_password(body: PasswordBody, user: dict = Depends(current_user)) -> dict:
    ok, message = auth_service.change_password(
        user["id"], body.old_password, body.new_password
    )
    if not ok:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, message)
    return {"message": message}
