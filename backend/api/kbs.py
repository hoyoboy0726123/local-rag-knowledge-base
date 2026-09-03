"""知識庫清單、文件閱讀與下載。"""

from __future__ import annotations

import urllib.parse

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel

from backend.deps import current_user
from database import get_setting
from services import ingest_service, kb_service

router = APIRouter(prefix="/api", tags=["kbs"])


class OpenBody(BaseModel):
    path: str


def _docs(df) -> list[dict]:
    return [] if df.empty else df.to_dict("records")


@router.get("/kbs")
def list_kbs(_: dict = Depends(current_user)) -> dict:
    """知識庫清單。**「通用」永遠在第一個**，它就是根目錄的檔案，不是資料夾。

    清單以磁碟為準（根目錄下的子資料夾），文件數以索引為準——
    使用者用檔案總管新建的資料夾也會出現，只是文件數是 0，直到建索引。
    """
    root = get_setting("knowledge_root", "")
    counts = kb_service.kb_doc_counts()
    names = ingest_service.list_kb_names(root)
    # 索引裡有、資料夾卻不見了的（例如被使用者手動刪掉）也要列，否則那些文件沒有入口
    for name in counts:
        if name and name not in names:
            names.append(name)
    kbs = [{"name": "", "label": "通用", "is_general": True, "doc_count": counts.get(None, 0)}]
    kbs += [{"name": n, "label": n, "is_general": False, "doc_count": counts.get(n, 0)}
            for n in sorted(names, key=str.casefold)]
    return {"kbs": kbs, "stats": kb_service.library_stats()}


@router.get("/kbs/documents")
def kb_documents(kb: str | None = None, _: dict = Depends(current_user)) -> dict:
    """某個知識庫的已索引文件。`kb=`（空字串）是通用；不帶參數是全部。"""
    return {"documents": _docs(kb_service.get_kb_documents(kb))}


@router.get("/documents/content")
def document_content(path: str, _: dict = Depends(current_user)) -> dict:
    content, error = kb_service.read_document(path)
    if error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, error)
    return {"content": content, "char_count": len(content)}


@router.get("/documents/download")
def document_download(
    path: str,
    inline: bool = False,
    _: dict = Depends(current_user),
) -> Response:
    """取得原始檔。`inline=true` 供「在新分頁開啟」使用。

    兩種模式差在 media type，不在檔案內容：
      * 預設（下載）：`application/octet-stream` + `attachment`，強制存檔。
      * `inline=true`：回真實 MIME（如 `application/pdf`），瀏覽器才會用
        內建檢視器顯示而不是下載。給錯 MIME 的話 PDF 一樣會被存下來。

    **`inline` 只是請求，不是命令。** 格式不在 `INLINE_VIEWABLE` 白名單內時
    一律退回 attachment：以真實 MIME 送出 `.svg`／`.html` 會讓其中的腳本
    在本應用的 origin 下執行（前端用 blob URL 開新分頁，而 blob URL 繼承來源），
    等於上傳一個檔案就能竊取其他人的 JWT。前端已經不會對這些格式送 inline，
    但把關必須在後端，否則手打一個網址就繞過去了。

    **不用 `os.startfile()`**——本系統可能部署在區網伺服器上，
    那個呼叫會在伺服器上開檔，遠端使用者點下去什麼也不會發生。
    真的要用本機程式開啟請走 `/documents/open`（僅限 localhost）。
    """
    data = kb_service.read_document_bytes(path)
    if data is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "檔案已不存在或不在知識庫索引中")
    name = urllib.parse.quote(path.replace("\\", "/").rsplit("/", 1)[-1])
    serve_inline = inline and kb_service.can_inline(path)
    disposition = "inline" if serve_inline else "attachment"
    media = kb_service.media_type_of(path) if serve_inline else "application/octet-stream"
    return Response(
        content=data,
        media_type=media,
        headers={"Content-Disposition": f"{disposition}; filename*=UTF-8''{name}"},
    )


# 迴圈位址。區網來的請求一律不給碰 `/documents/open`。
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}


def _is_local_request(request: Request) -> bool:
    client = request.client
    return bool(client) and client.host in _LOCAL_HOSTS


@router.get("/client/is-local")
def client_is_local(request: Request, _: dict = Depends(current_user)) -> dict:
    """前端用這個決定要不要顯示「用本機程式開啟」。

    純粹是介面提示，不是權限來源——真正的把關在 `/documents/open` 自己。
    區網使用者就算偽造這個回應，那支端點還是會擋下來。
    """
    return {"is_local": _is_local_request(request)}


@router.post("/documents/open")
def document_open(request: Request, body: OpenBody, _: dict = Depends(current_user)) -> dict:
    """用**執行後端那台機器**的預設程式開啟檔案。

    因此只有請求來自 localhost 時才允許——這時瀏覽器與後端在同一台電腦上，
    「伺服器上開檔」正好就是使用者自己的螢幕。區網使用者呼叫會拿到 403，
    而不是一個點了沒反應的按鈕。
    """
    if not _is_local_request(request):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "這個功能只能在執行本系統的那台電腦上使用。請改用下載。",
        )
    ok, error = kb_service.open_with_local_app(body.path)
    if not ok:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, error)
    return {"opened": True}
