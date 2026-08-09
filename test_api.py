"""V2 API 層測試。

V1 的 `test_core.py` 驗的是 services 層（檢索、Agent、Grounding、對話隔離），
那些邏輯在 V2 完全沿用，**不需要重測**。
這裡只驗 V2 新增的部分：JWT 認證、角色守衛、SSE 串流、API 契約。

執行前請先啟動後端：
    venv\\Scripts\\python.exe -m uvicorn backend.main:app --port 8600
執行：
    venv\\Scripts\\python.exe test_api.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

BASE = "http://127.0.0.1:8600"
_passed = _failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {label}")
    else:
        _failed += 1
        print(f"  [FAIL] {label}  {detail}")


def call(path: str, token: str = "", method: str = "GET", body: dict | None = None):
    """回傳 (狀態碼, 內容)。錯誤不拋例外，才能直接斷言狀態碼。"""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()
            # 首頁回的是 HTML，不是 JSON——一律 json.loads 會讓成功的請求
            # 被當成連線失敗（狀態變 0），實測就這樣誤報過一次。
            try:
                return resp.status, json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return resp.status, {"_raw": raw[:200]}
    except urllib.error.HTTPError as exc:
        return exc.code, {}
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": str(exc)}


def sse(path: str, token: str, body: dict) -> dict:
    """收完一條 SSE 串流，回傳彙整結果。"""
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")

    events, texts, done = [], 0, None
    name = None
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                name = line[7:]
                events.append(name)
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
                if name == "text":
                    texts += 1
                elif name == "done":
                    done = payload
    return {"events": events, "texts": texts, "done": done}


print("\n=== 1. 服務可用性 ===")
status, health = call("/api/health")
if status != 200:
    print("  [FAIL] 後端未啟動。請先執行：")
    print("         venv\\Scripts\\python.exe -m uvicorn backend.main:app --port 8600")
    sys.exit(1)
check("health 端點正常", health.get("status") == "ok", str(health))
check("首頁提供前端靜態檔", call("/")[0] == 200)

print("\n=== 2. 認證 ===")
check("未帶 token 一律 401", call("/api/stages")[0] == 401)
check("錯誤密碼回 401",
      call("/api/auth/login", method="POST",
           body={"username": "admin", "password": "wrong"})[0] == 401)

status, data = call("/api/auth/login", method="POST",
                    body={"username": "admin", "password": "demo1234"})
check("管理員可登入", status == 200 and "token" in data)
ADMIN = data.get("token", "")

status, data = call("/api/auth/login", method="POST",
                    body={"username": "user01", "password": "demo1234"})
check("一般使用者可登入", status == 200)
USER = data.get("token", "")

check("token 帶得到本人資訊", call("/api/auth/me", ADMIN)[1].get("role") == "ADMIN")
check("偽造 token 被拒", call("/api/auth/me", "not-a-real-token")[0] == 401)

print("\n=== 3. 角色守衛（權限在 API 層，不是靠前端隱藏）===")
check("一般使用者打管理端點回 403", call("/api/admin/chunks", USER)[0] == 403)
check("管理員打管理端點回 200", call("/api/admin/chunks", ADMIN)[0] == 200)
check("一般使用者不能改模型設定",
      call("/api/admin/models", USER, "PUT", {"llm_model": "x"})[0] == 403)

print("\n=== 4. 資料端點 ===")
status, data = call("/api/stages", USER)
check("六個階段", len(data.get("stages", [])) == 6, str(len(data.get("stages", []))))
check("有索引統計", data.get("stats", {}).get("documents", 0) > 0, str(data.get("stats")))

# 時間軸只有六階段，未歸屬的文件沒有這個端點就完全沒有入口
status, data = call("/api/documents/unclassified", USER)
check("共通文件端點可用（未歸屬階段的文件才有入口）",
      status == 200 and isinstance(data.get("documents"), list))

print("\n=== 5. SSE 串流問答 ===")
sid = call("/api/chat/sessions", USER, "POST")[1]["session_id"]
result = sse("/api/chat/ask", USER, {"session_id": sid, "question": "DVT 階段散熱測試要注意什麼？"})

check("有 search 事件（Agent 確實檢索了）", "search" in result["events"])
check("有逐字的 text 事件", result["texts"] > 5, f"{result['texts']} 則")
check("最後有 done 事件", result["done"] is not None)
check("done 帶回最終答案", len(result["done"].get("answer", "")) > 50)
check("done 帶回來源", len(result["done"].get("sources", [])) > 0)
# 事件順序：done 一定是最後一則，否則前端會提早收尾
check("done 是最後一個事件", result["events"][-1] == "done", str(result["events"][-2:]))

# 追問：語意全在前文裡，必須實際再檢索一次
follow = sse("/api/chat/ask", USER, {"session_id": sid, "question": "還有嗎"})
check("追問也會實際檢索", "search" in follow["events"], str(follow["events"]))

# 無關問題必須被拒答——這是防幻覺的最終防線。
#
# **關鍵字要列得夠寬。** 這是整份測試唯一比對 LLM 自由措辭的斷言，
# 也是唯一會間歇性失敗的一項：模型說「知識庫中**沒有**相關資訊」是正確的拒答，
# 但早期的關鍵字表（查無／不足／無法／抱歉／無關）接不住它，於是偶爾誤報。
# 判準是「有沒有表達出這件事不在知識庫裡」，不是「有沒有用某個特定詞」。
REFUSAL = ("查無", "不足", "無法", "抱歉", "無關", "沒有", "未提及", "未涵蓋", "不包含", "找不到")
off = sse("/api/chat/ask", USER, {"session_id": sid, "question": "今天午餐吃什麼"})
answer = off["done"].get("answer", "")
check("無關問題被拒答", any(k in answer for k in REFUSAL), answer[:120])

print("\n=== 6. 對話紀錄與隔離 ===")
mine = call(f"/api/chat/sessions/{sid}/messages", USER)[1]
check("本人讀得到自己的訊息", len(mine.get("messages", [])) > 0)

# 對話清單就是問答頁左側那一欄的資料來源；接續舊對話全靠它回得到 session_id。
listed = call("/api/chat/sessions", USER)[1].get("sessions", [])
check("剛才的對話出現在清單裡", any(s["ID"] == sid for s in listed))
check("清單帶標題與訊息數（前端要顯示）",
      all("標題" in s and "訊息數" in s for s in listed))

# 「看全文」要能成立，歷史訊息帶回的必須是**完整切片**而不是摘要。
# 側欄只顯示兩行是 CSS 截斷，資料層不該先截——一截就永遠看不到全文了。
# 範例知識庫的切片實測落在 280–400 字。
replies = [m for m in mine["messages"] if m["role"] == "assistant"]
check("歷史訊息帶回來源", bool(replies and replies[0].get("sources")))
body = replies[0]["sources"][0]["content"] if replies and replies[0].get("sources") else ""
check("來源是完整切片、沒有被截斷",
      len(body) > 120 and not body.rstrip().endswith(("…", "...")), f"{len(body)} 字")
as_admin = call(f"/api/chat/sessions/{sid}/messages", ADMIN)[1]
check("**管理員讀不到他人的對話**", as_admin.get("messages") == [],
      f"洩漏了 {len(as_admin.get('messages', []))} 則")

print("\n=== 7. 管理端 ===")
status, data = call("/api/admin/chunks", ADMIN)
check("可列出切片", len(data.get("chunks", [])) > 0)
# 切片頁要能依文件分組，靠的是「文件 ID」。用檔名分組會把同名不同階段的檔案併錯組。
check("切片帶文件 ID（分組用）", all("文件 ID" in c for c in data["chunks"]))
# 一封 .msg 可能切出上百段；靜默截斷會讓管理員以為「就只有這些」，
# 而這一頁的用途正是確認到底存進去了什麼。
check("切片清單回報是否被截斷", "truncated" in data and data["limit"] >= 1000,
      f"limit={data.get('limit')}")
cid = data["chunks"][0]["切片 ID"]
check("可取單一切片內容", call(f"/api/admin/chunks/{cid}", ADMIN)[1].get("content"))
check("切片帶 keywords 欄位", "keywords" in call(f"/api/admin/chunks/{cid}", ADMIN)[1])

check("一般使用者不能刪切片", call(f"/api/admin/chunks/{cid}", USER, "DELETE")[0] == 403)
check("刪不存在的切片回 404", call("/api/admin/chunks/999999", ADMIN, "DELETE")[0] == 404)

# ---- 編輯切片內容 ----
# 開放編輯的前提是三個保證，缺一不可：
#   1. 存檔立刻重算向量（否則改了內容檢索完全不變）
#   2. 保留 original_content（可追溯性：來歷必須查得到）
#   3. 另存一份（否則全量重建就消失）
before = call(f"/api/admin/chunks/{cid}", ADMIN)[1]
original_len = before["char_count"]
half = before["content"][: max(30, original_len // 2)]

status, data = call(f"/api/admin/chunks/{cid}/content", ADMIN, "PUT", {"content": half})
check("可編輯切片內容", status == 200, str(data)[:80])

after = call(f"/api/admin/chunks/{cid}", ADMIN)[1]
check("編輯後字數變短", after["char_count"] < original_len,
      f"{original_len} → {after['char_count']}")
check("標記為已編輯", after.get("edited") is True)
check("**原文完整保留**", len(after.get("original_content", "")) == original_len,
      f"原文 {len(after.get('original_content', ''))} 字 / 應為 {original_len}")

check("空內容被拒絕",
      call(f"/api/admin/chunks/{cid}/content", ADMIN, "PUT", {"content": "   "})[0] == 400)
check("一般使用者不能編輯",
      call(f"/api/admin/chunks/{cid}/content", USER, "PUT", {"content": "x"})[0] == 403)

status, data = call(f"/api/admin/chunks/{cid}/revert", ADMIN, "POST")
check("可還原成原文", status == 200, str(data)[:80])
reverted = call(f"/api/admin/chunks/{cid}", ADMIN)[1]
check("還原後字數回到原值", reverted["char_count"] == original_len,
      f"{reverted['char_count']} vs {original_len}")
check("還原後不再標記已編輯", reverted.get("edited") is False)
check("沒編輯過的切片不能還原",
      call(f"/api/admin/chunks/{cid}/revert", ADMIN, "POST")[0] == 400)

status, data = call("/api/admin/search-test", ADMIN, "POST", {"query": "散熱測試"})
check("檢索測試回傳距離", status == 200 and "distance" in (data.get("hits") or [{}])[0])

status, data = call("/api/admin/models", ADMIN)
check("模型設定可讀取", status == 200 and data.get("current", {}).get("embed_model"))
# V1 曾因為選單找不到目前的值就自動改成第一個，把 embed_model 換成生成模型，
# 之後每次檢索都 501。這裡確認「不送值就不會被改掉」。
before = data["current"]["embed_model"]
call("/api/admin/models", ADMIN, "PUT", {})
after = call("/api/admin/models", ADMIN)[1]["current"]["embed_model"]
check("空的更新請求不會動到既有設定", before == after, f"{before} -> {after}")

status, data = call("/api/admin/errors", ADMIN)
# 「缺視覺模型」與「解析失敗」必須分開，混在一起會讓人以為檔案壞了
check("解析狀態分為 needs_vlm 與 failures 兩類",
      "needs_vlm" in data and "failures" in data)

print("\n=== 8. 文件清單與刪除 ===")
status, data = call("/api/admin/documents", ADMIN)
docs = data.get("documents", [])
check("可列出知識庫文件", status == 200 and len(docs) > 0, f"{len(docs)} 份")
# 清單以磁碟為準——上傳完還沒建索引的檔案也要看得見，
# 那正是「檔案到底存進去沒有」最需要確認的時刻。
check("清單帶索引狀態與切片數",
      all("indexed" in d and "chunk_count" in d and "rel_path" in d for d in docs))
check("已索引的文件對得上切片數",
      any(d["indexed"] and d["chunk_count"] > 0 for d in docs),
      "沒有任何一份文件是已索引狀態——file_path 可能沒對上")

check("一般使用者不能列出文件", call("/api/admin/documents", USER)[0] == 403)
check("一般使用者不能刪除文件",
      call("/api/admin/documents?path=x.md", USER, "DELETE")[0] == 403)

# **路徑遍歷防護。** rel_path 來自前端，不可信；
# 沒有這道防護，`../../` 就能刪掉專案裡的任何檔案。
for evil in ("../../../SDD_v2.md", "..%2F..%2Fknowledge.db", "/etc/passwd"):
    code = call(f"/api/admin/documents?path={evil}", ADMIN, "DELETE")[0]
    check(f"擋下路徑遍歷 {evil[:22]}", code == 400, f"回了 {code}")
check("SDD_v2.md 沒有被刪掉", call("/api/admin/documents", ADMIN)[0] == 200)

check("刪除不存在的檔案回 400",
      call("/api/admin/documents?path=nope.md", ADMIN, "DELETE")[0] == 400)

# 支援格式從後端送出，前端不寫死——否則改了副檔名清單不會同步
status, data = call("/api/admin/index-status", ADMIN)
check("index-status 帶支援格式清單",
      len(data.get("supported", [])) > 10 and len(data.get("image_types", [])) > 3,
      f"文件 {len(data.get('supported', []))} 種 / 圖片 {len(data.get('image_types', []))} 種")

print("\n=== 9. 上傳 ===")


def upload(fname: str, content: bytes, stage: str = "Concept"):
    """模擬瀏覽器的 multipart 上傳。

    **這一段非測不可。** `save_uploads()` 是從 V1 原封複製的，
    而它是照 Streamlit 的 UploadedFile 寫的（`.name` / `.getbuffer()`）；
    FastAPI 的 UploadFile 是 `.filename` / `.file`。
    介面對不上時每次上傳都 500，而前端若沒有 try/catch 會整個吞掉，
    使用者看到的只是「按了沒反應」——這種 bug 光靠讀程式碼不會發現。
    """
    bd = "----t" + uuid.uuid4().hex
    body = (
        f'--{bd}\r\nContent-Disposition: form-data; name="files"; filename="{fname}"\r\n\r\n'
    ).encode() + content + f"\r\n--{bd}--\r\n".encode()
    req = urllib.request.Request(
        f"{BASE}/api/admin/upload?stage_code={stage}", data=body, method="POST")
    req.add_header("Authorization", f"Bearer {ADMIN}")
    req.add_header("Content-Type", f"multipart/form-data; boundary={bd}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {"_err": exc.read().decode()[:120]}


probe = f"上傳測試_{uuid.uuid4().hex[:6]}.md"
status, data = upload(probe, b"# probe\n\nhello world")
check("上傳成功回 200", status == 200, str(data)[:90])
check("回傳存入的相對路徑", data.get("saved") == [f"Concept/{probe}"], str(data.get("saved")))

check("不支援的格式被擋下並說明原因",
      "不支援的格式" in " ".join(upload("壞檔.exe", b"MZ")[1].get("errors", [])))
check("空檔被擋下",
      "空的" in " ".join(upload("空檔.md", b"")[1].get("errors", [])))

# 同名不覆蓋，加序號——覆蓋會讓管理員無聲失去舊檔
dup = upload(probe, b"# probe2")[1].get("saved", [""])[0]
check("同名檔案加序號不覆蓋", dup.endswith("_2.md"), dup)

# 上傳完就要出現在清單裡（即使還沒建索引）
docs = call("/api/admin/documents", ADMIN)[1].get("documents", [])
names = {d["file_name"] for d in docs}
check("剛上傳的檔案立刻出現在清單", probe in names)

# 收尾：把測試檔刪掉，順便再驗一次刪除
for rel in (f"Concept/{probe}", dup):
    call(f"/api/admin/documents?path={urllib.parse.quote(rel)}", ADMIN, "DELETE")
docs = call("/api/admin/documents", ADMIN)[1].get("documents", [])
check("測試檔已清除", probe not in {d["file_name"] for d in docs})


# ── 文件開啟與下載 ──────────────────────────────────────────────
print("\n§ 文件開啟與下載")


def head(path: str, token: str = "") -> tuple[int, dict]:
    """取回應標頭。下載端點回的是位元組，用 call() 會被當成 JSON 解析失敗。"""
    req = urllib.request.Request(BASE + path)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        return exc.code, {}
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": str(exc)}


_docs_seen = []
for _code in ("Concept", "Plan", "EVT", "DVT", "PVT", "MP"):
    _docs_seen += call(f"/api/stages/{_code}/documents", ADMIN)[1].get("documents", [])
DOC_PATH = _docs_seen[0]["路徑"] if _docs_seen else ""
DOC_Q = urllib.parse.quote(DOC_PATH)
check("取得一份已索引文件的路徑", bool(DOC_PATH), DOC_PATH[:60])

# 整份文件內容（「開啟完整文件」走這條，不下載）
code, body = call(f"/api/documents/content?path={DOC_Q}", ADMIN)
check("完整文件內容可讀取", code == 200 and body.get("char_count", 0) > 0, f"{code} / {body.get('char_count')}")

# **完整文件必須涵蓋所有切片。**
# 這條守的是一個實際發生過的 bug：閱讀頁當場重新解析，而解析結果取決於
# 當下的 VLM 設定。某份以「強制視覺解析」建索引的 PDF，切片裡有 VLM 產生的
# 圖片描述，閱讀頁卻只剩文字層——使用者看到的跟 AI 讀到的對不起來。
# 現在閱讀頁改讀索引時存下的 documents.content，兩邊保證同源。
def _body_of(chunk_content: str) -> str:
    """去掉切片開頭的麵包屑行（`> 出處`）——那是切片器加的，原文裡沒有。"""
    lines = chunk_content.strip().split("\n")
    while lines and (lines[0].startswith(">") or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def _squash(text: str) -> str:
    """去掉所有空白再比對。

    切片器在合併段落時會把換行正規化（原文的兩行在切片裡變成同一行），
    逐字比對會因此誤判成「切片內容不在原文裡」——實測 1262 個切片中
    有 248 個這樣誤報，內容其實一個字都沒少。
    **這項測試要守的是「內容有沒有同源」，不是「空白排列一不一樣」。**
    """
    return re.sub(r"\s+", "", text)


_chunks = call("/api/admin/chunks", ADMIN)[1].get("chunks", [])
_orphan = []
for _doc in _docs_seen:
    _name = _doc.get("文件名稱")
    _c, _b = call(f"/api/documents/content?path={urllib.parse.quote(_doc['路徑'])}", ADMIN)
    _text = _squash(_b.get("content", ""))
    for _ch in [c for c in _chunks if c.get("文件") == _name]:
        if _squash(_body_of(_ch.get("內容", "")))[:60] not in _text:
            _orphan.append(_name)
check(
    "每個切片都能在完整文件中找到",
    not _orphan,
    f"{len(_orphan)} 段找不到（閱讀頁與索引不同源）：{set(_orphan)}",
)

# 下載：強制存檔
code, hdrs = head(f"/api/documents/download?path={DOC_Q}", ADMIN)
check("下載回 octet-stream", code == 200 and hdrs.get("content-type") == "application/octet-stream", hdrs.get("content-type", ""))
check("下載帶 attachment", "attachment" in hdrs.get("content-disposition", ""), hdrs.get("content-disposition", ""))

# inline 白名單。**手打網址就能繞過前端，所以把關必須在後端。**
# .md 不在白名單：blob URL 下瀏覽器會用系統預設編碼解碼，中文變亂碼（實測踩過）。
# .svg/.html 更嚴重：blob URL 繼承本頁來源，其中的 <script> 會在應用 origin 下
# 執行並讀走 sessionStorage 裡的 JWT。這幾種一律退回 attachment。
code, hdrs = head(f"/api/documents/download?path={DOC_Q}&inline=true", ADMIN)
_is_md = DOC_PATH.lower().endswith((".md", ".txt"))
if _is_md:
    check("非白名單格式的 inline 退回 attachment",
          code == 200 and hdrs.get("content-disposition", "").startswith("attachment"),
          hdrs.get("content-disposition", ""))
    check("非白名單格式的 inline 退回 octet-stream",
          hdrs.get("content-type") == "application/octet-stream", hdrs.get("content-type", ""))
else:
    check("inline 回真實 MIME", code == 200 and "octet-stream" not in hdrs.get("content-type", ""), hdrs.get("content-type", ""))
    check("inline 帶 inline disposition", hdrs.get("content-disposition", "").startswith("inline"), hdrs.get("content-disposition", ""))

# 白名單本身（不依賴知識庫裡剛好有哪些格式）
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from services import stage_service as _ss  # noqa: E402

check("PDF 可 inline", _ss.can_inline("a.pdf"))
check("PNG 可 inline", _ss.can_inline("a.PNG"))
check("SVG 不可 inline（腳本會在本頁 origin 執行）", not _ss.can_inline("a.svg"))
check("HTML 不可 inline（同上）", not _ss.can_inline("a.html"))
check("Markdown 不可 inline（編碼會壞）", not _ss.can_inline("a.md"))
check("pptx 不可 inline（瀏覽器無法渲染）", not _ss.can_inline("a.pptx"))

# 路徑白名單。這幾條是安全防護，不是防呆：
# 少了它，任何登入者都能把 path 換成系統任意檔案整份讀走（含 .env）。
for _bad in (r"C:\Windows\win.ini", ".env", "../../.env", r"C:\Users\Public\desktop.ini"):
    _q = urllib.parse.quote(_bad)
    check(f"擋下讀取 {_bad}", call(f"/api/documents/content?path={_q}", ADMIN)[0] == 404)
    check(f"擋下下載 {_bad}", head(f"/api/documents/download?path={_q}", ADMIN)[0] == 404)

check("未登入不能下載", head(f"/api/documents/download?path={DOC_Q}")[0] == 401)
check("未登入不能讀內容", call(f"/api/documents/content?path={DOC_Q}")[0] == 401)

# 本機開啟。**不測真的開檔**——那會在每次跑測試時彈出視窗。
# 這裡守的是把關順序：白名單先擋下，才輪到 os.startfile。
check("is-local 回布林", isinstance(call("/api/client/is-local", ADMIN)[1].get("is_local"), bool))
check(
    "本機開啟擋下非索引檔案",
    call("/api/documents/open", ADMIN, "POST", {"path": r"C:\Windows\win.ini"})[0] == 400,
)
check(
    "本機開啟需登入",
    call("/api/documents/open", "", "POST", {"path": DOC_PATH})[0] == 401,
)


# ── 切片品質 ────────────────────────────────────────────────────
print("\n§ 切片品質")

from services.ingest_service import chunk_text, normalize_wide_tables  # noqa: E402

# 長文欄位的表格 → 逐筆記錄。這是一份實際上傳的問答紀錄 Excel 的縮影。
WIDE = (
    "## 問答紀錄\n"
    "| 輪次 | 編號 | 問題 | 完整答案 |\n"
    "| --- | --- | --- | --- |\n"
    "| 第1輪 | 1 | 鹽霧測試條件? | 回答：\\n\\n\\*\\*步驟 1\\*\\*：" + "鹽溶液濃度為百分之五。" * 40 + " |\n"
    "| 第1輪 | 2 | 振動測試流程? | 回答：\\n\\n" + "先做共振搜尋再做耐久試驗。" * 40 + " |\n"
)
wide_md = normalize_wide_tables(WIDE)
check("長文表格被改寫成逐筆記錄", wide_md.count("### ") == 2, wide_md[:80])
check("字面 \\n 已還原成真換行", "\\n" not in wide_md)
check("轉義星號已還原", "\\*" not in wide_md and "**步驟 1**" in wide_md)

# 欄位都很短的表格不該被動——那種表格本來就該保持表格樣子
NARROW = (
    "## 窗口一覽\n"
    "| 單位 | 窗口 | 分機 |\n"
    "| --- | --- | --- |\n"
    "| 研發 | 王小明 | 1234 |\n"
    "| 品保 | 李小華 | 5678 |\n"
)
check("短欄位表格維持原樣", normalize_wide_tables(NARROW) == NARROW)

wide_chunks = chunk_text(wide_md, 500, 80, "qa.xlsx")
check("長記錄被切成多段", len(wide_chunks) >= 4, str(len(wide_chunks)))
check("沒有切片超過上限", all(len(c["content"]) <= 520 for c in wide_chunks),
      str(max(len(c["content"]) for c in wide_chunks)))
# **這是這次修正的核心。** 改版前 127 個切片裡只有 1 個帶得到欄位表頭，
# 63% 看不出自己在回答什麼問題；續段被檢索到時等於一團沒有主語的文字。
check(
    "每個切片都帶得到自己的題目",
    all("?" in c["content"] for c in wide_chunks),
    f"{sum(1 for c in wide_chunks if '?' not in c['content'])} 段沒有",
)
check("locator 帶出處", all("›" in c["locator"] for c in wide_chunks))

# 原本就切得好的文件不該被影響
PLAIN = "# 作業規範\n\n## 目標\n驗證設計符合規格。\n\n## 產出物\n1. 測試計畫\n2. 測試報告\n"
check("短文件仍是一段", len(chunk_text(PLAIN, 500, 80, "a.md")) == 1)

# 沒有實質內容的區塊不該進索引，否則會產生語意近乎隨機的向量
EMPTY_SECTION = "<!-- Slide number: 1 -->\n### Notes:\n\n<!-- Slide number: 2 -->\n" + "實際內容。" * 30
check(
    "空區塊不進索引",
    all(len(c["content"].strip()) > 40 for c in chunk_text(EMPTY_SECTION, 500, 80, "b.pptx")),
)

print("\n" + "=" * 56)
print(f"  通過 {_passed} 項，失敗 {_failed} 項")
print("=" * 56)
sys.exit(1 if _failed else 0)
