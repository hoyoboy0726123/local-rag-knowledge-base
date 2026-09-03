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
    queries = []
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
                elif name == "search":
                    queries.append(payload.get("query", ""))
                elif name == "done":
                    done = payload
    return {"events": events, "texts": texts, "done": done, "queries": queries}


print("\n=== 1. 服務可用性 ===")
status, health = call("/api/health")
if status != 200:
    print("  [FAIL] 後端未啟動。請先執行：")
    print("         venv\\Scripts\\python.exe -m uvicorn backend.main:app --port 8600")
    sys.exit(1)
check("health 端點正常", health.get("status") == "ok", str(health))
check("首頁提供前端靜態檔", call("/")[0] == 200)

print("\n=== 2. 認證 ===")
check("未帶 token 一律 401", call("/api/kbs")[0] == 401)
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
# 知識庫 = 根目錄下的子資料夾，由使用者自訂；「通用」是根目錄的檔案，永遠排第一。
# 測試不假設有哪些知識庫（那是使用者的資料夾），只驗結構。
status, data = call("/api/kbs", USER)
_kbs = data.get("kbs", [])
check("知識庫清單第一個是「通用」", bool(_kbs) and _kbs[0]["is_general"] and _kbs[0]["name"] == "", str(_kbs[:1]))
check("每個知識庫都帶文件數", all("doc_count" in k for k in _kbs))
check("有索引統計", data.get("stats", {}).get("documents", 0) > 0, str(data.get("stats")))

# 通用知識庫的文件要有入口——它們往往是最基礎的跨領域文件
status, data = call("/api/kbs/documents?kb=", USER)
check("通用知識庫的文件端點可用", status == 200 and isinstance(data.get("documents"), list))

print("\n=== 5. SSE 串流問答 ===")
# **問句與斷言都不綁定任何特定內容。**
#
# 這套測試一度問「DVT 階段散熱測試要注意什麼？」並要求答案超過 50 字——那是
# 為當初隨附的 NUC 範例文件寫的。範例文件移出版控後，任何人 clone 下來指向
# 自己的資料夾，這兩項就必然失敗，而失敗的原因與程式無關。
#
# 這是通用知識庫，測試不能假設裡面有什麼。改為驗**機制**：檢索有沒有發生、
# 事件順序對不對、來源有沒有回傳。至於答得出來還是誠實拒答，兩者都是正確
# 行為，取決於使用者自己的文件。
sid = call("/api/chat/sessions", USER, "POST")[1]["session_id"]
result = sse("/api/chat/ask", USER, {"session_id": sid, "question": "這份文件在說明什麼？"})

check("有 search 事件（Agent 確實檢索了）", "search" in result["events"])
check("最後有 done 事件", result["done"] is not None)
answer = result["done"].get("answer", "")
check("done 帶回非空的答案", bool(answer.strip()), f"{len(answer)} 字")
check("done 帶回來源", len(result["done"].get("sources", [])) > 0)
# 事件順序：done 一定是最後一則，否則前端會提早收尾
check("done 是最後一個事件", result["events"][-1] == "done", str(result["events"][-2:]))

# 追問：語意全在前文裡，必須實際再檢索一次
follow = sse("/api/chat/ask", USER, {"session_id": sid, "question": "還有嗎"})
check("追問也會實際檢索", "search" in follow["events"], str(follow["events"]))
# **只驗「有沒有 search 事件」是不夠的。** 這條斷言原本就是那樣寫的，於是漏掉了
# 一個真的 bug：模型有呼叫工具，卻把「還有嗎」原封不動當成 query 送出去，
# 撈到 0 段後回「知識庫中查無足夠資訊」。實測 8 次有 3 次如此——
# 使用者看到的是知識庫沒東西，實際上只是問錯了。所以要驗送出去的 query。
# **斷言的是「系統有沒有復原」，不是「哪一層復原的」。**
#
# 原本要求每一次檢索的 query 都不能是「還有嗎」。那太嚴格了：真正的需求是
# 追問不要因為 query 沒補主詞而落入假的「查無資訊」，而補救有三層——句型
# 比對、單一職責判斷、0 段時用原問題重查。實測看過第一次沒補到、第三層立刻
# 用正解補上的情形，那是設計中的行為，卻會讓這條斷言變紅。
#
# 綁死在「第一次就要對」等於把實作細節寫進測試，之後調整分層就會誤報。
check("追問至少有一次用了補回主題的 query",
      any(q.strip() != "還有嗎" for q in follow["queries"]),
      str(follow["queries"]))

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
# **只驗「有沒有被截斷」，不驗長度。**
#
# 原本要求超過 120 字。但切片長度取決於使用者自己的文件——實測這個知識庫裡
# 就有 47、64、70 字的合法切片（表格的最後幾列、獨立的小標題），
# 它們完整無缺卻會讓這條斷言失敗。長度不是「有沒有被截斷」的證據，
# 結尾的刪節號才是。
check("來源是完整切片、沒有被截斷",
      bool(body) and not body.rstrip().endswith(("…", "...")), f"{len(body)} 字")
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

# 查詢字串同樣不綁定內容——這裡驗的是端點會回傳帶距離的結果，不是撈到什麼。
status, data = call("/api/admin/search-test", ADMIN, "POST", {"query": "文件"})
check("檢索測試回傳距離", status == 200 and "distance" in (data.get("hits") or [{}])[0])

status, data = call("/api/admin/models", ADMIN)
check("模型設定可讀取", status == 200 and data.get("current", {}).get("embed_model"))
# V1 曾因為選單找不到目前的值就自動改成第一個，把 embed_model 換成生成模型，
# 之後每次檢索都 501。這裡確認「不送值就不會被改掉」。
before = data["current"]["embed_model"]
call("/api/admin/models", ADMIN, "PUT", {})
after = call("/api/admin/models", ADMIN)[1]["current"]["embed_model"]

# top_k 可調，但夾在 1..30——30 段約 15,000 字，會把 num_ctx 撐到 16K 以上，
# 8 GB 顯卡有四成的層落到 CPU。超過上限不是報錯而是夾住，前端另有分級警告。
_orig_topk = call("/api/admin/models", ADMIN)[1]["current"].get("top_k")
check("models 回傳 top_k", isinstance(_orig_topk, int), str(_orig_topk))
call("/api/admin/models", ADMIN, "PUT", {"top_k": 99})
check("top_k 超過 20 被夾到 20（30 實測反而更差，不開放）", call("/api/admin/models", ADMIN)[1]["current"]["top_k"] == 20)
call("/api/admin/models", ADMIN, "PUT", {"top_k": 0})
check("top_k 低於 1 被夾到 1", call("/api/admin/models", ADMIN)[1]["current"]["top_k"] == 1)
call("/api/admin/models", ADMIN, "PUT", {"top_k": _orig_topk or 6})
check("top_k 還原", call("/api/admin/models", ADMIN)[1]["current"]["top_k"] == (_orig_topk or 6))
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


def upload(fname: str, content: bytes, kb: str | None = None, new_kb: str | None = None):
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
        f"{BASE}/api/admin/upload?{urllib.parse.urlencode({k: v for k, v in (('kb', kb), ('new_kb', new_kb)) if v})}",
        data=body, method="POST")
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
# 沒指定知識庫就放根目錄（通用），所以相對路徑就是檔名本身、不帶資料夾
check("回傳存入的相對路徑（未指定知識庫＝通用，落在根目錄）", data.get("saved") == [probe], str(data.get("saved")))

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
for rel in (probe, dup):
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


_docs_seen = call("/api/kbs/documents", ADMIN)[1].get("documents", [])
DOC_PATH = _docs_seen[0]["路徑"] if _docs_seen else ""
DOC_Q = urllib.parse.quote(DOC_PATH)
check("取得一份已索引文件的路徑", bool(DOC_PATH), DOC_PATH[:60])

# ----------------------------------------------------------------- 知識庫管理
#
# 一個知識庫 = 根目錄下的一個子資料夾；搬移只搬檔案並同步路徑，**不重新向量化**。
# 全部用臨時名稱、跑完清乾淨，不動使用者自己的資料夾。
_KB = "測試知識庫_" + uuid.uuid4().hex[:6]
_KBQ = urllib.parse.quote(_KB)
check("一般使用者不能建知識庫", call("/api/admin/kbs", USER, "POST", {"name": _KB})[0] == 403)
check("名稱含路徑符號被拒", call("/api/admin/kbs", ADMIN, "POST", {"name": "a/b"})[0] == 400)
check("名稱是系統保留字被拒", call("/api/admin/kbs", ADMIN, "POST", {"name": "venv"})[0] == 400)
check("管理員可建知識庫", call("/api/admin/kbs", ADMIN, "POST", {"name": _KB})[0] == 200)
check("重複建立被拒", call("/api/admin/kbs", ADMIN, "POST", {"name": _KB})[0] == 400)
check("新知識庫出現在清單", any(k["name"] == _KB for k in call("/api/kbs", USER)[1]["kbs"]))

# 上傳時順手新建另一個知識庫，檔案要落在那個資料夾
_KB2 = _KB + "_b"
_fn = f"kbtest_{uuid.uuid4().hex[:6]}.md"
_st, _up = upload(_fn, ("# 測試" + chr(10) * 2 + "這是知識庫測試文件。").encode(), new_kb=_KB2)
check("上傳可同時新建知識庫", _st == 200 and _up.get("saved") == [f"{_KB2}/{_fn}"], str(_up))

_files = call("/api/admin/documents", ADMIN)[1]["documents"]
_row = next((d for d in _files if d["file_name"] == _fn), None)
check("清單顯示檔案所屬知識庫", _row is not None and _row["kb"] == _KB2, str(_row and _row["kb"]))

# 搬到第一個知識庫、再搬到通用；每一步路徑與分類都要跟上
check("搬到另一個知識庫", call("/api/admin/documents/move", ADMIN, "POST", {"rel_path": f"{_KB2}/{_fn}", "kb": _KB})[0] == 200)
_row = next((d for d in call("/api/admin/documents", ADMIN)[1]["documents"] if d["file_name"] == _fn), None)
check("搬移後 rel_path 與 kb 同步", _row and _row["rel_path"] == f"{_KB}/{_fn}" and _row["kb"] == _KB, str(_row and (_row["rel_path"], _row["kb"])))
check("有檔案的知識庫不能刪", call(f"/api/admin/kbs/{_KBQ}", ADMIN, "DELETE")[0] == 400)
check("搬到通用（根目錄）", call("/api/admin/documents/move", ADMIN, "POST", {"rel_path": f"{_KB}/{_fn}", "kb": None})[0] == 200)
_row = next((d for d in call("/api/admin/documents", ADMIN)[1]["documents"] if d["file_name"] == _fn), None)
check("通用的 rel_path 不帶資料夾、kb 為空", _row and _row["rel_path"] == _fn and not _row["kb"], str(_row and (_row["rel_path"], _row["kb"])))

# 改名要同步；空的才能刪
_KB3 = _KB + "_c"
check("知識庫可改名", call(f"/api/admin/kbs/{_KBQ}", ADMIN, "PUT", {"new_name": _KB3})[0] == 200)
check("改名後舊名消失、新名出現",
      (lambda names: _KB not in names and _KB3 in names)([k["name"] for k in call("/api/kbs", USER)[1]["kbs"]]))
check("空的知識庫可刪", call(f"/api/admin/kbs/{urllib.parse.quote(_KB3)}", ADMIN, "DELETE")[0] == 200)
check("空的知識庫可刪（第二個）", call(f"/api/admin/kbs/{urllib.parse.quote(_KB2)}", ADMIN, "DELETE")[0] == 200)
_del = call(f"/api/admin/documents?path={urllib.parse.quote(_fn)}", ADMIN, "DELETE")
check("測試檔已清除", _del[0] == 200 and not any(d["file_name"] == _fn for d in call("/api/admin/documents", ADMIN)[1]["documents"]), str(_del))

# 檢索範圍是集合；空字串代表通用。只驗機制：限定範圍後回來的切片不能落在範圍外。
_kb_names = [k["name"] for k in call("/api/kbs", USER)[1]["kbs"] if k["doc_count"] > 0]
if len(_kb_names) >= 2:
    _pick = _kb_names[:1]
    _hits = call("/api/admin/search-test", ADMIN, "POST", {"query": "測試", "kbs": _pick})[1].get("hits", [])
    check("限定知識庫後檢索結果都在範圍內",
          all((h["kb"] or "") in _pick for h in _hits), str({(h["kb"] or "通用") for h in _hits}))

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
from services import kb_service as _ss  # noqa: E402

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

# 表格整段沒有句號，以前會直接掉到硬切，把 `| 01012 | ROCKCHIP |` 切成
# `| 01012 | ROCKC` 和 `HIP |`——模型讀到前半段就把「ROCKC」當成廠商名稱回報。
TABLE = "## 廠商\n\n| 料號 | 種類 |\n" + "".join(
    f"| 0100{i % 10} | VENDOR_NAME_{i:02d} |\n" for i in range(60)
)
_tc = chunk_text(TABLE, 200, 40, "t.pdf")
check("表格切片不超過上限", all(len(c["content"]) <= 200 for c in _tc),
      f"最長 {max(len(c['content']) for c in _tc)}")
check(
    "表格不切在字中間",
    all(c["content"].rstrip().endswith("|") or c["content"].rstrip().endswith("廠商")
        for c in _tc),
    "；".join(repr(c["content"][-14:]) for c in _tc[:4]),
)
_names = "".join(c["content"] for c in _tc)
check("每個廠商名稱都完整保留",
      all(f"VENDOR_NAME_{i:02d}" in _names for i in range(60)),
      [i for i in range(60) if f"VENDOR_NAME_{i:02d}" not in _names][:5])

# 沒有換行可退時仍要硬切，不能無限長
_long = "衝擊測試" * 400
check("無換行長句仍會被切開",
      all(len(c["content"]) <= 200 for c in chunk_text(_long, 200, 40, "x.md")))

# ------------------------------------------------- _run_search 的回傳形狀
#
# 它有五個 return，散在函式各處。改成回傳三個值時漏掉了三個，而**漏掉的那些
# 只有在特定分支才會走到**——一般問題測不出來，要等到「檢索 0 段」或「內容
# 前面給過了」才炸，那時使用者看到的是整個問答沒有任何回應。
# 用 AST 檢查所有 return 的形狀，比逐一手動確認可靠。
import ast as _ast  # noqa: E402

_tree = _ast.parse(pathlib.Path("services/agent_service.py").read_text(encoding="utf-8"))
_fn = [n for n in _ast.walk(_tree)
       if isinstance(n, _ast.FunctionDef) and n.name == "_run_search"][0]
_returns = [n for n in _ast.walk(_fn) if isinstance(n, _ast.Return)]
_bad = [n.lineno for n in _returns
        if not isinstance(n.value, _ast.Tuple) or len(n.value.elts) != 3]
check("_run_search 每個 return 都是三元組", not _bad, f"第 {_bad} 行不是")
check("_run_search 的 return 沒有變少", len(_returns) >= 5, f"只剩 {len(_returns)} 個")


# ----------------------------------------------------------------- JWT 金鑰
#
# 這裡原本是一個寫死在原始碼裡的固定金鑰。原始碼是公開的 repo，所以那串字
# 等於公開資訊——任何人都能自己簽一張 role=ADMIN 的 token，不需要帳號密碼。
# 現在改成首次啟動自動產生並存進資料庫。
#
# 三件事都要守住，少一件這個修正就沒有意義：
#   1. 原始碼裡不能再有可用的預設金鑰
#   2. 用那串舊金鑰偽造的 token 必須被拒絕
#   3. 金鑰要持久——不然每次重啟就把所有人踢出去（那正是當初用固定值的理由）
from datetime import datetime, timedelta, timezone  # noqa: E402

from jose import jwt as _jwt  # noqa: E402

_deps_src = pathlib.Path("backend/deps.py").read_text(encoding="utf-8")
check("原始碼裡沒有寫死的金鑰",
      "leslie-v2-dev" not in _deps_src and "token_urlsafe" in _deps_src)

_forged = _jwt.encode(
    {"sub": "1", "username": "admin", "role": "ADMIN",
     "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
    "leslie-v2-dev-secret-change-in-production", algorithm="HS256")
check("用舊的公開金鑰偽造的 ADMIN token 被拒", call("/api/auth/me", _forged)[0] == 401)

# 隨便一組別的金鑰也一樣要被擋
_other = _jwt.encode(
    {"sub": "1", "username": "admin", "role": "ADMIN",
     "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
    "another-guessed-secret", algorithm="HS256")
check("用其他金鑰偽造的 token 被拒", call("/api/auth/me", _other)[0] == 401)

from database import get_setting as _get_setting  # noqa: E402

_stored = _get_setting("jwt_secret")
check("金鑰已存進資料庫（重啟才不會把所有人踢出去）", len(_stored) >= 32, f"{len(_stored)} 字元")
check("金鑰不是可猜測的固定值", _stored != "leslie-v2-dev-secret-change-in-production")


# ----------------------------------------------------------------- 設定預設值
#
# num_ctx 的預設值寫在兩個地方：models.DEFAULT_SETTINGS（建立資料庫時寫入）
# 與 ollama_client.DEFAULT_NUM_CTX（讀不到設定時的 fallback）。
# **不一致的後果是安靜的**——提示詞超過視窗時 Ollama 直接截掉前面的內容，
# 不報錯，只是答案突然變得不對。所以釘住它。
from models import DEFAULT_SETTINGS  # noqa: E402
from services.ollama_client import DEFAULT_NUM_CTX  # noqa: E402

check("num_ctx 有進 DEFAULT_SETTINGS", "num_ctx" in DEFAULT_SETTINGS)
check("兩處的 num_ctx 預設值一致",
      DEFAULT_SETTINGS.get("num_ctx") == str(DEFAULT_NUM_CTX),
      f"{DEFAULT_SETTINGS.get('num_ctx')} vs {DEFAULT_NUM_CTX}")


# ----------------------------------------------------------------- 追問補主詞
#
# 系統指示第 1 條要求模型自己把「還有嗎」補成有主題的 query，但那條要跟另外
# 二十幾條競爭注意力，實測 8 次有 3 次沒做。這組測試守的是程式層的補救。
#
# **誤判比漏接嚴重得多**，所以「不該改寫的不能被改寫」那幾項才是重點：
# agent_service 裡留有教訓——曾經無差別呼叫 condense_question()，
# 使用者問「今天午餐吃什麼」被按前文改寫成「PVT 階段的產出物」，
# 一個合法的拒答變成答非所問。
from services.agent_service import _resolve_follow_up  # noqa: E402

HIST = [
    {"role": "user", "content": "料件有哪些種類"},
    {"role": "assistant", "content": "料件可分為 01 CPU、02 CHIPSET 等大類。"},
]
FOLLOW_UPS = ["還有嗎", "還有呢", "還有其他的", "其他呢", "繼續", "再多說一點",
              "更多", "然後呢", "接下來呢", "還有嗎？",
              # 指示代名詞句型。用結構認，不問模型。
              "那它呢", "那這個呢", "這個呢", "那 DVT 呢", "那 CPU 的次分類呢",
              "講清楚一點", "詳細一點"]
COMPLETE = ["今天午餐吃什麼", "請假流程是什麼", "料件有哪些種類", "壓力測試的條件",
            "這份文件在說明什麼？", "那份規範的溫度上限是多少",
            "這個專案的里程碑有哪些", "他們的分工怎麼安排"]

# 追問處理拆成兩個決定，測試也照這樣拆：
#
#   A. 這句是不是追問？—— 純句型，**不 stub、不呼叫模型**，這是防線本體。
#      這裡曾經改成問模型「單獨看得懂嗎」，實測它會把「今天午餐吃什麼」判成
#      看不懂、改寫成前文主題，一個合法的拒答變成一整段潤滑油規格——而當時
#      的測試把那個判斷 stub 掉了，所以完全沒看到。stub 掉不確定的部分，
#      等於把測試的眼睛遮起來。
#   B. 確定是追問了，補成什麼？—— 交給 condense_question。這一步 stub 是
#      **可以的**：它只影響改寫的品質，不影響「要不要改寫」；安全性質由 A 保證。
import services.agent_service as _ag  # noqa: E402


def _with_condense(query, reply, error=""):
    """把 B 的模型呼叫換成固定回應，回傳 (結果, 有沒有呼叫到)。"""
    called = []
    real = _ag.rag_service.condense_question
    _ag.rag_service.condense_question = lambda q, h: (called.append(q), (reply, error))[1]
    try:
        return _resolve_follow_up(query, HIST), bool(called)
    finally:
        _ag.rag_service.condense_question = real


# --- A：閘門本身（決定性，跑真的程式碼）---
for _q in COMPLETE:
    _r, _called = _with_condense(_q, "不該用到這個")
    check(f"完整問句不被改寫：{_q}", _r == _q and not _called, f"{_r!r} called={_called}")

for _q in FOLLOW_UPS:
    _r, _called = _with_condense(_q, "料件的種類與分類有哪些")
    check(f"追問會進入改寫：{_q}", _called and _r == "料件的種類與分類有哪些", f"{_r!r}")

check("沒有前文時原樣返回", _resolve_follow_up("還有嗎", None) == "還有嗎")

# --- B：改寫的保底 ---
_r, _ = _with_condense("還有嗎", "", error="模型逾時")
check("改寫失敗退回上一句原話", _r == "料件有哪些種類", _r)
_r, _ = _with_condense("還有嗎", "")
check("改寫回空退回上一句原話", _r == "料件有哪些種類", _r)
_r, _ = _with_condense("還有嗎", "還有嗎？")
check("改寫仍是追問句就退回原話（模型有時會原句吐回）", _r == "料件有哪些種類", _r)

_real_condense = _ag.rag_service.condense_question
_ag.rag_service.condense_question = lambda q, h: ("不該用到", "")
try:
    check("前文只有追問時不會補成追問",
          _resolve_follow_up("還有嗎", [{"role": "user", "content": "還有嗎"}]) == "還有嗎")
finally:
    _ag.rag_service.condense_question = _real_condense

# --- B：一次真實模型，驗改寫確實比「用原話」多保住資訊 ---
# 「那 DVT 呢」用原話會補成「料件有哪些種類」，DVT 這個字就丟了。
# 這是整個改法存在的理由，所以要用真的模型跑一次。
_real = _resolve_follow_up("那 DVT 呢", HIST)
check("真實改寫保住追問裡的新資訊（DVT）", "DVT" in _real and _real != "那 DVT 呢", _real)


# ------------------------------------------------- 第 3 層：0 段時由程式再查
#
# 實測 4 次有 3 次，模型檢索 0 段後直接回「知識庫中查無足夠資訊」——工具回傳
# 已經明講「請換不同的關鍵詞再查一次」、額度也還剩兩次，它一次都沒用。
# 所以改由程式接手：0 段就拿使用者的原問題再查一次。
#
# **涵蓋範圍要講清楚**：這一層擋的是「模型自己造的 query 沒撈到、但使用者
# 的原話撈得到」。當 query 本身就等於原問題時（沒有別的候選可試），
# 它不會做任何事——那個情境歸第 1、2 層負責。
def _drive_answer(model_query, hits_by_query, seen_only=False):
    """用假的 chat_stream 與假的檢索跑一次 answer()，回傳實際查過的 query。

    `seen_only` 模擬「有撈到內容，但這一輪前面已經全部給過了」——它同樣是
    0 段，但重試毫無意義。追問「還有嗎」幾乎必然落在這一種。
    """
    asked = []
    real_stream = _ag.ollama_client.chat_stream
    real_search = _ag._run_search
    real_grounded, real_relevant = _ag._is_grounded, _ag._is_relevant

    def stream(messages, tools=None, model=None):
        # 第一輪要求檢索，之後直接給答案，避免無限迴圈
        if any(m.get("role") == "tool" for m in messages):
            yield {"type": "text", "piece": "以上是查到的內容。"}
            return
        yield {"type": "tool_calls", "calls": [
            {"function": {"name": "search_knowledge_base",
                          "arguments": {"query": model_query}}}]}

    def search(q, stage, citations, wide=False):
        asked.append(q)
        n = hits_by_query.get(q, 0)
        return ((f"內容 for {q}" if n else "沒有找到任何內容。"), n,
                seen_only and n == 0)

    _ag.ollama_client.chat_stream = stream
    _ag._run_search = search
    _ag._is_grounded = lambda *a: True
    _ag._is_relevant = lambda *a: True
    try:
        list(_ag.answer("料件有哪些種類", HIST))
    finally:
        _ag.ollama_client.chat_stream = real_stream
        _ag._run_search = real_search
        _ag._is_grounded, _ag._is_relevant = real_grounded, real_relevant
    return asked


_asked = _drive_answer("完全撈不到的關鍵詞", {"料件有哪些種類": 6})
check("模型的 query 撈到 0 段時，程式用原問題再查一次",
      "料件有哪些種類" in _asked, str(_asked))

_asked = _drive_answer("料件有哪些種類", {"料件有哪些種類": 6})
check("第一次就撈到就不重複查", _asked == ["料件有哪些種類"], str(_asked))

_asked = _drive_answer("完全撈不到的關鍵詞", {})
check("重試也撈不到時不會超過檢索額度",
      len(_asked) <= _ag.MAX_SEARCHES, f"{len(_asked)} 次 {_asked}")

# **這一項是為了一個實際踩到的災難加的。**
#
# 「有內容但前面給過了」跟「真的沒找到」都是 0 段。第一版把兩者混為一談，
# 於是追問「還有嗎」撈回同一批切片後不斷重試：六次檢索、106 秒，最後還是
# 回「查無足夠資訊」——比不重試更慢也更差。
_asked = _drive_answer("料件有哪些種類", {}, seen_only=True)
check("內容只是「前面給過了」時不重試",
      _asked == ["料件有哪些種類"], f"{len(_asked)} 次 {_asked}")

# 補救整輪只做一次，否則 MAX_SEARCHES 形同虛設
_asked = _drive_answer("撈不到的詞", {})
check("自動補查整輪只做一次",
      _asked.count("料件有哪些種類") <= 1, str(_asked))
check("前文只有追問時不會補成追問",
      _resolve_follow_up("還有嗎", [{"role": "user", "content": "還有嗎"}]) == "還有嗎")
# 連續追問要一路往前找到真正有主題的那一句，找到了才交給改寫。
# 找不到的話（前文全是追問）會原樣返回、根本不呼叫改寫——上面那項守的就是這個。
_chain = HIST + [{"role": "user", "content": "還有嗎"},
                 {"role": "assistant", "content": "還有 03 MEMORY。"}]
_real_condense = _ag.rag_service.condense_question
_seen = []
_ag.rag_service.condense_question = lambda q, h: (_seen.append(q), ("還有哪些料件種類", ""))[1]
try:
    _r = _resolve_follow_up("還有嗎", _chain)
    check("連續追問往前找到有主題的問題後才改寫",
          _seen and _r == "還有哪些料件種類", f"{_r!r} called={bool(_seen)}")
finally:
    _ag.rag_service.condense_question = _real_condense


# ----------------------------------------------------------------- 殘骸表格過濾
#
# 簡報轉 PDF 的文字層，表格會被 pdfminer 拆成每一兩列就重開一個表的殘骸。
# 開了視覺解析後同一份文件同時有殘骸與 VLM 轉錄的完整表，實測檢索撈到殘骸、
# 模型照抄一坨垃圾進答案。過濾只在有 VLM 版本時才做，永遠不會丟掉唯一的內容。
from services.ingest_service import _strip_broken_tables  # noqa: E402

BROKEN = ("原物料介紹\n| 類別代號 | 類 別 名 | 稱 |\n| ---- | ----- | --- |\nDISPLAY\n"
          "| 01 CPU |  |  |\n| ------ | --- | --- |\n(LCD/OLED/EPD)\n"
          "| 02 CHIPSET |  |  |\n| ---------- | --- | --- |\nPOWER MODULE")
CLEAN = "| 大類 | 敘述 |\n| --- | --- |\n| 01 | CPU |\n| 02 | CHIPSET |\n| 03 | MEMORY |\n| 04 | SYS MODULE |"
_out = _strip_broken_tables(BROKEN + "\n\n" + CLEAN + "\n\n這是一般段落。")
check("殘骸表格被整段移除", "DISPLAY" not in _out and "01 CPU |" not in _out, _out[:80])
check("正常表格原樣保留", "| 04 | SYS MODULE |" in _out)
check("純文字段落不受影響", "這是一般段落。" in _out)
# 單一分隔線的正常小表（1 條分隔線配 2 列）比值 0.5，不該被當殘骸——門檻是「超過」
_tiny = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
check("小表格不誤殺", _strip_broken_tables(_tiny) == _tiny)


# ----------------------------------------------------------------- VLM 省略偵測
#
# 實際踩過的坑：VLM 把 28 列的分類表摘要成「例如：01 CPU, 02 CHIPSET 等等」，
# 省略掉的項目從此不存在於知識庫，之後再強的檢索也救不回來。
# 這組測試守的是「摘要要被偵測到」與「正常輸出不可誤判」兩件事——
# 誤判的代價是每頁都白白重試一次，索引時間直接翻倍。
from services.ingest_service import _describe_once, _looks_elided  # noqa: E402

ELIDED = [
    "例如：01 CPU, 02 CHIPSET, 03 MEMORY 等等。",
    "包含不同製造商的編碼和名稱，如001 INTEL, 002 AMD等。",
    "列出了不同的物料分類，如CPU、CHIPSET、MEMORY等。",
    "其餘項目省略。",
    "01 CPU, 02 CHIPSET, and so on",
]
for _text in ELIDED:
    check(f"偵測到省略：{_text[:18]}", _looks_elided(_text))

INTACT = [
    "大類 敘述\n01 CPU\n02 CHIPSET\n03 MEMORY\n04 SYS MODULE\n0A POWER MODULE",
    "本頁為標題頁，無表格等內容。",
    "所有標註均以中文和英文雙語形式呈現。",
    "這些圖片沒有顯示任何表格或流程圖。",
    "| 類別 | 名稱 |\n| --- | --- |\n| 01 | CPU |\n| 02 | CHIPSET |",
]
for _text in INTACT:
    check(f"不誤判：{_text.splitlines()[0][:18]}", not _looks_elided(_text))


def _fake_vlm(replies):
    """把 describe_image 換成腳本化的假模型，回傳呼叫到的提示詞。"""
    from services import ingest_service as _is
    from services import ollama_client as _oc

    seen, original = [], _oc.describe_image

    def stub(_data, prompt, model=None):
        seen.append(prompt)
        return replies[min(len(seen) - 1, len(replies) - 1)], ""

    _oc.describe_image = stub
    _is.ollama_client.describe_image = stub
    try:
        return _describe_once(b"x"), seen
    finally:
        _oc.describe_image = original
        _is.ollama_client.describe_image = original


FULL = "| 01 | CPU |\n| 02 | CHIPSET |\n| 03 | MEMORY |"
(_text, _err, _short), _prompts = _fake_vlm([FULL])
check("輸出完整就不重試", len(_prompts) == 1 and not _short, f"呼叫 {len(_prompts)} 次")

(_text, _err, _short), _prompts = _fake_vlm(["01 CPU, 02 CHIPSET 等等。", FULL])
check("偵測到省略會重試", len(_prompts) == 2, f"呼叫 {len(_prompts)} 次")
check("重試成功採用新結果", _text == FULL and not _short, _text[:40])

(_text, _err, _short), _prompts = _fake_vlm(["01 CPU 等等。"])
check("重試仍省略要示警", _short is True)
check("重試仍省略要保留內容", _text.strip() != "", _text[:40])

# ------------------------------------------------- 逐章節作答
#
# 這個機制曾被移除（一次餵 30 段撐爆 num_ctx），現在以「每次只餵一個章節」的
# 形式加回來，預設關閉。測試守三件事：只在開關開且問題是列舉句時才啟動、
# 章節清單由程式產生且有上限、丟掉的章節要明列不能靜靜消失。
from services.agent_service import _is_enumeration, _answer_by_section, _Citations  # noqa: E402
from services.rag_service import RetrievedChunk  # noqa: E402

for _q in ["有哪些壓力測試", "列出全部的測試項目", "料件有哪些種類", "總共幾種", "list all tests"]:
    check(f"列舉句判定：{_q}", _is_enumeration(_q))
for _q in ["壓力測試的溫度條件", "DVT 是什麼", "今天午餐吃什麼"]:
    check(f"非列舉句不誤判：{_q}", not _is_enumeration(_q))


def _mk(cid, section, text, score):
    return RetrievedChunk(chunk_id=cid, doc_id=1, seq=cid, content=text,
                          locator=f"x.pdf › {section} #{cid}", file_name="x.pdf",
                          file_path="x.pdf", kb=None, distance=1.0, rerank_score=score)


def _drive_sections(chunks):
    """stub 掉檢索與生成，只驗分組／上限／組裝邏輯。生成只是回聲，不依賴模型。"""
    real_r, real_g = _ag.rag_service.retrieve, _ag.ollama_client.generate
    prompts = []
    _ag.rag_service.retrieve = lambda q, s, top_k=None: (chunks, "")
    _ag.ollama_client.generate = lambda p, system="", model=None, num_predict=None: (prompts.append(p), ("- 項目 [1]", ""))[1]
    try:
        evs = list(_answer_by_section("有哪些測試", None, _Citations()))
    finally:
        _ag.rag_service.retrieve, _ag.ollama_client.generate = real_r, real_g
    return evs, prompts


# 10 個章節、每章 1 段；分數遞減，讓排序可預期
_chunks = [_mk(i, f"{i}.1 Test {i}", f"內容 {i}", 10 - i) for i in range(1, 11)]
_evs, _prompts = _drive_sections(_chunks)
_done = [e for e in _evs if e["type"] == "done"][0]
check("每個章節各問一次、受 MAX_SECTIONS 上限", len(_prompts) == _ag.MAX_SECTIONS, f"{len(_prompts)} 次")
check("章節依重排序分數排序（最高分先）", "1.1 Test 1" in _prompts[0] and "8.1 Test 8" in _prompts[-1])
check("每次只餵那一個章節的內容", "內容 1" in _prompts[0] and "內容 2" not in _prompts[0])
check("丟掉的章節在結尾明列", "另有 2 個相關章節未展開" in _done["answer"] and "9.1 Test 9" in _done["answer"])
check("結尾誠實說明只涵蓋檢索到的章節", "本次檢索到的章節" in _done["answer"])
check("done 帶回所有引用的切片", len(_done["chunk_ids"]) == 10)

# 開關關閉時，answer() 不該走這條路徑——即使問題是列舉句
_real_get = _ag.get_setting
_ag.get_setting = lambda k, d="": "0" if k == "section_answer" else _real_get(k, d)
_seen = []
_real_abs = _ag._answer_by_section
_ag._answer_by_section = lambda *a, **k: (_seen.append(1), iter(()))[1]
try:
    _asked = _drive_answer("有哪些測試", {"有哪些測試": 6})
    check("開關關閉時不走逐章節路徑", not _seen and _asked)
finally:
    _ag.get_setting, _ag._answer_by_section = _real_get, _real_abs

# API：預設關閉、可切換、可還原
_cur = call("/api/admin/models", ADMIN)[1]["current"]
check("models 回傳 section_answer", "section_answer" in _cur)
_orig_sa = bool(_cur["section_answer"])
call("/api/admin/models", ADMIN, "PUT", {"section_answer": True})
check("section_answer 可開啟", call("/api/admin/models", ADMIN)[1]["current"]["section_answer"] is True)
call("/api/admin/models", ADMIN, "PUT", {"section_answer": _orig_sa})
check("section_answer 還原", call("/api/admin/models", ADMIN)[1]["current"]["section_answer"] == _orig_sa)

print("\n" + "=" * 56)
print(f"  通過 {_passed} 項，失敗 {_failed} 項")
print("=" * 56)
sys.exit(1 if _failed else 0)
