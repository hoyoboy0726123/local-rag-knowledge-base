"""Ollama 本機推理客戶端。

全程只連 localhost，不對外發送任何請求 —— 這是本系統存在的前提。
Ollama 未啟動時所有函式都回傳明確的失敗訊息而非拋例外，
讓 UI 能停用 AI 功能但保留文件瀏覽與關鍵字搜尋。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from database import get_int_setting, get_setting

TIMEOUT_SHORT = 15
TIMEOUT_EMBED = 300
TIMEOUT_GENERATE = 300

# 連線復用。每次 requests.post() 都會重開 TCP 連線，
# 而建立連線的成本在這裡遠大於推理本身（見下方 _normalise_host 的說明）。
_SESSION = requests.Session()

# NO_THINK_NOTE ---------------------------------------------------------
# 所有生成請求都帶 "think": False。
#
# `gemma4:12b` 是推理模型。不指定 think 時它仍會產生推理 token，
# 但 Ollama 不會把推理內容回傳，於是那段時間對使用者而言純粹是空等。
# 同一個問題實測：
#
#   未指定 think ：首字 10.00s，全文 13.53s，eval 833 tok，可見輸出 364 字
#   think=False  ：首字  2.26s，全文  8.43s，eval 264 tok，可見輸出 398 字
#
# 約七成的 token 花在看不見的內容上。關掉之後答案品質未下降
# （拒答無關問題、標註來源皆正常，見 test_core.py 第 10.2 節）。
#
# `think: False` 對非推理模型也安全，Ollama 會忽略；
# 只有 `think: True` 才會讓非推理模型回 400。因此不需要先偵測模型能力。
# 若日後想看推理過程，把這裡改成 True 即可，但要有心理準備會慢一倍以上。
# -----------------------------------------------------------------------

# NUM_CTX_NOTE ----------------------------------------------------------
# 生成請求一律指定 num_ctx，**不能沿用 Ollama 的預設值**。
#
# Ollama 預設的上下文是 4096，而這個系統的提示詞（系統指令 + 檢索結果 +
# 前後文擴展）實測約 4100～4300 token，正好超過。超過的部分不會報錯，
# 而是被**安靜丟掉**，於是模型收到的是殘缺的提示。
#
# 同一組訊息實測（gemma4:12b，檢索結果 5688 字）：
#
#   num_ctx=4096 ：prompt 只剩 820 token → 檢索結果整段消失。模型手上
#                  沒有任何資料，於是把系統提示複述一遍當作答案
#   num_ctx=8192 ：prompt 4147 token     → 完整，正常作答
#   num_ctx=16384：prompt 4147 token     → 與 8192 相同，沒有額外好處
#
# 另一種症狀是提示詞剛好塞得下、卻沒有空間生成：某題 prompt 4047 token，
# 只剩 49 個可用，答案講到一半就 done_reason=length 斷掉。**調 num_predict
# 碰不到這個瓶頸**——實際可生成的量是 num_ctx 減掉提示詞，不是 num_predict。
#
# 8192 是實測的最小夠用值。調高會增加 KV cache 佔用：gemma4:12b 在 8192 下
# 總佔用 8.9 GB，8 GB 顯卡會有三成的層被丟到 CPU 而變慢（qwen3:8b 是
# 6.2 GB，100% 留在 GPU）。因此做成可設定項而不是寫死。
# -----------------------------------------------------------------------
DEFAULT_NUM_CTX = 8192

# 固定一個 num_ctx 必然會錯，因為提示詞長度差距很大。同一個知識庫實測：
#
#   一般問題    3,701 ~ 4,559 token
#   更多脈絡    切片夠多時上看 11,000（WIDE_TOTAL_MAX_CHUNKS=40）
#
# 設低了被安靜截斷，設高了白白吃顯示記憶體。所以依實際長度往上選。
#
# **量化成級距，不是算出精確值**：Ollama 每遇到一個沒見過的 num_ctx 就會重新
# 載入模型，逐題微調等於每題都重載。級距只有三段，實務上最多在兩種之間切換。
CTX_STEPS = (8192, 16384, 32768)

# 中英混排的實測換算：10,295 字 → 5,977 token、5,688 字 → 3,289 token，
# 都落在每字 0.58 附近。取 0.65 留安全邊際（估太低會回到被截斷的老問題）。
CHARS_TO_TOKENS = 0.65

# 留給生成的空間。逐章節作答的 num_predict 是 1200，列舉型答案更長；
# 留 2048 才不會發生「提示塞得下但沒有空間回答」——那也是 done_reason=length。
CTX_RESERVE_TOKENS = 2048


def _ctx_for(prompt_text: str) -> int:
    """依提示詞長度挑一個夠用的 num_ctx。見上方 CTX_STEPS 的說明。

    設定值當作**下限**而不是固定值：使用者調高代表願意付出記憶體換取更多
    脈絡，調低則不該讓系統因此開始安靜丟內容，所以只會往上選、不會往下。
    """
    floor = get_int_setting("num_ctx", DEFAULT_NUM_CTX)
    need = int(len(prompt_text) * CHARS_TO_TOKENS) + CTX_RESERVE_TOKENS
    for step in CTX_STEPS:
        if step >= need and step >= floor:
            return step
    return max(CTX_STEPS[-1], floor)


def _gen_options(extra: dict | None = None, prompt_text: str = "") -> dict:
    """生成請求共用的 options。見 NUM_CTX_NOTE 與 CTX_STEPS。"""
    options: dict = {
        "temperature": 0.1,
        "num_ctx": _ctx_for(prompt_text),
    }
    if extra:
        options.update(extra)
    return options


def _normalise_host(raw: str) -> str:
    """把 localhost 換成 127.0.0.1。

    **這不是美化，是效能修正。** Windows 上 `localhost` 會優先解析到 IPv6 `::1`，
    但 Ollama 預設只監聽 IPv4，於是每個請求都要先卡滿 IPv6 連線逾時才退回 IPv4。
    實測固定多付約 2.0 秒——單句 embed 從 0.06 秒變成 2.11 秒。
    使用者自行填寫的主機位址不動，只處理 localhost 這個特例。
    """
    return raw.replace("//localhost:", "//127.0.0.1:")


def host() -> str:
    raw = (get_setting("ollama_host", "http://127.0.0.1:11434") or "").rstrip("/")
    return _normalise_host(raw)


@dataclass
class OllamaStatus:
    alive: bool
    message: str
    models: list[str]


def check_status() -> OllamaStatus:
    try:
        resp = _SESSION.get(f"{host()}/api/tags", timeout=TIMEOUT_SHORT)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return OllamaStatus(True, f"已連線（{len(models)} 個模型可用）", models)
    except requests.exceptions.ConnectionError:
        return OllamaStatus(False, "無法連線至 Ollama，請確認服務已啟動", [])
    except Exception as exc:  # noqa: BLE001
        return OllamaStatus(False, f"{type(exc).__name__}: {str(exc)[:150]}", [])


def is_alive() -> bool:
    return check_status().alive


def embed(texts: list[str], model: str | None = None) -> tuple[list[list[float]], str]:
    """批次產生向量。回傳 (向量清單, 錯誤訊息)。"""
    if not texts:
        return [], ""

    model = model or get_setting("embed_model")
    try:
        resp = _SESSION.post(
            f"{host()}/api/embed",
            # EMBED_ON_CPU_NOTE --------------------------------------------
            # `num_gpu: 0` 把向量模型釘在 CPU 上跑。**這是為了讓生成模型能
            # 一直留在顯卡上，不是為了加速 embedding 本身。**
            #
            # 問答的每一輪都是「先 embed、再生成」。顯存不夠同時放兩個模型時，
            # Ollama 會在每一步把另一個踢掉——等於每問一題就要重載兩次模型，
            # 其中生成模型有 6.2 GB。
            #
            # 實測（RTX 5060 8 GB，Ollama 實際可用 6.4 GiB，qwen3:8b 佔 6.2 GB）：
            #
            #            embed          生成         每輪合計
            #   走 GPU   2.9~3.8s      4.1~4.7s      約 7.9s   ← 幾乎都是重載
            #   走 CPU   0.26s（暖）   0.27s（暖）   約 0.55s
            #
            # `ollama ps` 也直接看得到差別：走 GPU 時永遠只有一個模型，
            # 走 CPU 之後兩個並存。
            #
            # 代價幾乎沒有：向量模型只處理一個短問句，工作量小，CPU 上暖機後
            # 0.26 秒，跟它在 GPU 上的 0.23 秒相當。佔用的是系統記憶體。
            #
            # 索引時要對成千上萬段做 embedding，那才是吃算力的場景——但索引
            # 本來就不與生成交錯，沒有互相驅逐的問題。若日後發現大批次索引
            # 變慢，可以只在這條查詢路徑上釘 CPU。
            # ---------------------------------------------------------------
            json={"model": model, "input": texts, "options": {"num_gpu": 0}},
            timeout=TIMEOUT_EMBED,
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings", [])
        if len(embeddings) != len(texts):
            return [], f"回傳向量數量不符（預期 {len(texts)}，實得 {len(embeddings)}）"
        return embeddings, ""
    except requests.exceptions.ConnectionError:
        return [], "無法連線至 Ollama"
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {str(exc)[:200]}"


def generate_stream(prompt: str, system: str = "", model: str | None = None):
    """串流生成。逐段 yield 文字，讓使用者不必等整段跑完。"""
    model = model or get_setting("llm_model")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": False,  # 見 NO_THINK_NOTE
        "options": _gen_options(prompt_text=system + prompt),  # 見 NUM_CTX_NOTE
    }
    if system:
        payload["system"] = system

    try:
        with _SESSION.post(
            f"{host()}/api/generate", json=payload, stream=True, timeout=TIMEOUT_GENERATE
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        yield "\n\n⚠️ 無法連線至本機 AI 引擎，請確認 Ollama 服務已啟動。"
    except Exception as exc:  # noqa: BLE001
        yield f"\n\n⚠️ 生成失敗：{type(exc).__name__}: {str(exc)[:150]}"


def supports_tools(model: str | None = None) -> bool:
    """模型是否支援工具調用。

    管理員可以在「模型設定」換成任何已安裝的模型，其中不乏不支援工具的。
    問答流程據此決定要走 Agent 迴圈還是傳統的單次檢索。
    """
    model = model or get_setting("llm_model")
    try:
        resp = _SESSION.post(f"{host()}/api/show", json={"model": model}, timeout=TIMEOUT_SHORT)
        resp.raise_for_status()
        return "tools" in (resp.json().get("capabilities") or [])
    except Exception:  # noqa: BLE001
        return False


def chat_stream(messages: list[dict], tools: list[dict] | None = None,
                model: str | None = None):
    """串流對話。逐一 yield 事件字典：

        {"type": "text", "piece": str}          逐段文字
        {"type": "tool_calls", "calls": [...]}  模型要求呼叫工具
        {"type": "error", "message": str}       失敗

    工具調用與串流可以並存：模型決定呼叫工具時不會產生文字，
    決定作答時則逐字吐出，因此同一條路徑就能兼顧兩者。
    """
    model = model or get_setting("llm_model")
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": False,  # 見 NO_THINK_NOTE
        # 工具定義也會進提示詞，但它是固定成本且遠小於檢索結果，忽略不計。
        "options": _gen_options(
            prompt_text="".join(m.get("content") or "" for m in messages)
        ),
    }
    if tools:
        payload["tools"] = tools

    calls: list[dict] = []
    try:
        with _SESSION.post(
            f"{host()}/api/chat", json=payload, stream=True, timeout=TIMEOUT_GENERATE
        ) as resp:
            if resp.status_code != 200:
                yield {"type": "error", "message": f"HTTP {resp.status_code}: {resp.text[:200]}"}
                return
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = data.get("message") or {}
                if message.get("tool_calls"):
                    calls.extend(message["tool_calls"])
                piece = message.get("content", "")
                if piece:
                    yield {"type": "text", "piece": piece}
                if data.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        yield {"type": "error", "message": "無法連線至 Ollama"}
        return
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": f"{type(exc).__name__}: {str(exc)[:150]}"}
        return

    if calls:
        yield {"type": "tool_calls", "calls": calls}


def generate(
    prompt: str, system: str = "", model: str | None = None, num_predict: int | None = None
) -> tuple[str, str]:
    """一次性生成，回傳 (文字, 錯誤訊息)。

    供不需要串流的短任務使用（例如把追問改寫成完整問題）。
    """
    model = model or get_setting("llm_model")
    # 見 NUM_CTX_NOTE：num_predict 只限制「最多生成幾個」，真正的天花板是
    # num_ctx 扣掉提示詞之後剩下的量，兩者都要設。
    options = _gen_options({"num_predict": num_predict} if num_predict else None,
                           prompt_text=system + prompt)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,  # 見 NO_THINK_NOTE
        "options": options,
    }
    if system:
        payload["system"] = system

    try:
        resp = _SESSION.post(
            f"{host()}/api/generate", json=payload, timeout=TIMEOUT_GENERATE
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip(), ""
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {str(exc)[:150]}"


def describe_image(image_bytes: bytes, prompt: str, model: str | None = None) -> tuple[str, str]:
    """以 VLM 描述圖片內容。供圖多的簡報與掃描件使用。"""
    import base64

    model = model or get_setting("vlm_model")
    try:
        resp = _SESSION.post(
            f"{host()}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [base64.b64encode(image_bytes).decode()],
                "stream": False,
                "think": False,  # 見 NO_THINK_NOTE
                "options": {"temperature": 0.1},
            },
            timeout=TIMEOUT_GENERATE,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip(), ""
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {str(exc)[:150]}"
