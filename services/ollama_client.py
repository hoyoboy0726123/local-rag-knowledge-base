"""Ollama 本機推理客戶端。

全程只連 localhost，不對外發送任何請求 —— 這是本系統存在的前提。
Ollama 未啟動時所有函式都回傳明確的失敗訊息而非拋例外，
讓 UI 能停用 AI 功能但保留文件瀏覽與關鍵字搜尋。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from database import get_setting

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
            json={"model": model, "input": texts},
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
        "options": {"temperature": 0.1},
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
        "options": {"temperature": 0.1},
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
    options: dict = {"temperature": 0.1}
    if num_predict:
        options["num_predict"] = num_predict

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
