"""Cross-encoder 重排序（選配）。

**為什麼需要**：向量檢索是雙塔架構，問題與切片各自編碼再比距離，
分不出「沾邊」與「切題」。實測一題「Pressure test 有哪些種類」，
30 段候選的距離全部擠在 **0.922–0.962**（只差 0.04），等於沒有排序能力——
真正切題的那個章節排第 26，
而主題僅字面相近、實質不相關的章節反而排第 3。
結果模型讀到後段時已經寫完答案，穩定漏掉排在尾端的那一項（連跑 4 次都是 5/6）。

Cross-encoder 把「問題 + 切片」一起送進模型算相關性，能真正讀懂兩者的關係。
實測同一批候選重排後：三個原本排 26／30／23 的切題章節，分別升到 **4／2／5**。

**為什麼用 ONNX 而不是 sentence-transformers**：後者會帶進 PyTorch（約 4 GB），
而這個專案刻意避開了整條 torch 依賴鏈（見 README「技術選型」）。
`onnxruntime` 本來就在相依裡（MarkItDown 帶進來的），只需要再加一個
`tokenizers`（Rust 實作，3 MB）就能跑，**完全不需要 torch**。

**為什麼是選配**：模型檔 571 MB，不適合放進交付包。
模型不存在時 `rerank()` 直接回傳原順序，功能照常運作，只是排序不會變好。
"""

from __future__ import annotations

import threading
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "bge-reranker-v2-m3"
ONNX_PATH = MODEL_DIR / "model_quantized.onnx"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

# 模型來源。用量化版：571 MB，重排序只需要相對分數，精度損失可接受。
_BASE_URL = "https://huggingface.co/rafa754/bge-reranker-v2-m3/resolve/main"
DOWNLOADS = {
    ONNX_PATH: f"{_BASE_URL}/onnx/model_quantized.onnx",
    TOKENIZER_PATH: f"{_BASE_URL}/tokenizer.json",
}

# 送進 cross-encoder 的最大長度。bge-reranker 是 512 token 的模型，
# 而切片上限 500 字，加上問題剛好在範圍內。
MAX_LENGTH = 512

# 每段約 72 ms（CPU 實測）。候選太多會讓每次提問多等好幾秒，
# 因此設上限——排在 48 名之後的候選幾乎不可能被重排到前面。
MAX_CANDIDATES = 48

_lock = threading.Lock()
_session = None
_tokenizer = None
_load_error = ""


def is_installed() -> bool:
    """模型檔是否已備妥。"""
    return ONNX_PATH.exists() and TOKENIZER_PATH.exists()


def status() -> dict:
    return {
        "installed": is_installed(),
        "loaded": _session is not None,
        "error": _load_error,
        "path": str(MODEL_DIR),
        "size_mb": round(ONNX_PATH.stat().st_size / 1e6) if ONNX_PATH.exists() else 0,
    }


def _load() -> bool:
    """延遲載入。第一次呼叫才付出載入成本，沒裝模型的人完全不受影響。"""
    global _session, _tokenizer, _load_error
    if _session is not None:
        return True
    if not is_installed():
        return False
    with _lock:
        if _session is not None:
            return True
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
            tokenizer.enable_truncation(max_length=MAX_LENGTH)
            tokenizer.enable_padding()
            _tokenizer = tokenizer
            _session = ort.InferenceSession(
                str(ONNX_PATH), providers=["CPUExecutionProvider"]
            )
            _load_error = ""
            return True
        except Exception as exc:  # noqa: BLE001
            # 載入失敗不能讓問答掛掉——重排序是加分項，不是必要路徑。
            _load_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            _session = None
            return False


def score(query: str, documents: list[str]) -> list[float] | None:
    """回傳每份文件與查詢的相關性分數。無法使用時回傳 None。"""
    if not documents or not _load():
        return None
    try:
        import numpy as np

        encoded = _tokenizer.encode_batch([(query, doc) for doc in documents])
        feed = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
        }
        names = {i.name for i in _session.get_inputs()}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        logits = _session.run(None, feed)[0]
        return [float(x) for x in np.asarray(logits).reshape(-1)]
    except Exception as exc:  # noqa: BLE001
        global _load_error
        _load_error = f"推論失敗 {type(exc).__name__}: {str(exc)[:150]}"
        return None


def download(progress=None) -> tuple[bool, str]:
    """下載模型檔。回傳 (成功, 訊息)。

    用 `requests` 直接抓，不引入 `huggingface_hub`——為了兩個檔案多一個相依不划算。
    """
    import requests

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in DOWNLOADS.items():
        if path.exists():
            continue
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                step = max(total // 20, 1)
                next_mark = step
                with open(tmp, "wb") as fh:
                    for block in resp.iter_content(chunk_size=1 << 20):
                        fh.write(block)
                        done += len(block)
                        if progress and total and done >= next_mark:
                            progress(f"  {path.name} {done / total * 100:.0f}%"
                                     f"（{done / 1e6:.0f}/{total / 1e6:.0f} MB）")
                            next_mark += step
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            return False, f"下載 {path.name} 失敗：{type(exc).__name__}: {str(exc)[:200]}"
    return True, f"模型已備妥（{ONNX_PATH.stat().st_size / 1e6:.0f} MB）"
