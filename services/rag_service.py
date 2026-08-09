"""語意檢索與問答。

用 sqlite-vec 的好處在這裡最明顯：檢索時可以直接 JOIN documents，
以一般 SQL 的 WHERE 做階段過濾，不需要另一套 metadata filter 語法。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from database import get_int_setting, get_session, get_setting, raw_connection, serialize
from models import Chunk, Document
from services import ollama_client, reranker

# 距離保險絲，**不是**相關性判斷的主要依據。
#
# 早期版本用 1.05 作為硬性門檻，結果誤擋了 43% 的合理問題。
# 以 46 個真實問題（含系統自己的 FAQ 建議、短關鍵詞、口語化措辭）實測：
#
#   相關問題：0.724 ~ 1.209
#   無關問題：1.148 ~ 1.230
#
# **兩群明顯重疊**，單一絕對距離無法分開。原本的校準之所以看起來成功，
# 只是因為當時挑的題目都是措辭精準的完整問句，不具代表性。
# 最明顯的破綻是「有哪些常見的失誤或 Lesson Learnt？」——
# 這是系統自己放在 FAQ 建議按鈕裡的問題，距離 1.183 卻被自己擋掉。
#
# 改為由 LLM 判斷相關性：它能真的讀內容，距離不能。
# system instruction 已明確要求「參考資料不足時說知識庫中查無足夠資訊」，
# 實測對無關問題（午餐、股票、寫詩）都能正確拒答，
# 對距離偏高但實際相關的問題則能正確作答並標註來源。
#
# 這個常數現在只擋「連最近的切片都離譜地遠」的極端情況，
# 用來省下明顯無謂的 LLM 呼叫。
DISTANCE_THRESHOLD = 1.35

SYSTEM_INSTRUCTION = (
    "你是 NUC 研發流程的知識庫助理，服務對象是 RD 專案經理。\n"
    "規則：\n"
    "1. 只依據提供的參考資料作答，不要補充資料中沒有的內容。\n"
    "2. 每個結論後面標註來源編號，格式為 [1]、[2]。\n"
    "3. 若參考資料不足以回答，明確說「知識庫中查無足夠資訊」，不要猜測。\n"
    "4. 參考資料依文件分組。**不可把不同文件的數值或條件合併成同一組結論**——"
    "不同文件可能是不同的標準、版本或適用範圍。若說法不一致，分別說明各自的出處。\n"
    "5. 以繁體中文回答，條列說明，語氣專業簡潔。"
)


@dataclass
class RetrievedChunk:
    chunk_id: int
    doc_id: int
    seq: int
    content: str
    locator: str
    file_name: str
    file_path: str
    stage_code: str | None
    distance: float
    # 重排序分數（越大越相關）。沒啟用重排序時是 None。
    #
    # **必須帶出來，不能只用在排序後就丟掉。** 向量距離的鑑別力很差
    # （實測 30 段候選只差 0.04），下游若要做「選最相關的 N 個章節」這種決定，
    # 用距離會選錯——實測就把真正切題的章節排到第 12 名，
    # 反而選進了主題完全不同的章節。
    rerank_score: float | None = None


def retrieve(query: str, stage_code: str | None = None, top_k: int | None = None) -> tuple[list[RetrievedChunk], str]:
    """語意檢索。回傳 (切片清單, 錯誤訊息)。"""
    if not query.strip():
        return [], "查詢內容為空"

    top_k = top_k or get_int_setting("top_k", 6)

    vectors, error = ollama_client.embed([query])
    if error:
        return [], error
    if not vectors:
        return [], "無法產生查詢向量"

    conn = raw_connection()
    try:
        # sqlite-vec 的 KNN 需要先取一批候選，再以 SQL 條件過濾
        limit = (top_k * 8 if stage_code else top_k * 2) + reranker.MAX_CANDIDATES
        sql = """
            SELECT v.chunk_id, v.distance, c.doc_id, c.content, c.locator,
                   d.file_name, d.file_path, d.stage_code, c.seq
            FROM (
                SELECT chunk_id, distance FROM vec_chunks
                WHERE embedding MATCH ? ORDER BY distance LIMIT ?
            ) v
            JOIN chunks c ON c.id = v.chunk_id
            JOIN documents d ON d.id = c.doc_id
        """
        params: list = [serialize(vectors[0]), limit]
        if stage_code:
            sql += " WHERE d.stage_code = ?"
            params.append(stage_code)
        # 有重排序時多撈候選讓它排——只給它 top_k 段就沒有東西可以重排了。
        # 實測那題真正要的切片原本排第 26，只撈 6 段根本看不到它。
        want = top_k
        if reranker.is_installed() and get_setting("enable_rerank", "1") == "1":
            want = min(max(top_k * 2, top_k + 12), reranker.MAX_CANDIDATES)
        sql += " ORDER BY v.distance LIMIT ?"
        params.append(want)

        rows = conn.execute(sql, params).fetchall()
    except Exception as exc:  # noqa: BLE001
        return [], f"檢索失敗：{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        conn.close()

    results = [
        RetrievedChunk(
            chunk_id=r[0], distance=r[1], doc_id=r[2], content=r[3],
            locator=r[4], file_name=r[5], file_path=r[6], stage_code=r[7],
            seq=r[8],
        )
        for r in rows
    ]
    return _apply_rerank(query, results, top_k), ""


def _apply_rerank(query: str, chunks: list[RetrievedChunk],
                  top_k: int) -> list[RetrievedChunk]:
    """用 cross-encoder 重排，取前 top_k。模型沒裝就原樣回傳。

    **這一步是可選的加分項，任何失敗都不能影響問答。** 因此沒有例外處理分支：
    `reranker.score()` 自己吞掉所有錯誤並回 None，這裡只判斷有沒有拿到分數。
    """
    if len(chunks) <= 1 or get_setting("enable_rerank", "1") != "1":
        return chunks[:top_k]
    scores = reranker.score(query, [c.content for c in chunks])
    if not scores or len(scores) != len(chunks):
        return chunks[:top_k]
    for chunk, s in zip(chunks, scores):
        chunk.rerank_score = s
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    return [chunks[i] for i in order[:top_k]]


def best_score(chunks: list[RetrievedChunk]) -> float:
    """一組切片的相關性代表值，**越大越相關**。

    有重排序分數就用它；沒有時用距離取負號，讓排序方向一致。
    這樣呼叫端可以一律 `sorted(..., key=lambda g: -best_score(g))`，
    不必分兩種情況寫。
    """
    scored = [c.rerank_score for c in chunks if c.rerank_score is not None]
    if scored:
        return max(scored)
    return -min(c.distance for c in chunks)


def has_relevant(chunks: list[RetrievedChunk]) -> bool:
    """粗略的保險絲，**不是**最終的相關性判斷。

    真正判斷內容有沒有回答到問題的是 LLM（見 SYSTEM_INSTRUCTION 第 3 條）。
    這裡只擋「連最近的切片都離譜地遠」的極端情況。
    """
    return bool(chunks) and chunks[0].distance <= DISTANCE_THRESHOLD


# 帶入對話脈絡時，只取最近幾輪並截斷過長的回答。
# 助理的回答動輒上千字，全帶會把參考資料擠掉，反而讓 LLM 抓錯重點。
HISTORY_TURNS = 3
HISTORY_ANSWER_CHARS = 400

CONDENSE_SYSTEM = (
    "判斷使用者這句話能不能「不看對話紀錄也理解」。\n"
    "能 → 原封不動輸出。\n"
    "不能 → 補上缺少的主詞或範圍，改寫成獨立問題。\n\n"
    "規則：\n"
    "1. 只輸出問題本身，不要說明、引號或前綴。\n"
    "2. 不要加入使用者沒問的主題。**換了話題就是換了話題**，\n"
    "   不要把前一題的主題硬掛上來。\n"
    "3. 繁體中文，不超過 40 字。\n\n"
    "範例（前一題問的是「常見的失誤有哪些」）：\n"
    "「還有嗎」→ 還有哪些常見的失誤？\n"
    "「那 DVT 呢」→ DVT 階段有哪些常見的失誤？\n"
    "「為什麼」→ 為什麼會發生這些常見的失誤？\n"
    "「產出物有哪些」→ 產出物有哪些\n"
    "　（本身就看得懂，且已換到新話題，原封不動）\n"
    "「試產申請要多久」→ 試產申請要多久\n"
    "　（同上，不要補成「失誤相關的試產申請」）"
)


def _recent_turns(history: list[dict] | None) -> list[dict]:
    if not history:
        return []
    return history[-(HISTORY_TURNS * 2):]


def _format_history(history: list[dict], answer_chars: int) -> str:
    lines = []
    for message in history:
        who = "使用者" if message["role"] == "user" else "助理"
        content = message["content"]
        if message["role"] != "user" and len(content) > answer_chars:
            content = content[:answer_chars] + "…（略）"
        lines.append(f"{who}：{content}")
    return "\n".join(lines)


def condense_question(question: str, history: list[dict] | None) -> tuple[str, str]:
    """把追問改寫成可獨立檢索的問題。回傳 (檢索用問題, 錯誤訊息)。

    **這是讓追問能運作的關鍵。** 「還有嗎」「那 DVT 呢」這類句子單獨拿去
    向量檢索必然撈不到東西——它們的語意幾乎全在前文裡。
    改寫後才有東西可以比對。

    任何失敗都退回原問題：追問能不能運作是加分，
    但不該因為改寫出錯而讓整個問答掛掉。
    """
    turns = _recent_turns(history)
    if not turns:
        return question, ""

    prompt = (
        f"對話紀錄：\n{_format_history(turns, 200)}\n\n"
        f"使用者的追問：{question}\n\n"
        f"改寫後的完整問題："
    )
    rewritten, error = ollama_client.generate(
        prompt, system=CONDENSE_SYSTEM, num_predict=80
    )
    rewritten = rewritten.strip().strip("「」\"'")

    # 模型偶爾會多話或空手而回，這時用原問題比較安全。
    if error or not rewritten or len(rewritten) > 100:
        return question, error
    return rewritten, ""


# 命中切片的前後各補幾段。1 是刻意保守：
# 切片本來就有 80 字重疊，補一段就足以接回被切斷的句子與步驟；
# 補太多會讓真正命中的那一段在上下文裡被稀釋。
NEIGHBOR_WINDOW = 1

# 擴展後總段數上限。以 top_k=6、每段 500 字估算，18 段約 9000 字，
# 對 gemma4 的 131k context 綽綽有餘；設上限是為了擋掉極端情況。
MAX_CONTEXT_CHUNKS = 18

# ── 「用更多脈絡重問」模式 ───────────────────────────────────────
# 擴展到命中所在的**整個結構單元**（同一個麵包屑底下的所有切片），
# 而不是固定的前後一段。適合「答案講到一半」「條列只列出前幾項」這種情況。
#
# 上限是必要的，不是防呆：實測知識庫裡結構單元的中位數只有 2 片 / 498 字，
# 但最大的一個有 **50 片 / 21,233 字**。6 個命中若都落在大單元，
# 一次會塞進五萬字——模型裝得下，但「中間內容被忽略」會讓答案反而變差。
WIDE_UNIT_MAX_CHUNKS = 12    # 單一結構單元最多取幾片（超過就以命中段為中心取窗口）
WIDE_TOTAL_MAX_CHUNKS = 40   # 全部加起來的上限，依距離由近到遠分配


@dataclass
class ContextBlock:
    """餵給模型的一個連續區段，可能對應多個命中切片。

    `segments` 是區段內部的逐段結構：`(檢索順序編號 或 None, 位置, 內容)`。
    **命中段各自保留自己的編號，鄰居段沒有編號。**

    這個結構是必要的，不是為了好看：合併後若整塊只掛一個編號，模型會把
    整塊內容都標成那一個來源——實測就發生過：某個設備型號明明出自
    28.1.2 的切片，卻被標成 28.1.1 的編號。逐段標號才能讓
    引註真的指得回原本那一段。
    """

    file_name: str
    stage_code: str | None
    locator: str
    indices: list[int]   # 這個區段涵蓋的引註編號（沿用檢索順序）
    segments: list[tuple[int | None, str, str]]
    # 排序用的相關性代表值，**越大越相關**（見 `best_score`）。
    # 不用距離：距離的鑑別力太差，30 段候選常只差 0.04。
    rank_score: float


def section_of(locator: str) -> str:
    """從 locator 取出**最外層**的結構單元名稱，當作分組用的「章節」。

    locator 的格式一律是 `檔名 › 麵包屑 #序號`，而麵包屑可能有兩層
    （`12.1 XX 測試 › 12.1.6 Testing Specification`）。
    取最外層那一層——內層通常是 `Testing Specification` 這種每章都有的小節名，
    拿它分組會把同一個測試拆散。

    **完全從 locator 推導，不認得任何文件內容或格式。** 各種來源都適用：

        `12.1 XX 測試`                ← PDF 的編號章節
        `## 可靠度測試項目`            ← Markdown 標題
        `投影片 5`                    ← pptx
        `第 4 頁`                     ← VLM 逐頁判讀
        `第1輪 · 1 · 程序 · …`        ← Excel 轉出的逐筆記錄

    沒有麵包屑的切片回空字串，呼叫端會退回「只依文件分組」，也就是改版前的行為。
    """
    body = locator.rsplit(" #", 1)[0]
    parts = [p for p in body.split(" › ") if p.strip()]
    # 第一段是檔名，第二段才是最外層的結構單元
    return parts[1] if len(parts) >= 2 else ""


def _unit_key(locator: str) -> str:
    """從 locator 取出結構單元的識別。

    locator 的格式是 `檔名 › 麵包屑 #序號`，去掉序號就是這一段所屬的結構單元
    （章節／投影片／頁／記錄）。**用現成欄位推導，不必為此重建索引。**
    """
    return locator.rsplit(" #", 1)[0]


def _wide_ranges(chunks: list[RetrievedChunk]) -> dict[int, set[int]]:
    """「更多脈絡」模式：把每個命中擴展到它所屬的整個結構單元。

    兩層上限，缺一不可：
      * 單一單元超過 `WIDE_UNIT_MAX_CHUNKS` 時，以命中段為中心取窗口——
        知識庫裡最大的單元有 50 片 / 2 萬字，整片塞進去只會讓模型迷失。
      * 總量超過 `WIDE_TOTAL_MAX_CHUNKS` 時，**依距離由近到遠分配**，
        最相關的命中優先拿到完整脈絡。
    """
    with get_session() as session:
        rows = session.query(Chunk.doc_id, Chunk.seq, Chunk.locator).all()
    by_unit: dict[tuple[int, str], list[int]] = {}
    for doc_id, seq, locator in rows:
        by_unit.setdefault((doc_id, _unit_key(locator or "")), []).append(seq)

    wanted: dict[int, set[int]] = {}
    budget = WIDE_TOTAL_MAX_CHUNKS
    for c in sorted(chunks, key=lambda x: x.distance):
        unit = by_unit.get((c.doc_id, _unit_key(c.locator or "")), [c.seq])
        # **至少要涵蓋一般模式的前後一段。**
        # 結構單元只有一片時（例如一份短文件整份就是一個單元），
        # 光取單元會比一般模式還少——「更多脈絡」給得比原本少沒有道理。
        seqs = sorted(set(unit) | set(range(max(c.seq - NEIGHBOR_WINDOW, 1),
                                            c.seq + NEIGHBOR_WINDOW + 1)))
        if len(seqs) > WIDE_UNIT_MAX_CHUNKS:
            # 以命中段為中心截取
            pos = seqs.index(c.seq) if c.seq in seqs else 0
            half = WIDE_UNIT_MAX_CHUNKS // 2
            start = max(0, min(pos - half, len(seqs) - WIDE_UNIT_MAX_CHUNKS))
            seqs = seqs[start : start + WIDE_UNIT_MAX_CHUNKS]

        already = wanted.setdefault(c.doc_id, set())
        fresh = [s for s in seqs if s not in already]
        if len(fresh) > budget:
            # 預算不夠時仍要保住命中段本身
            keep = set(fresh[:max(budget, 0)])
            keep.add(c.seq)
            fresh = [s for s in fresh if s in keep]
        already.update(fresh)
        budget -= len(fresh)
    return wanted


def build_context_blocks(chunks: list[RetrievedChunk], wide: bool = False,
                         expand: bool = True) -> list[ContextBlock]:
    """把命中切片擴展成連續區段，相鄰的命中合併成同一塊。

    `wide=True` 時擴展到整個結構單元，供「用更多脈絡重問」使用。

    這是 sentence-window retrieval 的作法：**用小切片做檢索（精準），
    用大範圍餵給模型（完整）**。向量與索引完全不動，只在組提示時多撈幾段。

    解決的問題：500 字的切片常從半個程序中間開始，模型只讀到「步驟 3b…」
    卻看不到這是哪個測試的步驟。

    三個必要的細節：
      1. **連續區段要合併**：命中 #573 與 #574 時，擴展後是一個 572–575 的
         區塊，而不是兩段各自重複。否則模型會看到同一段文字兩次。
      2. **總量有上限**：極端情況下 6 個命中可能擴出很大一片。
      3. **引用仍指向命中段**：擴展只餵給模型，來源卡與引註編號維持指向
         真正命中的那一段，否則追溯性會糊掉——使用者點開來源時，
         看到的必須是「AI 為什麼這樣答」的那一段。
    """
    if not chunks:
        return []

    index_of = {c.chunk_id: i for i, c in enumerate(chunks, start=1)}

    if not expand:
        # 列舉型問題已經撈了很多段，不再補前後文
        wanted = {}
        for c in chunks:
            wanted.setdefault(c.doc_id, set()).add(c.seq)
    elif wide:
        wanted = _wide_ranges(chunks)
    else:
        # 每份文件要涵蓋哪些 seq（命中段 ± 視窗）
        wanted = {}
        for c in chunks:
            wanted.setdefault(c.doc_id, set()).update(
                range(max(c.seq - NEIGHBOR_WINDOW, 1), c.seq + NEIGHBOR_WINDOW + 1)
            )

        # 超出上限就放棄擴展，只用命中段本身——寧可少脈絡，不要爆上下文
        if sum(len(s) for s in wanted.values()) > MAX_CONTEXT_CHUNKS:
            wanted = {}
            for c in chunks:
                wanted.setdefault(c.doc_id, set()).add(c.seq)

    contents: dict[tuple[int, int], str] = {}
    with get_session() as session:
        for doc_id, seqs in wanted.items():
            rows = (
                session.query(Chunk.seq, Chunk.content)
                .filter(Chunk.doc_id == doc_id, Chunk.seq.in_(sorted(seqs)))
                .all()
            )
            for seq, content in rows:
                contents[(doc_id, seq)] = content
    # 命中段一定要在（理論上上面就撈到了，這裡是保險）
    for c in chunks:
        contents.setdefault((c.doc_id, c.seq), c.content)

    by_doc: dict[int, list[RetrievedChunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    blocks: list[ContextBlock] = []
    for doc_id, hits in by_doc.items():
        seqs = sorted(s for (d, s) in contents if d == doc_id)
        # 切成連續的區段
        runs: list[list[int]] = []
        for s in seqs:
            if runs and s == runs[-1][-1] + 1:
                runs[-1].append(s)
            else:
                runs.append([s])

        for run in runs:
            members = [h for h in hits if h.seq in run]
            if not members:      # 純鄰居、沒有命中段的區段不用單獨列出
                continue
            hit_by_seq = {h.seq: h for h in members}
            segments: list[tuple[int | None, str, str]] = []
            for s in run:
                hit = hit_by_seq.get(s)
                if hit:
                    segments.append((index_of[hit.chunk_id], hit.locator, contents[(doc_id, s)]))
                else:
                    segments.append((None, "", contents[(doc_id, s)]))
            blocks.append(
                ContextBlock(
                    file_name=members[0].file_name,
                    stage_code=members[0].stage_code,
                    locator=members[0].locator,
                    indices=sorted(index_of[m.chunk_id] for m in members),
                    segments=segments,
                    rank_score=best_score(members),
                )
            )

    # 依最佳命中的距離排序，最相關的區段排前面
    blocks.sort(key=lambda b: -b.rank_score)
    return blocks


def build_prompt(
    question: str, chunks: list[RetrievedChunk], history: list[dict] | None = None
) -> str:
    """組提示。**依文件分組，並明確禁止跨文件合併。**

    平鋪呈現時模型會把不同文件的內容融成一份看似完整的規格——實測就發生過：
    一份公司內部測試計畫與一份 MIL-STD 問答紀錄被合成同一組參數，
    而那組參數在任何一份文件裡都不存在。分組 + 明確指示是最低成本的防線。

    **編號沿用檢索順序，不因分組而改變**：來源面板與引註 `[n]` 是靠這個
    順序對應的（`chat_service.resolve_sources` 以 chunk_ids 的順序編號），
    重新編號會讓引註指到錯的來源卡。
    """
    blocks = build_context_blocks(chunks)

    grouped: dict[str, list[ContextBlock]] = {}
    for block in blocks:
        grouped.setdefault(block.file_name, []).append(block)

    sections = []
    for file_name, items in grouped.items():
        stage = items[0].stage_code
        header = f"### 文件：{file_name}" + (f"（{stage} 階段）" if stage else "")
        body = []
        for b in items:
            for idx, locator, text in b.segments:
                if idx is not None:
                    body.append(f"[{idx}] {locator}\n{text}")
                else:
                    body.append(f"（前後文，僅供理解，不可標註為來源）\n{text}")
        sections.append(header + "\n\n" + "\n\n".join(body))
    context = "\n\n---\n\n".join(sections)

    multi_doc = ""
    if len(grouped) > 1:
        multi_doc = (
            f"\n\n**注意：以上參考資料來自 {len(grouped)} 份不同文件。**\n"
            "不同文件可能描述不同的標準、版本或適用範圍，"
            "**不可把不同文件的數值或條件合併成同一組結論**。\n"
            "若各文件說法不一致，請分別說明各自出自哪份文件；"
            "若某份文件與問題無關，忽略它即可，不必勉強使用。"
        )

    turns = _recent_turns(history)
    context_block = ""
    if turns:
        context_block = (
            f"先前的對話（僅供理解使用者在問什麼，"
            f"**不可當作事實來源**）：\n{_format_history(turns, HISTORY_ANSWER_CHARS)}\n\n---\n\n"
        )

    return (
        f"{context_block}參考資料：\n\n{context}{multi_doc}\n\n---\n\n"
        f"問題：{question}\n\n請依據上述參考資料回答，並標註來源編號。"
    )


def apply_blocklist(text: str) -> str:
    """把敏感詞替換為遮罩。"""
    raw = get_setting("blocklist", "")
    words = [w.strip() for w in raw.splitlines() if w.strip()]
    for word in words:
        text = text.replace(word, "█" * len(word))
    return text


def answer_stream(
    question: str, chunks: list[RetrievedChunk], history: list[dict] | None = None
):
    """串流生成回答，逐段套用敏感詞遮罩。"""
    prompt = build_prompt(question, chunks, history)
    for piece in ollama_client.generate_stream(prompt, system=SYSTEM_INSTRUCTION):
        yield apply_blocklist(piece)


def suggest_questions(stage_code: str | None) -> list[str]:
    """依當前階段提供預設問題。"""
    generic = [
        "這個階段需要完成哪些產出物？",
        "有哪些常見的失誤或 Lesson Learnt？",
        "需要哪些部門協同作業？",
    ]
    by_stage = {
        "Concept": ["概念階段的市場評估要做到什麼程度？", "誰負責核准進入下一階段？"],
        "Plan": ["專案計畫書要包含哪些章節？", "資源規劃有什麼注意事項？"],
        "EVT": ["EVT 階段需要哪些測試報告？", "EVT 常見的設計問題有哪些？"],
        "DVT": ["DVT 階段的散熱驗證重點是什麼？", "DVT 要準備幾台機器？"],
        "PVT": ["試產申請的流程為何？", "PVT 階段的良率標準是多少？"],
        "MP": ["量產移交需要哪些文件？", "量產後的問題追蹤流程是什麼？"],
    }
    return (by_stage.get(stage_code or "", []) + generic)[:5]


def keyword_search(keyword: str, stage_code: str | None = None, limit: int = 30) -> pd.DataFrame:
    """關鍵字搜尋。Ollama 未啟動時的備援查詢路徑。"""
    if not keyword.strip():
        return pd.DataFrame()

    with get_session() as session:
        query = (
            session.query(Chunk, Document)
            .join(Document, Chunk.doc_id == Document.id)
            .filter(Chunk.content.like(f"%{keyword.strip()}%"))
        )
        if stage_code:
            query = query.filter(Document.stage_code == stage_code)
        rows = query.limit(limit).all()

    return pd.DataFrame(
        [
            {
                "文件": doc.file_name,
                "階段": doc.stage_code or "-",
                "位置": chunk.locator,
                "內容片段": chunk.content[:200] + ("..." if len(chunk.content) > 200 else ""),
            }
            for chunk, doc in rows
        ]
    )
