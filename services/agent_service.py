"""以工具調用驅動的問答流程。

與傳統 RAG 的差別在於**誰決定要查什麼**：

    傳統：程式先檢索 → 把結果塞給 LLM → LLM 只能就這批資料作答。
    這裡：LLM 自己決定檢索什麼、看完覺得不夠就換關鍵詞再查，
          真的查不到才說沒有。

這同時解決了三件事：

1. **追問**。「還有嗎」的語意全在前文裡，模型看得到對話紀錄，
   自然會把 query 補成「常見失誤有哪些」再去查，不需要另外做問題改寫。
2. **關鍵詞不對**。使用者的口語措辭常與文件用語對不上
   （「踩過什麼雷」vs「Lesson Learnt」），模型第一次查不到會自己換詞重試。
3. **誠實的沒有**。「查無資訊」是在模型嘗試過數種問法之後才說出口的，
   而不是第一次檢索距離偏高就放棄。

模型不支援工具調用時，`chat.py` 會退回 `rag_service` 的傳統單次檢索路徑。
"""

from __future__ import annotations

import json
import re

from services import ollama_client, rag_service

# 即使帶了 think=False，gemma4 偶爾仍會把推理通道的標記漏進正文，
# 開頭出現 "thought" 加上 "<channel|>" 之類的字樣。
# 出現機率不高但看得很清楚，逐字輸出時會直接秀在使用者面前。
_LEAK_PREFIX = re.compile(r"^\s*(thought)?\s*<channel\|?[^>]*>\s*", re.IGNORECASE)
_LEAK_INLINE = re.compile(r"<\|?/?channel\|?[^>]*>", re.IGNORECASE)

# 先攔住這麼多字元再開始逐字輸出，讓上面的標記有機會被完整看到並清掉。
# 標記約 20 字元，取 48 留餘裕；對觀感的影響是首字晚零點幾秒出現。
_LEAK_GUARD_CHARS = 48


def _strip_leak(text: str) -> str:
    return _LEAK_INLINE.sub("", _LEAK_PREFIX.sub("", text)).lstrip()

# 一個問題最多讓模型查幾次。
#
# 設 3 是實測折衷：多數問題一次就夠，換過一次關鍵詞仍找不到的，
# 再查第三次幾乎不會有新東西，只是讓使用者多等。
# 用完額度後最後一輪會把 tools 拿掉，逼模型用現有資料作答或誠實說沒有。
MAX_SEARCHES = 3

# 這裡曾經有一組「列舉型問題自動放大檢索」的機制（偵測「有哪些」就把 top_k
# 提到 30、額度提到 5 輪），**已經移除**。留下記錄避免有人再加回來：
#
# 它的出發點是對的——實測 top_k=6 對某些列舉題只涵蓋 2/6，top_k=30 才 6/6。
# 但它是在 `num_ctx` 還是 Ollama 預設 4096 的前提下設計的，而作者以為模型
# 有 131k（見當時的註解「30 段約 15,000 字，對 gemma4 的 131k context 沒有
# 壓力」）。實際上 30 段約 6,800 token，一進去就被截掉大半。
#
# 修好 num_ctx 之後，這個機制的代價才浮現：
#   * 30 段的提示詞讓 num_ctx 跳到 16384~32768，8 GB 顯卡有四成的層被丟到
#     CPU，回答慢到不堪用
#   * 多輪檢索會把每一輪的結果累積在對話裡，切片數少於 top_k 時後續幾輪
#     根本是同一批內容
#   * 撈進大量沾邊的段落，答案反而失焦——使用者的原話是「查出來反而沒有重點」
#
# 現在一律用預設 top_k，答案的完整性交給使用者判斷：不夠完整時可以按
# 「用更多脈絡重問」，或換個關鍵詞再問一次。

# 對話紀錄帶幾輪。助理的回答動輒上千字，全帶會把檢索結果擠掉。
HISTORY_TURNS = 3
HISTORY_ANSWER_CHARS = 400

# 這個工具**不提供限定階段的參數**，檢索範圍完全由 UI 上的選擇決定。
#
# 原本有一個 `stage_code`，描述寫明「不確定就不要填」。但那是靠模型自律，
# 而模型不一定會照做：實測 qwen3:8b 三次呼叫全部自行填入 `Concept`，
# 包含問題裡完全沒提到階段的時候。
#
# 後果比看起來嚴重：使用者在 UI 選的是「全部階段」，模型卻把範圍縮到一個
# 資料夾，其他階段的文件**被無聲排除**——畫面上不會有任何提示，使用者只會
# 覺得「知識庫裡明明有卻查不到」。
#
# 縮小範圍本來就該是使用者的決定，不是模型的。真正需要限定階段時，UI 的
# 選單就是那個入口；而問題本身提到階段名稱時，語意檢索也撈得到。
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "在這個知識庫中做語意檢索，"
            "回傳最相關的文件片段。這是取得公司流程資訊的唯一管道。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "檢索用的問題或關鍵詞。必須能獨立理解——"
                        "把對話中省略的主詞補回去，不要直接送出「還有嗎」這種句子。"
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

SYSTEM_INSTRUCTION = (
    "你是這個知識庫的助理。\n\n"
    "**你對知識庫的主題沒有任何內建知識。** 任何與知識庫內容有關的問題，"
    "都必須先呼叫 search_knowledge_base 查詢，"
    "絕對不可以憑自己的印象作答。\n\n"
    "規則：\n"
    "1. 檢索用的 query 要能獨立理解。對話中省略的主詞要補回去——\n"
    "   例如前面在談「常見失誤」，使用者問「還有嗎」，\n"
    "   query 應該是「常見失誤 Lesson Learnt」而不是「還有嗎」。\n"
    "2. 檢索結果不足以回答時，**換個說法再查一次**。\n"
    "   使用者的口語常與文件用語對不上，例如「踩過什麼雷」在文件裡\n"
    "   可能寫成「Lesson Learnt」或「常見疏漏」。也可以把問題拆小再查。\n"
    "2-1. **列舉型問題（「有哪些」「列出全部」）一定要查多次。**\n"
    "   一次檢索只回傳最相關的幾段，涵蓋不了所有項目。要換不同關鍵詞\n"
    "   （同義詞、上位詞、你在結果裡看到的相關名詞）多查幾輪再作答。\n"
    "   若無法確定已窮盡，明講「可能還有其他項目未列出」，不要假裝完整。\n"
    "3. **先前的對話只能用來理解使用者在問什麼，不是事實來源。**\n"
    "   即使上一則回答裡已經有相關內容，這一輪仍然必須重新檢索；\n"
    "   沒有本輪的檢索結果就不准標註任何來源編號。\n"
    "4. 只依據本輪檢索到的內容作答，每個結論後面標註來源編號 [1]、[2]。\n"
    "   來源編號沿用檢索結果給的編號，不要自己重新編、也不要沿用上一輪的編號。\n"
    # 這一條是實測補上的。原本只寫「只依據檢索到的內容作答」，模型仍然會
    # 自行補例子——它不認為「舉例說明」算是新增內容。實測捏造過的有：
    # 替 MEMORY 補上「DDR3、DDR4」、替 SYSTEM COMPONENT 補上「主板、散熱模組」、
    # 憑空生出「Others 其他未歸類」與「HARDWARE」兩個原文沒有的類別。
    # 使用者看不出哪些是原文、哪些是模型加的，所以必須逐項點名禁止。
    "4-0. **不可以補上參考資料裡沒有的內容，「舉例說明」也不行。**\n"
    "   替某個類別自行舉例、把清單補成你認為完整的樣子、加上「其他」這種\n"
    "   收尾項目——全部禁止。資料裡只有名稱就只寫名稱。\n"
    "4-0-1. **參考資料裡若有表格，而那張表就是答案，原樣保留整張表**\n"
    "   （所有欄位都要，包含代號、編號這類看似次要的欄位）。\n"
    "   改寫成條列會掉欄位——實測「大類／敘述」對照表被改寫後代號欄整欄消失，\n"
    "   而代號與名稱的對應正是那張表存在的意義。\n"
    # 「原樣保留」與「逐項標註來源」看起來衝突，模型會二選一：實測它保住了
    # 表格卻整篇不標任何編號，而追溯性是這個系統的核心承諾，不能被犧牲。
    # 表格內不塞編號是對的（會破壞排版），但表格外必須交代出處。
    "4-0-2. 保留表格時，**來源編號寫在表格前的說明句或表格後面那一行**\n"
    "   （例如「下表出自 [3]：」）。不要為了讓表格保持原樣就整段不標來源——\n"
    "   表格裡不塞編號是對的，但表格外一定要交代它出自哪裡。\n"
    "4-1. 檢索結果依文件分組呈現。**不可把不同文件的數值或條件合併成同一組結論**——\n"
    "   不同文件可能是不同的標準、版本或適用範圍。若各文件說法不一致，\n"
    "   分別說明各自出自哪份文件；與問題無關的文件直接忽略，不必勉強使用。\n"
    "4-1-1. **同一份文件內也依章節分組，章節之間同樣不可互相借用數值。**\n"
    "   某個章節底下沒有列出規格數值時，就說那個章節未提供，\n"
    "   **不可以拿隔壁章節的數值填上去**。條件、規格、判定標準都適用這一條。\n"
    "4-2. **編號要標在該內容真正出處的那一段上。** 每個段落都有自己的編號，\n"
    "   若某個數值出自標了 [6] 的段落，就標 [6]，不要因為它在 [2] 附近就標成 [2]。\n"
    "5. 換過幾種問法仍然查不到時，明確說「知識庫中查無足夠資訊」，"
    "不要用自己的知識補足。\n"
    "6. 與知識庫主題明顯無關的問題（閒聊、寫作、投資…）直接婉拒，不需要檢索。\n"
    "7. 以繁體中文回答，條列說明，語氣專業簡潔。\n"
    "8. **不要使用 LaTeX 數學語法。** 溫度直接寫 25°C，不要寫 $25^\\circ\\text{C}$；"
    "專有名詞直接寫 Tjmax，不要寫 $\\text{Tjmax}$。這些內容是規格與數值，不是數學式。"
)

# 檢索結果與問題無關時的統一回覆。
NO_RELEVANT_ANSWER = "知識庫中查無足夠資訊。"

# 相關性把關的判斷提示。**只做一件事、只回一個詞**，不跟其他規則競爭。
RELEVANCE_SYSTEM = (
    "你只做一件事：判斷一段答案的**主要內容**是不是來自提供的資料。\n"
    "只回 YES 或 NO 一個詞，不要解釋、不要標點。\n"
    "答案的主要結論在資料裡找得到 → YES。\n"
    "答案講的是資料裡完全沒有的主題（就算它聽起來很合理）→ NO。\n"
    "資料在講別的主題 → NO，即使兩者都是某種流程或制度。\n"
    # 這一條是實測補的。原本要求逐字有據，結果「OEM 跟 ODM 差在哪」被擋掉——
    # 實質內容（由 ASUS 設計、代工廠組裝）完全正確且有來源，只因為模型順手
    # 把縮寫展開成 Original Equipment Manufacturer 就整段作廢。
    # 抓對了，但處置不合比例：使用者因此拿不到本來答得出來的資訊。
    "只有零星的補充說明（例如展開一個縮寫、翻譯一個名詞）不在資料裡，\n"
    "但主要結論有據 → 仍然回 YES。"
)

# 檢索當下的把關用這個（還沒有答案可以比對，只能看資料與問題對不對題）。
RELEVANCE_ONLY_SYSTEM = (
    "你只做一件事：判斷提供的資料裡，有沒有能回答使用者問題的內容。\n"
    "只回 YES 或 NO 一個詞，不要解釋、不要標點。\n"
    "資料在講別的主題就回 NO，即使它看起來也是某種流程或制度。\n"
    "資料有直接回答到問題才回 YES。"
)

# SAMPLING_NOTE ---------------------------------------------------------
# 這裡曾經只取每段的開頭送去判斷，想省下把關的時間。**實測相反，取樣更慢。**
#
#              通過   把關耗時   總計
#   取樣 700 字  7/7    27.4s    93.2s
#   完整內容     7/7    10.7s    58.1s
#
# 原因幾乎確定是提示快取：完整的檢索內容本來就在對話裡（作答要用），把關再
# 送一次同樣的文字就命中快取；取樣則是把內容切碎重組成一份全新的輸入，
# 對快取而言沒見過，得整份重新 prefill。為了省字反而多花時間。
#
# 取樣還踩過準確度的坑：240 字時「料件查詢要給什麼條件」被誤擋，因為答案藏在
# 表格下方的一行小字，只看開頭根本看不到。
#
# 結論：直接送完整內容，又快又準，程式也少一層。
# -----------------------------------------------------------------------


def _is_grounded(question: str, answer: str, content: str) -> bool:
    """答案是不是從檢索到的資料寫出來的。**程式層的防線，不依賴模型自律。**

    判斷對象是**答案**而不是檢索內容，因為答案才是使用者會看到的東西；
    但光看答案抓不出編造——那段憑空生出的五步驟請假流程單獨讀完全合理，
    沒有任何內在破綻。編造唯一的破綻是「它講的東西在來源裡找不到」，
    所以必須拿答案去跟資料比對。

    資料送完整內容而不是摘要，理由見 SAMPLING_NOTE——取樣反而更慢。
    """
    if not content.strip() or not answer.strip():
        return False
    verdict, error = ollama_client.generate(
        f"問題：{question}\n\n知識庫資料：\n{content}\n\n"
        f"這是助理寫的答案：\n{answer[:1500]}\n\n"
        f"這個答案是根據上面的資料寫的嗎？只回 YES 或 NO。",
        system=RELEVANCE_SYSTEM, num_predict=8,
    )
    if error:
        return True
    return "NO" not in verdict.strip().upper()[:6]


def _is_relevant(question: str, content: str) -> bool:
    """檢索到的內容有沒有回答到問題。**這是程式層的防線，不依賴模型自律。**

    為什麼需要它：系統指示第 5 條已經要求「查不到就說查無資訊」，但那條規則
    要和另外二十幾條競爭注意力，實測會被忽略。同一個問題「請假流程是什麼」——
    知識庫裡完全沒有請假相關內容——在**帶三輪對話脈絡**時，qwen3:8b 連續三次
    編出完整的五步驟請假流程並附上假的來源編號 [7]。無脈絡時則正確拒答。

    距離門檻救不了：那題最近的切片距離 1.048，比作者實測會誤擋 43% 合理問題
    的 1.05 門檻還低（見 rag_service 的 DISTANCE_THRESHOLD 說明）。距離量的是
    「語意像不像」，而這裡要判斷的是「有沒有回答到」，是兩件事。

    改成單獨問一次的好處是這個判斷沒有其他指令競爭，輸出只有一個詞。
    判斷失敗（模型出錯、回了別的東西）時**視為相關**——寧可讓內容通過再由
    後續規則把關，也不要因為判斷器故障就讓整個系統拒絕回答。
    """
    if not content.strip():
        return False
    verdict, error = ollama_client.generate(
        f"問題：{question}\n\n資料：\n{content}\n\n"
        f"這些資料裡有沒有能回答上述問題的內容？只回 YES 或 NO。",
        system=RELEVANCE_ONLY_SYSTEM, num_predict=8,
    )
    if error:
        return True
    return "NO" not in verdict.strip().upper()[:6]


FORCE_SEARCH_NUDGE = (
    "你還沒有檢索知識庫。請先呼叫 search_knowledge_base 查詢，"
    "再依照檢索結果回答；若查完確實沒有相關內容，再說查無資訊。"
)


def _history_messages(history: list[dict] | None) -> list[dict]:
    if not history:
        return []
    messages = []
    for record in history[-(HISTORY_TURNS * 2):]:
        content = record["content"]
        if record["role"] != "user" and len(content) > HISTORY_ANSWER_CHARS:
            content = content[:HISTORY_ANSWER_CHARS] + "…（略）"
        messages.append({"role": record["role"], "content": content})
    return messages


class _Citations:
    """跨多次檢索維持穩定的來源編號。

    同一個切片在第一次與第三次檢索都出現時必須是同一個編號，
    否則模型標的 [2] 會指向不同的東西。
    """

    def __init__(self) -> None:
        self._index: dict[int, int] = {}

    def number(self, chunk_id: int) -> int:
        if chunk_id not in self._index:
            self._index[chunk_id] = len(self._index) + 1
        return self._index[chunk_id]

    def seen(self, chunk_id: int) -> bool:
        """這個切片是否已經在先前的輪次給過模型了。"""
        return chunk_id in self._index

    @property
    def chunk_ids(self) -> list[int]:
        return sorted(self._index, key=self._index.get)


def _run_search(query: str, stage_code: str | None, citations: _Citations,
                wide: bool = False) -> tuple[str, int]:
    """執行檢索並排版成模型讀得懂的文字。回傳 (內容, 命中數)。"""
    chunks, error = rag_service.retrieve(query, stage_code)
    if error:
        return f"檢索失敗：{error}", 0
    if not chunks:
        return "沒有找到任何內容。請換不同的關鍵詞再查一次。", 0

    # **已經給過的切片不再重複餵。**
    #
    # 工具回傳會一輪一輪累積在對話裡，而換關鍵詞再查時撈到的往往是同一批段落
    # ——知識庫的切片數少於 top_k 時更是必然（實測一個 25 片的知識庫，列舉型
    # 問題連查兩次都回傳同樣那 25 片）。重複餵有兩個代價，而且會互相加乘：
    #
    #   1. 上下文翻倍。動態 num_ctx 忠實地跟著放大到 32768，KV cache 撐爆
    #      8 GB 顯卡，模型有四成的層被丟到 CPU，慢到不堪用。
    #   2. 擠掉生成空間。實測答案列到第 8 項就被 done_reason=length 切斷。
    #
    # 模型上面已經看得到那些內容，重述一次沒有任何資訊增益。
    fresh = [c for c in chunks if not citations.seen(c.chunk_id)]
    if not fresh:
        return (f"這次檢索到的 {len(chunks)} 段內容，前面都已經提供過了，沒有新的資料。\n"
                f"請改用**明顯不同**的關鍵詞再查，或直接根據已有的內容作答。"), 0
    chunks = fresh

    # 引註編號一定要在擴展前先配好：`citations.number()` 是跨輪次累積的，
    # 而來源面板是靠這個編號對應回切片的。順序也不能因為分組而改變。
    numbers = {c.chunk_id: citations.number(c.chunk_id) for c in chunks}

    # 命中段前後各補一段，讓模型看得到完整的上下文。
    # 相鄰的命中會合併成同一個區段，不會重複餵。
    context_blocks = rag_service.build_context_blocks(chunks, wide=wide)

    # 依文件分組。**平鋪呈現時模型會把不同文件的內容融成一份看似完整的規格**——
    # 實測發生過：一份公司內部測試計畫與一份 MIL-STD 問答紀錄被合成同一組參數，
    # 而那組參數在任何一份文件裡都不存在。
    grouped: dict[str, list] = {}
    for block in context_blocks:
        grouped.setdefault(block.file_name, []).append(block)

    sections = []
    for file_name, items in grouped.items():
        stage = f"（{items[0].stage_code} 階段）" if items[0].stage_code else ""

        # 文件底下再依**章節**分組。
        #
        # 只依文件分組時，同一份大型 PDF 的 6 個章節會平鋪成一串，模型得自己
        # 從 `28.2.6` 這種編號回推「這是哪個測試的規格」——實測它對錯了：
        # 把某章節的規格數值掛到相鄰章節頭上。
        # 答案通順、標了來源，數字卻是錯的。
        #
        # 分組後歸屬由版面決定，不需要模型推論。分組鍵完全從 locator 推導
        # （見 `rag_service.section_of`），對所有格式一致，沒有麵包屑就退回單一組。
        by_section: dict[str, list] = {}
        for block in items:
            by_section.setdefault(rag_service.section_of(block.locator), []).append(block)

        chunks_of_section = []
        for section, blocks in by_section.items():
            body_parts = []
            for block in blocks:
                # **逐段標號，不是整塊掛一個編號。**
                # 整塊只掛一個編號的話，模型會把鄰居段的內容也標成那個來源——
                # 實測發生過：壓力測試機型號出自 28.1.2，卻被標成 28.1.1 的編號。
                # 鄰居段在這裡才配到編號，所以命中段的編號一定比較小——
                # 上面的 `numbers` 已經先照檢索順序把命中段編完了。
                for chunk_id, locator, text in block.segments:
                    body_parts.append(
                        f"[{citations.number(chunk_id)}] {locator}\n{text}")
            head = f"#### 章節：{section}\n（以下內容全部屬於這個章節，不可用於其他章節）\n\n" if section else ""
            chunks_of_section.append(head + "\n\n".join(body_parts))

        sections.append(
            f"### 文件：{file_name}{stage}\n\n" + "\n\n".join(chunks_of_section)
        )
    body = "\n\n---\n\n".join(sections)

    warning = ""
    if len(grouped) > 1:
        warning = (
            f"\n\n**以上內容來自 {len(grouped)} 份不同文件。**"
            "不同文件可能是不同的標準、版本或適用範圍，"
            "**不可把不同文件的數值或條件合併成同一組結論**。"
            "若說法不一致，分別說明各自的出處；與問題無關的文件直接忽略即可。"
        )

    # 相關性把關。撈到東西不等於撈到答案——語意檢索一定會回傳最接近的幾段，
    # 問「請假流程」也會撈到一批作業流程，距離甚至很近（實測 1.048）。
    # 這裡先確認它們真的回答得到問題，不相關就不要把內容交出去，
    # 免得模型拿別的主題拼湊出一個看起來合理的答案。
    if not _is_relevant(query, body):
        return ("檢索到的內容與這個問題無關，知識庫裡沒有這方面的資料。\n"
                f"請直接回覆「{NO_RELEVANT_ANSWER}」，"
                "不要用其他主題的內容拼湊答案。"), 0

    tail = "\n\n（若以上內容不足以回答使用者的問題，請換個關鍵詞再查一次。）"
    return f"檢索到 {len(chunks)} 段內容：\n\n{body}{warning}{tail}", len(chunks)


def answer(question: str, history: list[dict] | None = None,
           stage_code: str | None = None, wide: bool = False):
    """執行一次問答。逐一 yield 事件字典：

        {"type": "search", "query": str, "stage": str|None, "hits": int}
        {"type": "text", "piece": str}
        {"type": "error", "message": str}
        {"type": "done", "answer": str, "chunk_ids": [int], "searches": int}

    `stage_code` 是 UI 上選定的檢索範圍，也是**唯一**的範圍來源——
    縮小檢索範圍是使用者的決定，模型無權代勞（見 SEARCH_TOOL 的說明）。
    """
    citations = _Citations()
    messages = (
        [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        + _history_messages(history)
        + [{"role": "user", "content": question}]
    )

    answer_text = ""
    searches = 0
    nudged = False
    # 全程是否曾經撈到與問題相關的內容。沒有的話，不管模型寫了什麼都不採用。
    found_relevant = False
    # 每一輪檢索交給模型的內容，最後用來比對答案是不是從裡面寫出來的。
    searched_text: list[str] = []

    for round_no in range(MAX_SEARCHES + 2):
        # 額度用完的最後一輪不給工具，逼模型用手上的資料作答或誠實說沒有。
        tools = [SEARCH_TOOL] if round_no < MAX_SEARCHES else None

        # 一次都還沒查就吐出來的文字不能直接顯示給使用者。
        #
        # 實測過的失敗案例：問完「產出物有哪些」之後追問「還有嗎」，
        # 模型看到前一則回答裡已經有內容，就完全不檢索直接續寫，
        # 還編出 [8] 這種當輪根本不存在的來源編號。
        # 因此先緩衝，確認這一輪真的有檢索過（或已經催過一次）才敢即時串流。
        #
        # 正常路徑不會因此失去逐字輸出：第一輪模型是去呼叫工具的，本來就沒有文字，
        # 真正的答案出現在檢索之後的那一輪。
        #
        # 判斷條件只看「查過沒有」，不看「催過沒有」——催了不一定會照做，
        # 若以催過為準，模型第二次仍不查時，那段沒有依據的文字already 進了畫面。
        #
        # 條件是「有沒有撈到**相關**內容」而不只是「查過沒有」：檢索一定會回傳
        # 東西，但不相關的內容會誘發編造（見 `_is_relevant`）。先緩衝著，
        # 最後若確認全程都沒有相關內容，就整段換成「查無足夠資訊」——
        # 使用者不會看到編造的文字閃過去。
        stream_live = found_relevant

        piece_buffer = ""
        emitted = 0  # 已經送出去幾個字元，用來銜接前置攔截與後續逐字輸出
        calls: list[dict] = []
        for event in ollama_client.chat_stream(messages, tools=tools):
            if event["type"] == "error":
                yield event
                return
            if event["type"] == "text":
                piece_buffer += event["piece"]
                if not stream_live or len(piece_buffer) < _LEAK_GUARD_CHARS:
                    continue
                clean = _strip_leak(piece_buffer)
                if len(clean) > emitted:
                    yield {
                        "type": "text",
                        "piece": rag_service.apply_blocklist(clean[emitted:]),
                    }
                    emitted = len(clean)
            elif event["type"] == "tool_calls":
                calls = event["calls"]

        # 內容短到還沒突破攔截門檻就結束了，補送剩下的。
        if stream_live:
            clean = _strip_leak(piece_buffer)
            if len(clean) > emitted:
                yield {"type": "text", "piece": rag_service.apply_blocklist(clean[emitted:])}
        piece_buffer = _strip_leak(piece_buffer)

        if not calls:
            # 完全沒查就想作答 —— 擋下來。
            #
            # 這在追問時特別容易發生：模型看到前一則回答裡已經有內容，
            # 就直接續寫，還會編出當輪根本不存在的來源編號。
            if searches == 0:
                if not nudged:
                    # 先好好講一次。多數情況這樣就會乖乖去查。
                    nudged = True
                    messages.append({"role": "assistant", "content": piece_buffer})
                    messages.append({"role": "user", "content": FORCE_SEARCH_NUDGE})
                    continue

                # 催過還是不查，就別再指望它了 —— 自己查，把結果塞回去。
                # 「答案必須有依據」是這個系統的存在前提，不能交給模型自律。
                #
                # **這裡刻意用原始問題，不做依前文的改寫。**
                # 曾經在這裡呼叫 condense_question()，結果是：使用者問「今天午餐吃什麼」，
                # 模型本來已經正確拒答，卻因為前文在談 PVT，
                # 改寫把問題變成「PVT 階段的產出物」，查回一堆 PVT 內容，
                # 模型就照著答了——一個合法的拒答被硬生生變成答非所問。
                #
                # 走到這一步代表模型已經不配合，此時再加一層猜測只會放大錯誤。
                # 用原始問題最壞的情況是查不到而回「查無資訊」，方向是安全的。
                content, hits = _run_search(question, stage_code, citations, wide)
                query = question
                searches += 1
                found_relevant = found_relevant or hits > 0
                if hits:
                    searched_text.append(content)
                yield {"type": "search", "query": query, "stage": stage_code,
                       "hits": hits}
                # 檢索結果放在 user 訊息裡，不用 tool 訊息。
                #
                # `tool` 角色的訊息在對話模板中必須跟在帶 tool_calls 的 assistant
                # 訊息後面才會被算進去；這裡是我們自己補查的，沒有對應的 tool_call，
                # 模型會完全看不到內容，然後回你「您尚未提供檢索結果」。
                messages.append({"role": "assistant", "content": piece_buffer})
                messages.append({
                    "role": "user",
                    "content": (
                        f"以下是知識庫的檢索結果：\n\n{content}\n\n"
                        f"請依據以上內容回答我的問題「{question}」並標註來源編號；"
                        f"若內容確實無法回答，就說知識庫中查無足夠資訊。"
                    ),
                })
                continue

            answer_text = rag_service.apply_blocklist(piece_buffer)

            # **程式層的最後防線。**
            #
            # 全程都沒撈到相關內容時，模型寫了什麼都不採用——它此時手上只有
            # 別的主題的資料與對話脈絡，任何答案都是拼湊出來的。實測 qwen3:8b
            # 在這種情況下會編出完整的五步驟請假流程並附上假的來源編號。
            #
            # 這一層刻意不依賴提示詞。系統指示第 5 條早就要求「查不到就說查無
            # 資訊」，但那條規則要跟另外二十幾條競爭，實測會被忽略。
            if not found_relevant:
                answer_text = NO_RELEVANT_ANSWER
                yield {"type": "text", "piece": answer_text}
                break

            # 第二道：資料對題，但答案有沒有真的從資料寫出來。
            #
            # 前一道看的是「資料與問題對不對題」，擋得住問請假撈到採購流程
            # 這種主題不符；擋不住「資料相關，但答案多寫了資料裡沒有的東西」
            # ——實測模型會自行把 OEM 展開成 Original Equipment Manufacturer，
            # 那四個字在知識庫裡一個都沒有。
            #
            # 這一道的判斷對象是**答案**，因為那才是使用者看到的東西；但單看
            # 答案沒有用（編造的內容自身毫無破綻），必須拿去跟來源比對。
            if answer_text.strip() and not _is_grounded(
                    question, answer_text, "\n".join(searched_text)):
                answer_text = NO_RELEVANT_ANSWER
                yield {"type": "text", "piece": answer_text}
                break

            # 模型有可能什麼都不回就結束（實測 gemma4:e2b 在追問時常這樣），
            # 直接存下去會在對話紀錄裡留一則空白訊息，使用者也不知道發生什麼事。
            if not answer_text.strip():
                answer_text = (
                    "（模型未產生回覆）已完成檢索但沒有得到答案，請換個說法再問一次。"
                    "若持續發生，請管理員到「模型設定」確認目前的生成模型是否適用。"
                )
                yield {"type": "text", "piece": answer_text}
            elif not stream_live:  # 緩衝下來的內容還沒送出去，補上
                yield {"type": "text", "piece": answer_text}
            break

        # 模型偶爾會在呼叫工具前先講一句話，那不是最終答案，不保留。
        messages.append({"role": "assistant", "content": piece_buffer, "tool_calls": calls})

        for call in calls:
            function = call.get("function", {})
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):  # 有些版本會回字串
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            query = (arguments.get("query") or question).strip()
            # 檢索範圍只認 UI 上的選擇。模型就算硬塞 stage_code 也不採用，
            # 原因見 SEARCH_TOOL 上方的說明。
            content, hits = _run_search(query, stage_code, citations, wide)
            searches += 1
            found_relevant = found_relevant or hits > 0
            if hits:
                searched_text.append(content)
            yield {"type": "search", "query": query, "stage": stage_code,
                   "hits": hits}
            messages.append({"role": "tool", "content": content})

    else:
        # 迴圈跑完仍沒有定案（模型一直在呼叫工具）。用最後一輪的文字，
        # 沒有就給一句明確的說明，不要回空白。
        #
        # **把關在這裡也要做一次。** 這條路徑一度沒有經過任何檢查——
        # 實測時發現三題答得出來的問題全部由此收尾，等於防線形同虛設。
        answer_text = rag_service.apply_blocklist(piece_buffer) or (
            "知識庫中查無足夠資訊。已嘗試多種關鍵詞檢索，仍找不到能回答此問題的內容。"
        )
        if not found_relevant or not _is_grounded(
                question, answer_text, "\n".join(searched_text)):
            answer_text = NO_RELEVANT_ANSWER
        yield {"type": "text", "piece": answer_text}

    yield {
        "type": "done",
        "answer": answer_text,
        "chunk_ids": citations.chunk_ids,
        "searches": searches,
    }
