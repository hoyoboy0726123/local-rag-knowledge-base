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

from database import get_setting
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
# 原本有一個 `kbs`，描述寫明「不確定就不要填」。但那是靠模型自律，
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


# 明顯依賴前文、單獨看沒有主題的追問句。
#
# **為什麼是窮舉，不是長度也不是模型判斷：**
#   * 長度不行——「今天午餐吃什麼」只有 7 個字，卻是一個完整的問題。
#   * 模型判斷不行——見下方 `_run_search` 呼叫處的教訓：曾經無差別呼叫
#     `condense_question()`，把離題但完整的問題按前文改寫，結果一個合法的
#     拒答被改成答非所問。
#
# 所以只認「拿掉前文就什麼都不剩」的句型。寧可漏接（模型多半自己會補），
# 也不要誤判——誤判的代價是把使用者問的東西換掉，比沒改寫嚴重得多。
_FOLLOW_UP = re.compile(
    r"^(還有(呢|嗎|其他的?|別的)?"
    r"|其他的?呢?|別的呢?"
    r"|繼續(說)?|再說|再多說[一點些]*"
    r"|更多|多一點|詳細(說明|一點)?[說明]*|[講說]清楚一點|說明白一點"
    r"|然後呢?|接下來呢?|後來呢?"
    # 指示代名詞開頭、以「呢／嗎」收尾——拿掉前文就完全沒有主題的句型。
    # 「那份規範的溫度上限是多少」不會中（沒有以呢／嗎收尾），
    # 「這份文件在說明什麼？」也不會，兩者本來就自帶主題。
    r"|[那這][^，。？?！!\n]{0,12}[呢嗎]"
    r"|(它|他|這個|那個)[^，。？?！!\n]{0,6}[呢嗎]"
    r")[？?。！!、,，\s]*$"
)


def _resolve_follow_up(query: str, history: list[dict] | None) -> str:
    """把「還有嗎」「那 DVT 呢」這種追問補成可以獨立檢索的問題。

    **這是程式層防線，不是提示詞。** 系統指示第 1 條已經要求模型自己補主詞，
    但那條要跟另外二十幾條競爭注意力，實測 8 次有 3 次直接把「還有嗎」原封不動
    送去檢索，撈到 0 段，然後回「知識庫中查無足夠資訊」——使用者看到的是
    知識庫沒東西，實際上只是問錯了。

    **拆成兩個決定，各用適合的工具：**

      A. 這句是不是追問？—— **只用句型，不問模型。**
         曾經在這裡問模型「這句話單獨看得懂嗎」，實測它會把「今天午餐吃什麼」
         判成看不懂，於是按前文改寫，一個合法的拒答變成一整段潤滑油規格。
         這正是這個檔案早就記載過的那個回歸。誤判的代價（把使用者問的東西
         換掉）遠大於漏接，所以這一步一律用結構。列不到的說法就讓它漏。

      B. 確定是追問了，要補成什麼？—— **交給 `condense_question` 改寫。**
         這一步用模型是安全的：A 已經確定這句話離開前文就沒有意義，改寫
         不可能劫持一個完整的問題。而且它明顯比「直接用上一句原話」好：
         「那 DVT 呢」用原話會補成「料件有哪些種類」，DVT 這個字就丟了；
         改寫會得到「DVT 階段有哪些料件種類」。

    改寫失敗、回空、或改完仍是追問句時，退回上一句原話——那是保底，
    不是主要路徑。
    """
    if not history or not _FOLLOW_UP.match(query.strip()):
        return query

    previous = ""
    for record in reversed(history):
        if record.get("role") != "user":
            continue
        candidate = (record.get("content") or "").strip()
        # 上一句自己也是追問就再往前找，否則會補成「還有嗎 還有嗎」
        if candidate and not _FOLLOW_UP.match(candidate):
            previous = candidate
            break
    if not previous:
        return query

    rewritten, error = rag_service.condense_question(query, history)
    rewritten = (rewritten or "").strip()
    # 模型偶爾會把追問原句吐回來、或回一個空字串；那時保底用上一句原話
    if error or not rewritten or _FOLLOW_UP.match(rewritten):
        return previous
    return rewritten
    for record in reversed(history):
        if record.get("role") != "user":
            continue
        previous = (record.get("content") or "").strip()
        # 上一句自己也是追問就再往前找，否則會補成「還有嗎 還有嗎」
        if previous and not _FOLLOW_UP.match(previous):
            return previous
    return query


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


def _run_search(query: str, kbs: str | None, citations: _Citations,
                wide: bool = False) -> tuple[str, int, bool]:
    """執行檢索並排版成模型讀得懂的文字。回傳 (內容, 新增命中數, 是否都給過了)。

    **第三個值不能省。** `hits == 0` 代表兩件完全不同的事：

      * 真的沒有相關內容 —— 換個關鍵詞重試是合理的
      * 有內容，但這一輪前面已經全部給過了 —— 重試毫無意義

    兩者混在一起的代價是實測出來的：追問「還有嗎」會撈回同一批切片，
    於是自動重試機制一輪一輪地重查，六次檢索、106 秒，最後還是回
    「查無足夠資訊」——比不重試更慢也更差。
    """
    chunks, error = rag_service.retrieve(query, kbs)
    if error:
        return f"檢索失敗：{error}", 0, False
    if not chunks:
        return "沒有找到任何內容。請換不同的關鍵詞再查一次。", 0, False

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
                f"**不要再檢索了。** 請直接根據前面已經提供的內容回答；"
                f"如果該列的都已經列出來了，就說明知識庫中沒有更多相關資料。"), 0, True
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
        kb_tag = f"（知識庫：{items[0].kb}）" if items[0].kb else ""

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
            f"### 文件：{file_name}{kb_tag}\n\n" + "\n\n".join(chunks_of_section)
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
                "不要用其他主題的內容拼湊答案。"), 0, False

    tail = "\n\n（若以上內容不足以回答使用者的問題，請換個關鍵詞再查一次。）"
    return f"檢索到 {len(chunks)} 段內容：\n\n{body}{warning}{tail}", len(chunks), False


# ------------------------------------------------------------ 逐章節作答
#
# 列舉型問題（「有哪些」「列出全部」）的失敗模式是：模型讀了 20 段卻只列出
# 5 項。實測調 top_k 完全沒用——6 到 20 段同一個水準，30 段反而最差（一次
# 0/9），因為餵越多它摘要得越狠，而且相關性把關會把一整批多數不相關的段落
# 整個否決。
#
# 這裡改成**不讓模型做彙整**：清單由程式依 `section_of` 分組產生（確定性），
# 模型只負責「描述單一章節的內容」——那是它做得好的事。每次呼叫只餵一個章節，
# 歸屬在結構上不可能出錯。
#
# 這個機制曾經存在、後來被移除，理由是當時它把 30 段一次餵進去，num_ctx
# 撐到 32K、8 GB 顯卡四成的層落到 CPU。現在的版本**每次只餵一個章節**（通常
# 1–3 段），記憶體壓力反而比單次 top_k=6 還小；代價是時間——每個章節一次
# 模型呼叫，8 個章節約 1–3 分鐘。所以預設關閉，由管理員決定要不要用時間換完整。
SECTION_TOP_K = 30          # 只用來撈候選；每次生成只餵一個章節
MAX_SECTIONS = 8            # 超過的章節在結尾明列「未展開」，不靜靜丟掉
MIN_SECTIONS_TO_SPLIT = 3   # 章節太少不值得拆，一次問完更快也沒有混淆可言

_ENUMERATION = re.compile(
    r"有哪些|哪些|列出|列舉|全部|所有|各項|清單|一覽|完整|總共|幾種|幾個"
    r"|\ball\b|\blist\b|\bevery\b", re.IGNORECASE)


def _is_enumeration(question: str) -> bool:
    """這是不是「要把東西列全」的問題。只認字面，不問模型——理由同 _FOLLOW_UP。"""
    return bool(_ENUMERATION.search(question))


SECTION_SYSTEM = (
    "你是知識庫助理。**只依據使用者提供的這一個章節的內容作答。**\n"
    "規則：\n"
    "1. 這段內容全部屬於同一個章節。不要提到、不要推測其他章節的內容。\n"
    "2. 問題問到的項目若本章節沒有提供，明講「本章節未提供」，"
    "**絕對不可以從別處推測或填補數值**。\n"
    "3. 每個結論後面標註來源編號 [n]，沿用內容裡給的編號，不要自己編。\n"
    "4. 回答要短。條列，只講與問題直接相關的內容，不要重述整個章節。\n"
    "5. 繁體中文。不要用 LaTeX。"
)


def _answer_by_section(question: str, kbs: str | None, citations: _Citations):
    """列舉型問題：依章節拆開、一節問一次、程式負責組裝。

    **這保證的是「檢索到的章節都會被列出」，不是「知識庫裡所有的都會被列出」**——
    後者取決於檢索，程式無法保證，所以結尾會誠實說明涵蓋範圍。
    """
    chunks, error = rag_service.retrieve(question, kbs, top_k=SECTION_TOP_K)
    if error:
        yield {"type": "error", "message": f"檢索失敗：{error}"}
        return
    if not chunks:
        yield {"type": "text", "piece": NO_RELEVANT_ANSWER}
        yield {"type": "done", "answer": NO_RELEVANT_ANSWER, "chunk_ids": [], "searches": 1}
        return

    for chunk in chunks:
        citations.number(chunk.chunk_id)
    yield {"type": "search", "query": question, "kbs": kbs,
           "hits": len(chunks), "seen_only": False}

    groups: dict[tuple[str, str], list] = {}
    for chunk in chunks:
        groups.setdefault((chunk.file_name, rag_service.section_of(chunk.locator)), []).append(chunk)
    # 用重排序分數挑章節，不要用向量距離——距離的鑑別力太差，
    # 實測會把真正切題的章節排到第 12 名而選進不相關的。
    ordered = sorted(groups.items(), key=lambda kv: -rag_service.best_score(kv[1]))
    picked, dropped = ordered[:MAX_SECTIONS], ordered[MAX_SECTIONS:]

    header = f"檢索到 **{len(ordered)} 個相關章節**"
    if dropped:
        header += f"（以下展開最相關的 {len(picked)} 個）"
    parts = [header + "：\n\n"]
    yield {"type": "text", "piece": parts[0]}

    for (file_name, section), members in picked:
        title = section or file_name
        block = "\n\n".join(
            f"[{citations.number(c.chunk_id)}] {c.locator}\n{c.content}" for c in members)
        prompt = (f"章節：{title}（出自 {file_name}）\n\n內容：\n{block}\n\n---\n\n"
                  f"問題：{question}\n\n請只根據上面這個章節的內容回答。")
        text, err = ollama_client.generate(prompt, system=SECTION_SYSTEM, num_predict=1200)
        body = rag_service.apply_blocklist(_strip_leak(text).strip()) or "（本章節未取得內容）"
        if err:
            body = f"（此章節處理失敗：{err[:80]}）"
        piece = f"### {title}\n\n{body}\n\n"
        parts.append(piece)
        yield {"type": "text", "piece": piece}

    if dropped:
        names = "、".join(s or f for (f, s), _ in dropped)
        tail = (f"\n> 另有 {len(dropped)} 個相關章節未展開：{names}。"
                f"若需要它們的細節，請針對該章節單獨提問。\n")
        parts.append(tail)
        yield {"type": "text", "piece": tail}

    note = ("\n> 以上涵蓋的是**本次檢索到的章節**。知識庫中可能還有未被檢索到的相關內容，"
            "若懷疑有遺漏，可換關鍵詞再問一次。\n")
    parts.append(note)
    yield {"type": "text", "piece": note}
    yield {"type": "done", "answer": "".join(parts), "chunk_ids": citations.chunk_ids, "searches": 1}


def answer(question: str, history: list[dict] | None = None,
           kbs: str | None = None, wide: bool = False):
    """執行一次問答。逐一 yield 事件字典：

        {"type": "search", "query": str, "kbs": list[str]|None, "hits": int,
         "seen_only": bool}   hits=0 且 seen_only=True 代表「有撈到但前面已給過」
        {"type": "text", "piece": str}
        {"type": "error", "message": str}
        {"type": "done", "answer": str, "chunk_ids": [int], "searches": int}

    `kbs` 是 UI 上勾選的知識庫集合（None＝全部），也是**唯一**的範圍來源——
    縮小檢索範圍是使用者的決定，模型無權代勞（見 SEARCH_TOOL 的說明）。
    """
    citations = _Citations()
    # 逐章節作答：預設關閉，管理員在「模型與設定」打開。只接列舉型問題，
    # 而且要先把追問補回主題（「還有嗎」本身不是列舉句，補完可能才是）。
    if get_setting("section_answer", "0") == "1":
        resolved = _resolve_follow_up(question, history)
        if _is_enumeration(resolved):
            yield from _answer_by_section(resolved, kbs, citations)
            return
    messages = (
        [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        + _history_messages(history)
        + [{"role": "user", "content": question}]
    )

    answer_text = ""
    searches = 0
    nudged = False
    # 整輪只自動補查一次，見下方 `auto_retried` 的說明
    auto_retried = False
    # 連續幾次檢索回報「這些前面都給過了」。到 2 就收掉工具。
    seen_only_streak = 0
    exhausted = False
    # 全程是否曾經撈到與問題相關的內容。沒有的話，不管模型寫了什麼都不採用。
    found_relevant = False
    # 每一輪檢索交給模型的內容，最後用來比對答案是不是從裡面寫出來的。
    searched_text: list[str] = []

    for round_no in range(MAX_SEARCHES + 2):
        # 額度用完的最後一輪不給工具，逼模型用手上的資料作答或誠實說沒有。
        #
        # `exhausted` 是第二道收口：**檢索額度還沒用完，但知識庫已經榨乾了**。
        # 追問「還有嗎」時第一次就撈到全部相關內容，之後每次檢索都回
        # 「前面都已經提供過了」，模型卻會一直換關鍵詞再試。實測一次追問查了
        # 5 次、花 88 秒，最後因為拿不到新東西而回「查無足夠資訊」11 個字——
        # 而它手上明明有第一次撈到的 6 段。
        #
        # 提示詞已經改成「不要再檢索了」，但那又是一條要跟其他規則競爭的指示。
        # 連續兩次都是「給過了」就直接收掉工具，讓它只能作答。
        tools = ([SEARCH_TOOL] if round_no < MAX_SEARCHES and not exhausted
                 else None)

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
                # **這裡不做無差別的依前文改寫。**
                # 曾經在這裡呼叫 condense_question()，結果是：使用者問「今天午餐吃什麼」，
                # 模型本來已經正確拒答，卻因為前文在談 PVT，
                # 改寫把問題變成「PVT 階段的產出物」，查回一堆 PVT 內容，
                # 模型就照著答了——一個合法的拒答被硬生生變成答非所問。
                #
                # 走到這一步代表模型已經不配合，此時再加一層猜測只會放大錯誤。
                #
                # 唯一的例外是 `_resolve_follow_up`：它只認「還有嗎」這種拿掉前文
                # 就什麼都不剩的句型，「今天午餐吃什麼」是完整問句、不在其中，
                # 所以不會重蹈上面那個覆轍。差別在**無差別改寫 vs 窮舉的句型**。
                query = _resolve_follow_up(question, history)
                content, hits, seen_only = _run_search(
                    query, kbs, citations, wide)
                searches += 1
                found_relevant = found_relevant or hits > 0
                if hits:
                    searched_text.append(content)
                yield {"type": "search", "query": query, "kbs": kbs,
                       "hits": hits, "seen_only": seen_only}
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
            # 模型該自己把「還有嗎」補成有主題的 query，但實測 8 次有 3 次沒做。
            # 這一行是程式層的補救，只對純追問句生效，理由見 `_resolve_follow_up`。
            query = _resolve_follow_up(query, history)
            # 檢索範圍只認 UI 上的選擇。模型就算硬塞 kbs 也不採用，
            # 原因見 SEARCH_TOOL 上方的說明。
            content, hits, seen_only = _run_search(
                query, kbs, citations, wide)
            searches += 1
            found_relevant = found_relevant or hits > 0
            if hits:
                searched_text.append(content)
            # `seen_only` 要送到前端。**0 段有兩種意思**，介面上分不出來的話，
            # 使用者看到一排「0 段」只會以為知識庫裡沒東西——實際上多半是
            # 「這些前面已經給過了」，那是正常且合理的結果。
            yield {"type": "search", "query": query, "kbs": kbs,
                   "hits": hits, "seen_only": seen_only}

            seen_only_streak = seen_only_streak + 1 if seen_only else 0
            if seen_only_streak >= 2:
                exhausted = True

            # 真的一段都沒撈到時，**由程式用使用者的原話再查一次**。
            #
            # 為什麼需要：實測 4 次有 3 次，模型檢索 0 段後直接回「知識庫中查無
            # 足夠資訊」——工具回傳已經明講「請換不同的關鍵詞再查一次」、額度
            # 也還剩兩次，它一次都沒用。該由程式保證的事不要寄望模型自律。
            #
            # 三個限制缺一不可，少任何一個都會變成上一版那種災難
            # （六次檢索、106 秒、最後還是回查無資訊，比不重試更慢也更差）：
            #
            #   1. `not seen_only` —— 「有內容但前面給過了」也是 0 段，但那種
            #      情況重試毫無意義，只是把同一批切片再撈一次。追問「還有嗎」
            #      幾乎必然落在這一種。
            #   2. `not auto_retried` —— 整輪只補救一次。原本寫成每一輪都能觸發，
            #      於是 MAX_SEARCHES 形同虛設。
            #   3. 只用使用者的原話當備選，不自行造新詞——猜關鍵詞是模型的工作。
            if (hits == 0 and not seen_only and not auto_retried
                    and searches < MAX_SEARCHES):
                auto_retried = True
                candidate = _resolve_follow_up(question, history).strip()
                if candidate and candidate != query:
                    retry, retry_hits, _ = _run_search(
                        candidate, kbs, citations, wide)
                    searches += 1
                    yield {"type": "search", "query": candidate,
                           "kbs": kbs, "hits": retry_hits,
                           "seen_only": False}
                    if retry_hits:
                        content, hits = retry, retry_hits
                        found_relevant = True
                        searched_text.append(content)

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
