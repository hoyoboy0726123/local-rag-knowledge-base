"""建立範例資料。

產生一個模擬的知識庫資料夾，以 NUC 六大階段為子目錄，
內含 SOP、Checklist、Lesson Learnt、會議記錄等文件，
用來驗證階段自動歸類、多格式解析與語意檢索。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from database import get_session, init_db, set_setting
from models import ROLE_ADMIN, ROLE_USER, STAGES, Stage, User
from services.auth_service import hash_password, new_salt

DEMO_PASSWORD = "demo1234"
KB_DIR = Path(__file__).parent / "sample_knowledge_base"

USERS = [
    ("admin", "知識庫管理員", ROLE_ADMIN),
    ("user01", "一般使用者 A", ROLE_USER),
    ("user02", "一般使用者 B", ROLE_USER),
]

STAGE_INFO = {
    "Concept": (
        "產品概念形成與可行性評估。此階段結束時需通過 Concept Gate 審查。",
        ["市場需求書 (MRD)", "初步規格草案", "競品分析報告", "可行性評估", "初步成本試算"],
    ),
    "Plan": (
        "專案正式立項，完成資源與時程規劃。",
        ["產品需求書 (PRD)", "專案計畫書", "資源配置表", "風險評估表", "Key Part 選型清單"],
    ),
    "EVT": (
        "工程驗證階段。驗證設計是否可行，重點在功能完整性。",
        ["EVT 測試計畫", "功能測試報告", "電性量測報告", "問題追蹤清單", "EVT 檢討會議記錄"],
    ),
    "DVT": (
        "設計驗證階段。驗證設計是否符合規格，重點在可靠度與量產性。",
        ["DVT 測試計畫", "可靠度測試報告", "散熱驗證報告", "EMI/EMC 報告", "DFM 檢討報告"],
    ),
    "PVT": (
        "量產驗證階段。以量產製程與治具進行小批量試產。",
        ["試產申請單", "試產良率報告", "產線 SOP", "治具驗收紀錄", "PVT 檢討會議記錄"],
    ),
    "MP": (
        "量產階段。正式移交製造並持續追蹤品質。",
        ["量產移交檢查表", "量產品質報告", "客訴處理紀錄", "ECN 變更紀錄"],
    ),
}

DOCUMENTS = {
    "Concept": [
        ("NUC26A_市場需求書.md", """# NUC26A 市場需求書

## 產品定位
面向中小企業與家庭工作室的迷你桌上型電腦，主打體積小、低噪音、擴充彈性。

## 目標市場
- 中小企業辦公用機：預估年需求 12 萬台
- 家庭工作室與創作者：預估年需求 5 萬台

## 核心規格方向
- CPU：Intel Core Ultra 系列，提供 U5 / U7 / U9 三個級距
- 記憶體：支援雙通道 DDR5，最高 64GB
- 儲存：雙 M.2 插槽，支援 PCIe 4.0
- 體積目標：小於 1.2 公升

## 競品分析
主要競爭對手在散熱設計上採用單風扇方案，於高負載時噪音達 42dB。
本案目標將滿載噪音控制在 35dB 以下，此為主要差異化訴求。

## Concept Gate 審查重點
1. 市場需求量預估是否有足夠依據
2. 目標成本是否可達成
3. 關鍵技術風險是否已識別
"""),
        ("概念階段_可行性評估_Checklist.md", """# 概念階段可行性評估檢查表

## 技術可行性
- [ ] 關鍵零組件是否有穩定供應來源
- [ ] 是否需要新製程或新技術導入
- [ ] 散熱方案是否已有初步模擬結果
- [ ] 是否有專利風險

## 商業可行性
- [ ] 目標售價與 BOM 成本的毛利率是否達標
- [ ] 開發費用回收期是否在可接受範圍
- [ ] 通路與客戶是否已有初步承諾

## 資源可行性
- [ ] RD 人力是否足夠
- [ ] 是否與其他專案有資源衝突
- [ ] 實驗室與測試設備檔期是否可安排

## 常見疏漏
過去專案在此階段最常遺漏「認證時程評估」。
各國安規與無線認證動輒需要 8 至 12 週，若到 DVT 才開始安排會直接影響上市時程。
"""),
    ],
    "Plan": [
        ("NUC26A_專案計畫書.md", """# NUC26A 專案計畫書

## 專案里程碑
| 階段 | 預計完成日 | 主要交付 |
|------|-----------|---------|
| Concept Gate | 2026-03-15 | MRD 核准 |
| Plan Gate | 2026-04-30 | PRD 與專案計畫書核准 |
| EVT | 2026-07-15 | 工程樣機與測試報告 |
| DVT | 2026-09-30 | 設計驗證完成 |
| PVT | 2026-11-15 | 試產良率達標 |
| MP | 2026-12-20 | 量產移交 |

## 資源配置
- 硬體工程師 3 名
- 韌體工程師 2 名
- 機構工程師 2 名
- 測試工程師 2 名

## 風險評估
| 風險 | 影響 | 對策 |
|------|------|------|
| CPU 供貨不穩 | 高 | 提早鎖定配額，並規劃 2nd source |
| 散熱設計不達標 | 高 | Plan 階段即進行熱模擬，不等到 EVT |
| 認證時程延誤 | 中 | Plan 階段就與認證實驗室預約檔期 |

## 跨部門協同窗口
- 採購：負責 Key Part 議價與交期追蹤
- 品保：負責測試標準制定與驗收
- 製造：負責 DFM 檢討與產線規劃
- 業務：負責客戶需求確認與樣機安排
"""),
        ("Key_Part選型作業辦法.md", """# Key Part 選型作業辦法

## 適用範圍
本辦法適用於 CPU、記憶體、儲存裝置、面板、電源模組等關鍵零組件的選型作業。

## 選型原則
1. 每個 Key Part 原則上需具備 Primary 與 2nd Source 各一。
2. 單一供應商佔比不得超過該料件需求量的 70%。
3. 選型時需同時評估：規格符合度、供貨穩定度、價格競爭力、技術支援能力。

## 作業流程
1. RD 提出規格需求 → 2. 採購提供候選廠商 → 3. 聯合評估與打樣
4. → 5. 樣品測試 → 6. 選型決議 → 7. 納入 AVL

## 注意事項
2nd Source 的驗證常被壓縮。實務上建議在 EVT 階段就將 2nd Source 納入驗證機台，
若延到 DVT 才開始，一旦發現相容性問題將沒有時間更換。
"""),
    ],
    "EVT": [
        ("EVT階段作業規範.md", """# EVT 階段作業規範

## 階段目標
驗證設計的功能完整性，確認各項規格可達成。

## 必要產出物
1. EVT 測試計畫（需於階段開始前一週完成核准）
2. 功能測試報告
3. 電性量測報告
4. 問題追蹤清單（Issue List）
5. EVT 檢討會議記錄

## 測試項目
### 基本功能測試
- 開關機測試 500 次
- 各介面功能驗證（USB、HDMI、網路、音訊）
- BIOS 設定項目完整性

### 電性測試
- 電源時序量測
- 各電壓軌漣波量測
- 待機功耗與滿載功耗

### 初步熱測試
- 室溫滿載溫度分布
- 風扇轉速與噪音初測

## 進入下一階段的條件
- 所有 P1 等級問題已關閉
- P2 等級問題已有明確對策與時程
- 功能測試通過率達 95% 以上
"""),
        ("EVT_LessonLearnt_歷年彙整.md", """# EVT 階段 Lesson Learnt 歷年彙整

## 2024 年 A 專案：USB 相容性問題
**問題**：EVT 階段僅測試自家配件，量產後大量客訴外接硬碟無法識別。
**根因**：USB 訊號完整性在長線材下衰減過大，EVT 未涵蓋此情境。
**改善**：EVT 測試計畫加入「第三方配件相容性測試」，至少涵蓋 20 種市售配件。

## 2024 年 B 專案：BIOS 版本管理混亂
**問題**：不同工程師手上的樣機 BIOS 版本不一，測試結果無法比對。
**根因**：缺乏統一的版本發布與紀錄機制。
**改善**：建立 BIOS 版本發布單，每次更版需記錄版本號、變更內容、適用機種。

## 2025 年 C 專案：電源時序不符規格
**問題**：EVT 量測發現電源上電時序不符 Intel 規範，導致偶發開機失敗。
**根因**：電源 IC 選型時未詳讀時序要求。
**改善**：Key Part 選型階段就需確認時序規格，並在 EVT 第一週優先量測。

## 2025 年 D 專案：2nd Source 記憶體相容性
**問題**：2nd Source 記憶體在特定 CPU 組合下無法開機。
**根因**：EVT 僅驗證 Primary Source，2nd Source 延到 DVT 才測。
**改善**：EVT 機台配置需納入 2nd Source，且與 Primary 使用相同測試項目。
"""),
    ],
    "DVT": [
        ("DVT階段作業規範.md", """# DVT 階段作業規範

## 階段目標
驗證設計符合規格且具備量產可行性，重點在可靠度與製造性。

## 必要產出物
1. DVT 測試計畫
2. 可靠度測試報告
3. 散熱驗證報告
4. EMI/EMC 測試報告
5. DFM 檢討報告

## 可靠度測試項目
- 高低溫循環測試：-20°C 至 60°C，50 循環
- 高溫高濕測試：40°C / 90%RH，168 小時
- 振動測試：依 IEC 60068-2-6
- 跌落測試：依包裝運輸情境

## 散熱驗證
- 各環境溫度下的滿載溫度分布（25°C / 35°C / 40°C）
- CPU 與 SSD 溫度不得超過各自的 Tjmax 減 10°C
- 滿載噪音需低於 35dB(A)，量測距離 1 公尺

## EMI/EMC
需於認證實驗室進行預測試，預留至少兩輪修改時間。
""" ),
        ("DVT_散熱問題_LessonLearnt.md", """# DVT 階段散熱相關 Lesson Learnt

## 2023 年 E 專案：高溫環境降頻
**問題**：DVT 在 25°C 環境測試通過，客戶在 35°C 機房實際使用時大幅降頻。
**根因**：測試環境僅涵蓋室溫，未模擬實際部署環境。
**改善**：散熱驗證需涵蓋 25°C / 35°C / 40°C 三個環境溫度。

## 2024 年 A 專案：SSD 過熱
**問題**：CPU 溫度正常但 SSD 在持續寫入時達 78°C 觸發保護降速。
**根因**：散熱設計只考慮 CPU，M.2 插槽位於風道死角。
**改善**：熱模擬需納入所有發熱源，M.2 需獨立導熱片或調整風道。

## 2024 年 B 專案：導熱膏塗佈不均
**問題**：同批機台溫度差異達 8°C。
**根因**：手工塗佈導熱膏，用量與位置不一致。
**改善**：改用預貼式導熱片，或制定塗佈治具與作業標準。

## 2025 年 C 專案：風扇噪音超標
**問題**：滿載噪音 39dB，超出 35dB 目標。
**根因**：風扇轉速曲線設定過於保守，溫度一升高就全速運轉。
**改善**：重新調校風扇曲線，並在 EVT 階段就先量測噪音而非等到 DVT。

## 散熱驗證的通用建議
熱模擬應在 Plan 階段就開始，不要等到有實體樣機才做。
模擬與實測的落差通常在 3 至 5°C，若設計裕度不足這個落差就會致命。
"""),
    ],
    "PVT": [
        ("試產申請作業流程.md", """# 試產申請作業流程

## 申請時機
DVT 階段所有 P1 問題關閉後，即可提出 PVT 試產申請。
建議於預計試產日前 **四週**提出，以利產線排程與物料備齊。

## 申請流程
1. **RD PM 填寫試產申請單**，內容包含：
   - 試產數量與機種配置
   - 預計試產日期與產線需求
   - 物料到齊狀況
   - 測試項目與驗收標準

2. **製造部門確認產線檔期**（需時約 3 個工作天）

3. **採購確認物料到齊**
   - 所有 Key Part 需於試產日前一週入庫
   - 缺料項目需明確標示替代方案

4. **品保確認測試標準**與治具備妥

5. **主管核准**：專案經理 → 部門主管 → 製造處主管

## 試產驗收標準
- 直通率（FPY）需達 90% 以上
- 所有測試站點的治具與程式需驗收通過
- 產線 SOP 需完成並經作業員實際操作確認

## 常見問題
最常見的延誤原因是「物料未到齊就申請試產」。
建議在申請前先向採購取得書面的物料到齊確認，避免產線空排。
"""),
    ],
    "MP": [
        ("量產移交檢查表.md", """# 量產移交檢查表

## 文件交付
- [ ] 產品規格書最終版
- [ ] BOM 表最終版並已於 ERP 建檔
- [ ] 產線 SOP 與作業指導書
- [ ] 測試程式與治具清單
- [ ] 各項認證證書

## 品質確認
- [ ] PVT 直通率達標且連續三批穩定
- [ ] 所有 P1 / P2 問題已關閉
- [ ] 可靠度測試報告齊備
- [ ] 客戶樣機確認完成

## 供應鏈確認
- [ ] 所有 Key Part 已簽訂正式採購合約
- [ ] 2nd Source 已完成驗證並納入 AVL
- [ ] 安全庫存水位已設定

## 售後準備
- [ ] 維修手冊與零件料號清單
- [ ] 客服 FAQ 與問題排除指引
- [ ] 保固政策確認

## 移交後追蹤
量產後前三個月為重點觀察期，需每週檢視：
- 產線直通率趨勢
- 客訴類型與件數
- 零件異常率
"""),
    ],
}

CROSS_STAGE = [
    ("NUC新產品開發管理辦法_總則.md", """# NUC 新產品開發管理辦法（總則）

## 目的
規範新產品從概念形成到量產移交的完整開發流程，
確保各階段產出物完整、品質可控、跨部門協同順暢。

## 適用範圍
本辦法適用於所有 NUC 系列新產品開發專案。

## 階段劃分
本辦法將開發流程劃分為六個階段，各階段之間設有 Gate 審查：

| 階段 | 代碼 | 主要目標 |
|------|------|---------|
| 概念階段 | Concept | 確認市場需求與技術可行性 |
| 規劃階段 | Plan | 完成專案立項與資源規劃 |
| 工程驗證 | EVT | 驗證設計功能完整性 |
| 設計驗證 | DVT | 驗證規格符合度與可靠度 |
| 量產驗證 | PVT | 驗證量產製程可行性 |
| 量產階段 | MP | 正式量產與品質追蹤 |

## Gate 審查機制
每個階段結束時需通過 Gate 審查方可進入下一階段。
審查會議由專案經理召集，需有下列單位代表出席：
RD、品保、製造、採購、業務。

未通過審查的專案需提出改善計畫並重新審查，
不得以「時程壓力」為由跳過審查。

## 角色與職責
- **RD 專案經理（RD PM）**：專案整體推動、跨部門協調、階段產出物把關
- **硬體工程師**：電路設計、零件選型、電性驗證
- **機構工程師**：結構設計、散熱設計、DFM
- **測試工程師**：測試計畫制定與執行
- **採購**：Key Part 議價、交期追蹤、供應商管理
- **品保**：測試標準制定、品質數據分析

## 文件管理
所有階段產出物需存放於指定的知識庫資料夾，
並依階段代碼分類。文件命名需包含專案代碼與文件類型。
"""),
    ("跨部門協同窗口一覽.md", """# 跨部門協同窗口一覽

## 各階段主要協同對象

| 階段 | 主要協同部門 | 協同事項 |
|------|-------------|---------|
| Concept | 業務、行銷 | 市場需求確認、競品資訊 |
| Plan | 採購、品保 | Key Part 選型、測試標準制定 |
| EVT | 品保、實驗室 | 測試執行、問題分析 |
| DVT | 製造、認證單位 | DFM 檢討、認證安排 |
| PVT | 製造、採購 | 產線排程、物料備齊 |
| MP | 製造、客服 | 量產移交、售後支援 |

## 常見協同問題

**問題一：認證單位檔期沒有提早預約**
認證實驗室檔期通常需提前 6 至 8 週預約。
建議在 Plan 階段就先預約 DVT 時段，寧可改期也不要臨時排不進去。

**問題二：採購與 RD 對「交期」的定義不一致**
RD 說的交期通常指「料件到我手上可以開始測試」，
採購說的交期則常是「廠商出貨日」，兩者可能差一到兩週。
建議統一以「入庫日」為準並在需求單上明確標示。

**問題三：品保測試標準與 RD 設計規格脫節**
測試標準應在 Plan 階段就由 RD 與品保共同制定，
避免 DVT 才發現測試條件比設計規格更嚴苛。
"""),
]


def build_knowledge_base() -> int:
    KB_DIR.mkdir(exist_ok=True)
    count = 0

    for stage_code, docs in DOCUMENTS.items():
        stage_dir = KB_DIR / stage_code
        stage_dir.mkdir(exist_ok=True)
        for filename, content in docs:
            (stage_dir / filename).write_text(content, encoding="utf-8")
            count += 1

    common = KB_DIR / "共通文件"
    common.mkdir(exist_ok=True)
    for filename, content in CROSS_STAGE:
        (common / filename).write_text(content, encoding="utf-8")
        count += 1

    return count


def seed_db(knowledge_root: str | None = None) -> None:
    """建立資料表、六大階段定義與展示帳號。**不產生任何文件。**

    `knowledge_root` 留空時不寫入該設定——這是刻意的：
    這套系統本身與領域無關，知識庫要指向哪個資料夾應該由使用者決定。
    預設塞一個範例資料夾，會讓人以為系統只能用在那批文件上。
    """
    init_db()
    if knowledge_root:
        set_setting("knowledge_root", knowledge_root)

    with get_session() as session:
        if session.query(Stage).count() == 0:
            for code, name_zh, seq in STAGES:
                description, deliverables = STAGE_INFO.get(code, ("", []))
                session.add(
                    Stage(
                        code=code, name_zh=name_zh, seq=seq,
                        description=description,
                        deliverables=json.dumps(deliverables, ensure_ascii=False),
                    )
                )

        if session.query(User).count() == 0:
            for username, display_name, role in USERS:
                salt = new_salt()
                session.add(
                    User(
                        username=username,
                        password_hash=hash_password(DEMO_PASSWORD, salt),
                        salt=salt,
                        display_name=display_name,
                        role=role,
                        must_change_pwd=False,
                    )
                )

        session.commit()


def main(with_demo_docs: bool = False) -> None:
    """初始化。

    **預設只建帳號，不產生文件。** 這套系統與領域無關——
    附帶一批 NUC 研發流程的範例文件會讓人以為它只能用在那個場景，
    而且新使用者第一件事通常是指向自己的資料夾，範例只是噪音。
    想看範例再加 `--with-demo-docs`。
    """
    count = build_knowledge_base() if with_demo_docs else 0
    seed_db(str(KB_DIR) if with_demo_docs else None)

    print("[OK] 初始化完成")
    print(f"     帳號密碼統一為：{DEMO_PASSWORD}")
    for username, display_name, role in USERS:
        print(f"     - {username:<8} {display_name} ({role})")
    print()
    if with_demo_docs:
        print(f"     已產生 {count} 份範例文件於：{KB_DIR}")
        print("     下一步：以 admin 登入 → 知識庫維護 → 增量更新，即可建立索引。")
    else:
        print("     下一步：以 admin 登入 → 知識庫維護 → 指定你的文件資料夾")
        print("             → 按「增量更新」建立索引。")
        print("     （想先看範例：python seed_data.py --with-demo-docs）")


if __name__ == "__main__":
    import sys

    main(with_demo_docs="--with-demo-docs" in sys.argv)
