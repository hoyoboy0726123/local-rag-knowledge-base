import { useEffect, useState } from 'react'
import { api, stream } from '../app/api'

/* 模型設定。
 *
 * **選單對不上時絕對不能靜默改設定。**
 * V1 的寫法是「找不到就 index=0」，於是顯示清單第一個模型並判定
 * 「使用者換了模型」而寫回設定——結果 embed_model 被換成生成模型，
 * 之後每次檢索都收到 501 Not Implemented，而且只要打開這個分頁就會發生。
 *
 * 這裡：比對忽略 :latest 後綴；真的找不到就保留原值並顯示警告。 */
const match = (current, options) => {
  if (!current) return null
  if (options.includes(current)) return current
  return options.find((o) => o.split(':')[0] === current.split(':')[0]) || null
}

function Picker({ label, name, value, options, hint, onChange }) {
  const matched = match(value, options)
  const list = matched ? options : [value, ...options].filter(Boolean)
  return (
    <div className="field">
      <label>{label}</label>
      <select className="sel" style={{ width: '100%' }} value={matched || value}
              onChange={(e) => onChange(name, e.target.value)}>
        {list.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
      <div style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 4 }}>{hint}</div>
      {!matched && value && (
        <div className="note amber" style={{ marginTop: 6 }}>
          ⚠️ 目前設定的 <code>{value}</code> 不在已安裝清單中。
          設定維持不變；若要更換請從上方選擇，或先安裝該模型。
        </div>
      )}
    </div>
  )
}

/* 上下文視窗。這不是效能微調，是「檢索結果會不會被安靜丟掉」的關鍵設定：
 * 設太小時超出的部分會被丟棄，而且不會有任何錯誤訊息——使用者只會看到
 * 模型答非所問或把指示複述一遍。 */
const CTX_OPTIONS = [4096, 8192, 16384, 32768]

function ContextField({ value, onChange }) {
  const list = CTX_OPTIONS.includes(value) ? CTX_OPTIONS : [value, ...CTX_OPTIONS]
  return (
    <div className="field">
      <label>上下文視窗下限（num_ctx）</label>
      <select className="sel" style={{ width: '100%' }} value={value}
              onChange={(e) => onChange('num_ctx', Number(e.target.value))}>
        {list.map((o) => <option key={o} value={o}>{o.toLocaleString()} token</option>)}
      </select>
      <div style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 4 }}>
        模型單次能讀進去的最大量（提示詞 + 回答共用）。
        <b>這是下限</b>——檢索結果較多時系統會自動往上調（8K → 16K → 32K），
        避免內容被安靜截斷。調高下限會增加顯示記憶體佔用，
        可能使模型被拆到 CPU 執行而明顯變慢。
      </div>
      {value < 8192 && (
        <div className="note amber" style={{ marginTop: 6 }}>
          ⚠️ 低於 8192 時實測會發生：檢索結果放不下而被安靜丟掉，
          模型收到殘缺的提示，可能改為複述系統指示而不是回答問題。
        </div>
      )}
    </div>
  )
}

/* 每次檢索餵給模型幾段。預設 6，上限 30。
   曾經有「列舉型問題自動放大到 30 段」的機制，實測對某些列舉題 6 段只涵蓋
   2/6、30 段才 6/6——但 30 段約 15,000 字，會把 num_ctx 撐到 16K～32K，
   8 GB 顯卡有四成的層被丟到 CPU。所以改成讓使用者自己決定要完整還是要快，
   警告依數值分級，讓代價在調的當下就看得到。 */
const TOPK_OPTIONS = [4, 6, 8, 10, 12, 16, 20]

function TopKField({ value, onChange }) {
  const list = TOPK_OPTIONS.includes(value) ? TOPK_OPTIONS : [value, ...TOPK_OPTIONS].sort((a, b) => a - b)
  const chars = value * 500
  return (
    <div className="field">
      <label>每次檢索段數（top_k）</label>
      <select className="sel" style={{ width: '100%' }} value={value}
              onChange={(e) => onChange('top_k', Number(e.target.value))}>
        {list.map((o) => <option key={o} value={o}>{o} 段（約 {(o * 500).toLocaleString()} 字）</option>)}
      </select>
      <div style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 4 }}>
        一次檢索交給模型的切片數。<b>調高不保證答案更完整</b>——實測兩題列舉題在
        6～20 段是同一個水準，段數越多模型反而摘要得越狠。想把清單列全，請改開下方的
        「逐章節作答」。預設 6。
      </div>
      {value > 12 && (
        <div className="note amber" style={{ marginTop: 6 }}>
          ⚠️ {value} 段約 {chars.toLocaleString()} 字，上下文視窗會升到 16K 以上；
          <b>8 GB 顯卡實測會有約四成的模型層被移到 CPU 執行，回答時間可能拉長到數倍</b>，
          而且完整度並不會因此變好。除非你的文件切片特別短，否則不建議。
        </div>
      )}
    </div>
  )
}

export default function Models() {
  const [data, setData] = useState(null)
  const [saved, setSaved] = useState('')
  const [rr, setRr] = useState(null)
  const [dl, setDl] = useState(null)   // 下載進度

  const load = () => api.get('/api/admin/models').then(setData)
  const loadRr = () => api.get('/api/admin/reranker').then(setRr)
  useEffect(() => { load(); loadRr() }, [])

  /* 下載重排序模型。571 MB，用 SSE 回報進度。
   * 下載完**不需要重啟**——後端是延遲載入，下一次檢索就會用到。 */
  const downloadRr = async () => {
    setDl({ lines: ['準備中…'], busy: true })
    try {
      await stream('/api/admin/reranker/download', {}, (name, d) => {
        if (name === 'log') setDl((s) => ({ ...s, lines: [...s.lines, d.line] }))
        else if (name === 'done') {
          setDl({ lines: [d.message], busy: false, ok: d.ok })
          loadRr()
        }
      })
    } catch (e) {
      setDl({ lines: [`下載失敗：${e.message}`], busy: false, ok: false })
    }
  }

  const toggleRr = async (enabled) => {
    await api.put('/api/admin/reranker', { enabled })
    loadRr()
  }

  const change = async (key, value) => {
    await api.put('/api/admin/models', { [key]: value })
    setSaved(`已更新 ${key}`); load()
  }


  if (!data) return <div className="empty">載入中…</div>

  return (
    <>
      <header className="top">
        <div><h1>模型與設定</h1><div className="crumb">全部在本機執行，不對外送出任何內容</div></div>
      </header>
      <div className="page" style={{ maxWidth: 680 }}>
        <div className="panel">
          <div className="phead"><b>模型</b><span className="pill">{data.available.length} 個可用</span></div>
          <div style={{ padding: 15 }}>
            <Picker label="Embedding 模型" name="embed_model" value={data.current.embed_model}
                    options={data.available} onChange={change}
                    hint="產生向量用。更換後必須執行全量重建——不同模型的向量空間不相容。" />
            <Picker label="生成模型" name="llm_model" value={data.current.llm_model}
                    options={data.available} onChange={change}
                    hint="回答問題用。建議選擇支援工具調用（tools）的模型。" />
            <Picker label="VLM 模型" name="vlm_model" value={data.current.vlm_model}
                    options={data.available} onChange={change}
                    hint="解析圖片與掃描件用。換模型後請按下方按鈕自檢。" />
            <ContextField value={data.current.num_ctx} onChange={change} />
            <TopKField value={data.current.top_k ?? 6} onChange={change} />
            <div className="field">
              <label>
                <input type="checkbox" checked={!!data.current.section_answer}
                       onChange={(e) => change('section_answer', e.target.checked)} />
                {' '}逐章節作答（列舉型問題）
              </label>
              <div style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 4 }}>
                問「有哪些」「列出全部」這類問題時，改由程式把檢索到的章節列成清單，
                模型一次只描述一個章節，最後由程式組裝。這是目前<b>唯一實測能拉高列舉完整度</b>的方法——
                單次餵再多段，模型都會自己摘要掉一半。
              </div>
              {data.current.section_answer
                ? <div className="note amber" style={{ marginTop: 6 }}>
                    ⚠️ 每個章節一次模型呼叫，最多 8 個章節，<b>一題約 1～3 分鐘</b>
                    （依生成模型速度而定）。記憶體壓力比調高段數小——每次只餵一個章節——
                    但總時間長很多。只影響列舉型問題，一般問答不受影響。
                  </div>
                : <div className="note" style={{ marginTop: 6, fontSize: 11 }}>
                    目前關閉：列舉題走一般流程，快但可能漏項；答案不完整時可再追問「還有嗎」。
                  </div>}
            </div>
            {saved && <div className="note accent">{saved}</div>}

            {/* 重排序是選配：小文件用不到，大型規格書才有明顯差別。
                因此模型不隨專案附帶，需要時在這裡下載。 */}
            {rr && (
              <div style={{ marginTop: 20, paddingTop: 16, borderTop: '1px solid var(--line)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 9, flexWrap: 'wrap' }}>
                  <b style={{ fontSize: 12.5 }}>重排序模型</b>
                  <span className={`tag ${rr.installed ? '' : 'muted'}`}>
                    {rr.installed ? `已安裝 ${rr.size_mb} MB` : '未安裝'}
                  </span>
                  {rr.installed && (
                    <label style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 5 }}>
                      <input type="checkbox" checked={rr.enabled}
                             onChange={(e) => toggleRr(e.target.checked)} />
                      啟用
                    </label>
                  )}
                  {!rr.installed && (
                    <button className="btn" disabled={dl?.busy} onClick={downloadRr}>
                      {dl?.busy ? '下載中…' : '⬇ 下載模型（571 MB）'}
                    </button>
                  )}
                </div>
                <div className="hint" style={{ marginTop: 7 }}>
                  用 cross-encoder 重新排序檢索結果。<b>小型文件不需要</b>——
                  實測小型 Markdown 開不開都是 5/5；但數百頁的規格書差別明顯
                  （壓力測試列舉 5/6 → 6/6、原物料類別 6/7 → 7/7）。
                  下載後立即生效，不必重啟。
                </div>
                {rr.error && <div className="note amber">載入失敗：{rr.error}</div>}
                {dl && (
                  <pre className="log" style={{ marginTop: 9, maxHeight: 150 }}>
                    {dl.lines.slice(-8).join('\n')}
                  </pre>
                )}
              </div>
            )}

            {data.supports_tools
              ? <div className="note accent" style={{ marginTop: 10 }}>
                  ✅ 目前的生成模型支援工具調用，AI 問答以 Agent 模式運作。
                </div>
              : <div className="note danger" style={{ marginTop: 10 }}>
                  ❌ <b>目前的生成模型不支援工具調用，問答品質會明顯下降。</b>
                  建議改用支援工具調用的模型（<code>ollama show &lt;模型&gt;</code> 的
                  Capabilities 要列出 <code>tools</code>）。
                </div>}
          </div>
        </div>

      </div>
    </>
  )
}
