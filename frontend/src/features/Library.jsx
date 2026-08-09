import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api, getToken } from '../app/api'

/* 檢索關鍵字編輯。
 *
 * **切片內容本身不開放修改**，理由有兩個：
 *   1. content 同時是使用者看到的「原文」與 LLM 作答的依據，
 *      改了它，來源卡片就會顯示原始文件裡沒有的文字 → 破壞可追溯性。
 *   2. 向量是索引當下用 content 算的，只改欄位不重算向量，
 *      檢索完全不會變——管理員以為調好了，其實什麼都沒發生。
 *
 * 做成標籤式輸入而非純文字框，因為它本質是一組詞而不是一段話。 */
function Keywords({ chunkId, initial, onSaved }) {
  const [tags, setTags] = useState([])
  const [draft, setDraft] = useState('')
  const [state, setState] = useState(null)

  useEffect(() => {
    setTags(initial ? initial.split(/[\s、,，]+/).filter(Boolean) : [])
    setDraft(''); setState(null)
  }, [chunkId, initial])

  const add = () => {
    const v = draft.trim()
    if (v && !tags.includes(v)) setTags([...tags, v])
    setDraft('')
  }

  const save = async () => {
    setState({ busy: true })
    try {
      await api.put(`/api/admin/chunks/${chunkId}/keywords`, { keywords: tags.join(' ') })
      setState({ ok: '已更新並重算向量' })
      onSaved?.()
    } catch (e) {
      setState({ error: e.message })
    }
  }

  return (
    <div className="kwbox">
      <h5>◎ 檢索關鍵字</h5>
      <p className="exp">
        使用者的口語常與文件用語對不上（「以前踩過什麼雷」vs「Lesson Learnt」）。
        在這裡補同義詞可提高這一段被找到的機率——這些字
        <b>只影響向量比對，不會出現在回答或來源卡片裡</b>。切片內容本身不開放修改。
      </p>
      <div className="kwin" onClick={(e) => e.currentTarget.querySelector('input')?.focus()}>
        {tags.map((t) => (
          <span className="kwtag" key={t}>
            {t}<button onClick={() => setTags(tags.filter((x) => x !== t))}>×</button>
          </span>
        ))}
        <input
          value={draft} placeholder={tags.length ? '' : '輸入後按 Enter 新增…'}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') { e.preventDefault(); add() }
            if (e.key === 'Backspace' && !draft && tags.length) setTags(tags.slice(0, -1))
          }}
          onBlur={add}
        />
      </div>
      <div className="kwfoot">
        <span className="kwnote">
          {state?.ok ? `✓ ${state.ok}` : state?.error ? `⚠️ ${state.error}`
            : '儲存後立即重算此切片向量，重新索引時自動還原'}
        </span>
        <button className="btn pri" onClick={save} disabled={state?.busy}>
          {state?.busy ? '重算中…' : '儲存並重算向量'}
        </button>
      </div>
    </div>
  )
}

/* 切片內容編輯器。
 *
 * **這件事原本是禁止的，開放的前提是三個保證：**
 *   1. 存檔立刻重算向量——否則改了內容檢索完全不變，管理員以為調好了其實沒有
 *   2. 保留 original_content——可追溯性從「內容必等於原文」變成「來歷必查得到」
 *   3. 另存一份（chunk_keywords.edited_content）——否則下次全量重建就消失
 * 少任何一項都不該開放編輯。 */
function ChunkEditor({ detail, onDone }) {
  const [text, setText] = useState(detail.content)
  const [busy, setBusy] = useState(false)
  useEffect(() => { setText(detail.content) }, [detail.id, detail.content])

  const dirty = text !== detail.content
  const diff = detail.content.length - text.length

  const save = async () => {
    setBusy(true)
    try {
      const r = await api.put(`/api/admin/chunks/${detail.id}/content`, { content: text })
      onDone(r.message)
    } catch (e) { onDone(`儲存失敗：${e.message}`) }
    finally { setBusy(false) }
  }

  const revert = async () => {
    if (!window.confirm('還原成檔案原本解析出來的內容？你的編輯會被捨棄。')) return
    setBusy(true)
    try {
      const r = await api.post(`/api/admin/chunks/${detail.id}/revert`)
      onDone(r.message)
    } catch (e) { onDone(`還原失敗：${e.message}`) }
    finally { setBusy(false) }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ fontSize: 11.5, color: 'var(--ink-3)', lineHeight: 1.6 }}>
        刪掉不需要的段落即可——收件人清單、簽名檔、免責聲明、重複的頁首。
        <b>存檔後會立刻重算這一段的向量</b>，檢索結果馬上改變。
      </div>
      <textarea
        value={text} onChange={(e) => setText(e.target.value)} spellCheck={false}
        style={{
          width: '100%', minHeight: 260, resize: 'vertical', padding: '10px 12px',
          border: '1px solid var(--line-strong)', borderRadius: 'var(--r-sm)',
          fontFamily: 'ui-monospace, Menlo, monospace', fontSize: 12.5,
          lineHeight: 1.7, color: 'var(--ink)', background: 'var(--surface)',
        }} />
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 11.5, color: dirty ? 'var(--accent)' : 'var(--ink-3)' }}>
          {text.length} 字{dirty && diff !== 0 && `（${diff > 0 ? '少' : '多'}了 ${Math.abs(diff)} 字）`}
        </span>
        <div style={{ flex: 1 }} />
        {detail.edited && (
          <button className="btn" onClick={revert} disabled={busy}>還原成原文</button>
        )}
        <button className="btn pri" onClick={save} disabled={busy || !dirty || !text.trim()}>
          {busy ? '重算向量中…' : '儲存並重算向量'}
        </button>
      </div>

      {detail.edited && (
        <details style={{ fontSize: 12 }}>
          <summary style={{ cursor: 'pointer', color: 'var(--ink-2)' }}>
            檔案原本解析出來的內容（{detail.original_content.length} 字）
          </summary>
          <pre style={{
            whiteSpace: 'pre-wrap', margin: '8px 0 0', padding: 10, fontSize: 11.5,
            background: 'var(--raised)', border: '1px solid var(--line)',
            borderRadius: 'var(--r-sm)', maxHeight: 220, overflowY: 'auto',
          }}>{detail.original_content}</pre>
        </details>
      )}
    </div>
  )
}

function Chunks() {
  const [list, setList] = useState([])
  const [q, setQ] = useState('')
  const [pickedId, setPickedId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [view, setView] = useState('rendered')
  const [open, setOpen] = useState({})   // 依文件 ID 記展開狀態
  const [msg, setMsg] = useState('')

  const load = () => api.get(`/api/admin/chunks?keyword=${encodeURIComponent(q)}`).then((d) => setList(d.chunks))
  const refresh = () => {
    if (pickedId) api.get(`/api/admin/chunks/${pickedId}`).then(setDetail).catch(() => setDetail(null))
    load()
  }
  useEffect(() => { load() }, [q])
  useEffect(() => {
    if (!pickedId) return setDetail(null)
    api.get(`/api/admin/chunks/${pickedId}`).then(setDetail).catch(() => setDetail(null))
  }, [pickedId])

  /* 依文件分組。一封 .msg 可能切出上百段，平鋪成一長串完全沒辦法找東西。
     用「文件 ID」而不是檔名分組——同名不同階段的檔案會被併成同一組。 */
  const groups = []
  const seen = new Map()
  for (const c of list) {
    const key = c['文件 ID']
    if (!seen.has(key)) {
      const g = { id: key, name: c.文件, stage: c.階段, items: [] }
      seen.set(key, g); groups.push(g)
    }
    seen.get(key).items.push(c)
  }
  // 搜尋時全部展開——不然搜到的東西還藏在收合的群組裡，等於沒搜
  const isOpen = (id) => (q.trim() ? true : open[id] ?? groups.length <= 2)

  const removeChunk = async (c, e) => {
    e.stopPropagation()
    if (!window.confirm(
      `刪除切片 #${c['切片 ID']}（${c.位置}，${c.字數} 字）？\n\n`
      + '它會立刻退出檢索範圍。\n'
      + '注意：切片是從檔案解析出來的，執行「全量重建」時會原樣長回來。')) return
    try {
      const r = await api.del(`/api/admin/chunks/${c['切片 ID']}`)
      setMsg(r.message)
      if (pickedId === c['切片 ID']) setPickedId(null)
    } catch (err) {
      setMsg(`刪除失敗：${err.message}`)
    }
    load()
  }

  return (
    <div className="grid2">
      <div className="panel">
        <div className="phead">
          <b>切片</b>
          <span className="pill">{groups.length} 份文件 · {list.length} 片</span>
        </div>
        <div style={{ padding: '10px 13px' }}>
          <input className="sel" style={{ width: '100%' }} placeholder="搜尋切片內容…"
                 value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        {msg && (
          <div className={`note ${msg.includes('失敗') ? 'danger' : 'accent'}`}
               style={{ margin: '0 13px 10px' }}>{msg}</div>
        )}
        <div style={{ maxHeight: 560, overflowY: 'auto' }}>
          {groups.map((g) => (
            <div key={g.id}>
              <div className="row" style={{ background: 'var(--raised)' }}
                   onClick={() => setOpen({ ...open, [g.id]: !isOpen(g.id) })}>
                <span className="caret">{isOpen(g.id) ? '▾' : '▸'}</span>
                <div className="rinfo">
                  <b title={g.name}>{g.name}</b>
                  <span>{g.items.length} 個切片</span>
                </div>
                {g.stage !== '未分類' && <span className="tag">{g.stage}</span>}
              </div>

              {isOpen(g.id) && g.items.map((c) => (
                <div key={c['切片 ID']}
                     className={`row ${pickedId === c['切片 ID'] ? 'on' : ''}`}
                     style={{ paddingLeft: 26 }}
                     onClick={() => setPickedId(c['切片 ID'])}>
                  <div className="ft">#{c['切片 ID']}</div>
                  <div className="rinfo">
                    <b style={{ fontWeight: 500 }}>{c.位置}</b>
                    <span>{c.字數} 字 · {String(c.內容).slice(0, 28)}…</span>
                  </div>
                  <button className="x" title="刪除這個切片"
                          onClick={(e) => removeChunk(c, e)}>×</button>
                </div>
              ))}
            </div>
          ))}
          {!list.length && <div className="empty">找不到符合的切片</div>}
        </div>
      </div>

      <div>
        {!detail ? (
          <div className="panel"><div className="empty">從左側選一個切片查看內容</div></div>
        ) : (
          <div className="panel">
            <div className="phead">
              <div>
                <b>{detail.file_name}</b>
                <div className="crumb">
                  {detail.locator} · {detail.char_count} 字 · 階段 {detail.stage_code || '未分類'}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <div className="seg">
                  <button className={view === 'raw' ? 'on' : ''} onClick={() => setView('raw')}>原始文字</button>
                  <button className={view === 'rendered' ? 'on' : ''} onClick={() => setView('rendered')}>渲染後</button>
                  <button className={view === 'edit' ? 'on' : ''} onClick={() => setView('edit')}>✎ 編輯</button>
                </div>
                <button className="btn" onClick={(e) => removeChunk(
                  { '切片 ID': detail.id, 位置: detail.locator, 字數: detail.char_count }, e)}>
                  刪除此切片
                </button>
              </div>
            </div>

            {detail.edited && (
              <div className="note accent" style={{ margin: '0 15px 10px' }}>
                <b>✎ 這段內容已被編輯過</b>，與檔案原本解析出來的結果不同
                （原文 {detail.original_content.length} 字 → 現在 {detail.char_count} 字）。
                下方可切換到「編輯」比對與還原。
              </div>
            )}

            <div className="docbody">
              {view === 'edit' ? (
                <ChunkEditor detail={detail} onDone={(m) => { setMsg(m); refresh() }} />
              ) : view === 'raw' ? (
                <pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{detail.content}</pre>
              ) : (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{detail.content}</ReactMarkdown>
              )}
            </div>

            <Keywords chunkId={detail.id} initial={detail.keywords}
                      onSaved={() => api.get(`/api/admin/chunks/${detail.id}`).then(setDetail)} />

            <div className="note amber" style={{ margin: '0 15px 15px' }}>
              <b>這一段是 AI 實際讀到的文字。</b>
              若原始文件有某段內容但這裡搜不到，代表是解析階段掉的，不是檢索問題。
              <div style={{ marginTop: 6 }}>
                刪除切片會讓它立刻退出檢索範圍——適合清掉郵件的收件人清單、簽名檔這類佔名額又答不出東西的噪音。
                <b>但切片是從檔案解析出來的：「增量更新」不會讓它回來（檔案沒變就跳過），
                「全量重建」會原樣長回來。</b>要永久排除得改動來源檔案本身。
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function SearchTest() {
  const [q, setQ] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)

  const run = async () => {
    if (!q.trim()) return
    setBusy(true); setResult(null)
    try { setResult(await api.post('/api/admin/search-test', { query: q })) }
    catch (e) { setResult({ error: e.message }) }
    finally { setBusy(false) }
  }

  return (
    <div className="panel">
      <div className="phead"><b>🧪 檢索測試</b></div>
      <div style={{ padding: 15 }}>
        <div style={{ fontSize: 12, color: 'var(--ink-3)', marginBottom: 10 }}>
          輸入問題看實際會檢索到哪些切片與距離，用來確認索引品質。距離越小代表越相似。
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <input className="sel" style={{ flex: 1 }} value={q} placeholder="測試問題…"
                 onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && run()} />
          <button className="btn pri" onClick={run} disabled={busy}>{busy ? '檢索中…' : '執行'}</button>
        </div>

        {result?.error && <div className="note danger" style={{ marginTop: 12 }}>{result.error}</div>}
        {result?.hits && (
          <>
            <div style={{ fontSize: 11.5, color: 'var(--ink-3)', margin: '12px 0 6px' }}>
              保險絲門檻 {result.threshold}（僅擋極端情況，相關性由 LLM 判斷）
            </div>
            <table className="tbl">
              <thead><tr><th>距離</th><th>文件</th><th>位置</th><th>內容</th></tr></thead>
              <tbody>
                {result.hits.map((h) => (
                  <tr key={h.chunk_id}>
                    <td style={{ fontWeight: 600, color: 'var(--accent)' }}>{h.distance}</td>
                    <td>{h.file_name}</td><td>{h.locator}</td>
                    <td style={{ maxWidth: 340, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{h.content}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </div>
  )
}

function Indexing() {
  const [info, setInfo] = useState(null)
  const [log, setLog] = useState([])
  const [busy, setBusy] = useState(false)
  const [files, setFiles] = useState(null)
  const [stage, setStage] = useState('')
  const [stages, setStages] = useState([])
  const [docs, setDocs] = useState([])
  // 上傳的回饋要顯示在上傳區，不能跟索引 log 共用一個位置——
  // 訊息出現在別的面板裡，使用者不會把它跟自己剛按的按鈕連起來。
  const [upLog, setUpLog] = useState([])
  const [forceVlm, setForceVlm] = useState(false)

  const loadDocs = () => api.get('/api/admin/documents').then((d) => setDocs(d.documents))
  const load = () => Promise.all([
    api.get('/api/admin/index-status').then(setInfo),
    loadDocs(),
  ])
  useEffect(() => { load(); api.get('/api/stages').then((d) => setStages(d.stages)) }, [])

  /* 刪除是不可回復的，而且會一併清掉切片與向量——一定要確認。
     訊息裡寫出切片數，讓管理員知道自己刪掉的是什麼份量的東西。 */
  /* 勾選只寫入設定，**不自動重建索引**。
     重建是耗時且會改動整個知識庫的動作，應該由使用者明確按下「增量更新」觸發，
     而不是勾一個核取方塊就默默跑起來。這裡的責任是把「還沒生效」講清楚。 */
  const toggleVlm = async (d, on) => {
    try {
      const r = await api.put(
        `/api/admin/documents/options?path=${encodeURIComponent(d.rel_path)}&force_vlm=${on}`)
      setUpLog([
        `${d.file_name}：${r.message}`,
        on
          ? '→ 尚未生效。請按上方「增量更新」，這一份會用視覺模型重新解析圖片內容。'
          : '→ 尚未生效。請按上方「增量更新」，這一份會改回只讀文字層。',
      ])
    } catch (e) {
      setUpLog([`設定失敗：${e.message}`])
    }
    loadDocs()
  }

  const removeDoc = async (d) => {
    const extra = d.chunk_count ? `，連同 ${d.chunk_count} 個切片與其向量` : ''
    if (!window.confirm(`刪除「${d.file_name}」${extra}？\n檔案會從知識庫資料夾移除，此動作無法復原。`)) return
    try {
      const r = await api.del(`/api/admin/documents?path=${encodeURIComponent(d.rel_path)}`)
      setLog([r.message])
    } catch (e) {
      setLog([`刪除失敗：${e.message}`])
    }
    load()
  }

  const run = async (full) => {
    setBusy(true); setLog([])
    const res = await fetch(`/api/admin/index?full=${full}`, {
      method: 'POST', headers: { Authorization: `Bearer ${getToken()}` },
    })
    const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = ''
    while (true) {
      const { done, value } = await reader.read(); if (done) break
      buf += dec.decode(value, { stream: true })
      const blocks = buf.split('\n\n'); buf = blocks.pop()
      for (const b of blocks) {
        const dline = b.split('\n').find((l) => l.startsWith('data: '))
        if (!dline) continue
        const d = JSON.parse(dline.slice(6))
        if (d.line) setLog((l) => [...l, d.line])
        else setLog((l) => [...l, `完成：新增 ${d.new}、更新 ${d.updated}、失敗 ${d.failed}，共 ${d.chunks} 個切片`])
      }
    }
    setBusy(false); load()
  }

  /* 上傳一定要有 try/catch 與明確回饋。
     沒有的話，後端一個 500 會被整個吞掉——畫面毫無變化，
     使用者只會看到「按了沒反應」，而那正是最難回報的一種 bug。 */
  const upload = async () => {
    if (!files?.length) return
    setBusy(true)
    setUpLog([`上傳中…（${files.length} 個檔案）`])
    try {
      const fd = new FormData()
      for (const f of files) fd.append('files', f)
      const params = new URLSearchParams()
      if (stage) params.set('stage_code', stage)
      if (forceVlm) params.set('force_vlm', 'true')
      const q = params.toString() ? `?${params}` : ''
      const r = await api.upload(`/api/admin/upload${q}`, fd)
      const lines = []
      if (r.saved.length) lines.push(`已存入 ${r.saved.length} 個檔案：${r.saved.join('、')}`)
      lines.push(...(r.errors || []))
      if (!r.saved.length && !r.errors?.length) lines.push('沒有任何檔案被存入')
      if (r.saved.length) lines.push('→ 檔案已在下方清單，執行「增量更新」後才會進入檢索範圍。')
      setUpLog(lines)
      setFiles(null)
      // 清掉 input 的選檔狀態，否則畫面上還留著剛才那個檔名
      const input = document.querySelector('input[type=file]')
      if (input) input.value = ''
    } catch (e) {
      setUpLog([`上傳失敗：${e.message}`])
    } finally {
      setBusy(false)
      load()
    }
  }

  if (!info) return <div className="empty">載入中…</div>

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* 有圖片型檔案但沒有可用 VLM 時提示——但不擋下索引。
          文字檔沒有理由被圖片檔拖住。 */}
      {!info.vlm_ready && info.image_files.length > 0 && (
        <div className="note amber">
          <b>📷 這個資料夾裡有 {info.image_files.length} 個圖片型檔案</b>
          （{info.image_files.slice(0, 4).join('、')}），但目前無法解析內容——{info.vlm_reason}。
          <div style={{ marginTop: 6 }}>
            安裝視覺模型後重新索引即可讀取：<code>ollama pull {info.recommended_vlm}</code>
          </div>
          <div style={{ marginTop: 6 }}>
            現在就索引也沒問題——圖片型檔案會保留檔名與階段歸類，只是圖片裡的文字不會進入索引。
          </div>
        </div>
      )}

      <div className="panel">
        <div className="phead"><b>上傳文件</b></div>
        <div style={{ padding: 15, display: 'flex', gap: 9, alignItems: 'center', flexWrap: 'wrap' }}>
          <select className="sel" value={stage} onChange={(e) => setStage(e.target.value)}>
            <option value="">（未分類）</option>
            {stages.map((s) => <option key={s.code} value={s.code}>{s.code} — {s.name_zh}</option>)}
          </select>
          {/* accept 讓檔案選擇器預設只列得出支援的格式，
              少一次「選了才被退回」的來回。清單來自後端，不在前端寫死。 */}
          <input type="file" multiple
                 accept={[...info.supported, ...info.image_types].join(',')}
                 onChange={(e) => setFiles(e.target.files)} style={{ fontSize: 12.5 }} />
          <button className="btn pri" onClick={upload} disabled={!files?.length}>上傳</button>
        </div>

        {/* 強制視覺解析。預設關閉——VLM 每張圖都要跑一次推論，
            對純文字文件是純浪費，不該變成預設行為。 */}
        <div style={{ padding: '0 15px 12px' }}>
          <label style={{ display: 'flex', gap: 7, alignItems: 'flex-start', cursor: 'pointer' }}>
            <input type="checkbox" checked={forceVlm} disabled={!info.vlm_ready}
                   onChange={(e) => setForceVlm(e.target.checked)}
                   style={{ marginTop: 3 }} />
            <span style={{ fontSize: 12.5, color: 'var(--ink-2)', lineHeight: 1.6 }}>
              <b>強制以視覺模型解析圖片內容</b>
              {!info.vlm_ready && <span style={{ color: 'var(--amber)' }}>（視覺模型未就緒，無法使用）</span>}
              <div style={{ fontSize: 11.5, color: 'var(--ink-3)', marginTop: 3 }}>
                圖多的簡報、含流程圖的規範書勾這個。
                <b>預設只讀文字層</b>，圖片裡的架構圖、流程圖、表格截圖不會進入索引。
                勾選後文字與圖片內容**兩者都會保留**，代價是每張圖要跑一次推論，索引變慢。
              </div>
            </span>
          </label>
        </div>

        {/* 支援格式必須寫清楚。不寫的話使用者只能靠上傳失敗來試出來，
            而圖片型檔案還有「需要視覺模型才讀得到內容」這個附加條件，
            跟其他格式不是同一回事，要分開講。 */}
        <div style={{ padding: '0 15px 12px', display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div>
            <div style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--ink-2)', marginBottom: 5 }}>
              支援的文件格式（{info.supported.length} 種，內容會被解析並進入檢索）
            </div>
            <div className="chips">
              {info.supported.map((ext) => <span className="chip" key={ext}>{ext}</span>)}
            </div>
          </div>

          <div>
            <div style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--ink-2)', marginBottom: 5 }}>
              圖片格式（{info.image_types.length} 種，
              {info.vlm_ready
                ? <span style={{ color: 'var(--accent)' }}>視覺模型已就緒，圖中文字讀得到</span>
                : <span style={{ color: 'var(--amber)' }}>需要視覺模型才讀得到圖中文字</span>})
            </div>
            <div className="chips">
              {info.image_types.map((ext) => <span className="chip" key={ext}>{ext}</span>)}
            </div>
          </div>
        </div>

        <div style={{ padding: '0 15px 14px', fontSize: 11.5, color: 'var(--ink-3)', lineHeight: 1.7 }}>
          選擇階段會存進該階段的子資料夾，檢索時就能用階段過濾。
          直接用檔案總管把檔案複製到 <code>{info.root}</code> 效果相同。
          <br />
          <b>其他副檔名會被擋下並在上傳結果列出原因</b>——不會靜默略過。
          掃描型 PDF（整頁是圖片、選不到文字）走的是圖片路徑，同樣需要視覺模型。
        </div>

        {upLog.length > 0 && (
          <div style={{ padding: '0 15px 15px' }}>
            <div className={`note ${upLog.some((l) => l.includes('失敗')) ? 'danger' : 'accent'}`}>
              {upLog.map((l, i) => <div key={i}>{l}</div>)}
            </div>
          </div>
        )}
      </div>

      <div className="panel">
        <div className="phead">
          <b>建立索引</b>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn" onClick={() => run(false)} disabled={busy}>增量更新</button>
            <button className="btn" onClick={() => run(true)} disabled={busy}>全量重建</button>
          </div>
        </div>
        <div style={{ padding: 15 }}>
          <div style={{ fontSize: 12.5, color: 'var(--ink-2)', marginBottom: 10 }}>
            {info.stats.documents} 份文件 · {info.stats.chunks} 個切片 · 最後索引 {info.stats.last_indexed}
          </div>
          {(busy || log.length > 0) && <div className="log">{busy && !log.length ? '索引中…' : log.join('\n')}</div>}
        </div>
      </div>

      {/* 文件清單。**列的是磁碟上的檔案，不是 documents 資料表。**
          上傳完、還沒建索引之前沒有 Document 列，只查資料表的話
          剛上傳的檔案會看不見——那正是最需要確認「存進去沒有」的時刻。 */}
      <div className="panel">
        <div className="phead">
          <b>知識庫文件</b>
          <span className="pill">{docs.length}</span>
        </div>
        {docs.length === 0 ? (
          <div className="empty" style={{ fontSize: 12.5 }}>資料夾裡還沒有檔案</div>
        ) : docs.map((d) => (
          <div className="row" key={d.rel_path} style={{ cursor: 'default' }}>
            <div className="ft">{d.file_type.slice(0, 4).toUpperCase()}</div>
            <div className="rinfo">
              <b>{d.file_name}</b>
              <span>
                {d.stage_code && <span className="tag" style={{ marginRight: 6 }}>{d.stage_code}</span>}
                {(d.file_size / 1024).toFixed(0)} KB
                {d.indexed
                  ? ` · ${d.chunk_count} 個切片 · ${d.indexed_at}`
                  : <b style={{ color: 'var(--amber)' }}> · 尚未索引</b>}
                {d.used_vlm && ' · 視覺解析'}
              </span>
            </div>
            <label title="強制以視覺模型解析圖片內容（改完需重新索引）"
                   style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11,
                            color: d.force_vlm ? 'var(--accent)' : 'var(--ink-3)', cursor: 'pointer' }}
                   onClick={(e) => e.stopPropagation()}>
              <input type="checkbox" checked={!!d.force_vlm}
                     onChange={(e) => toggleVlm(d, e.target.checked)} />
              視覺解析
            </label>
            <button className="btn" onClick={() => removeDoc(d)}>刪除</button>
          </div>
        ))}
        <div style={{ padding: '10px 15px', fontSize: 11.5, color: 'var(--ink-3)', borderTop: '1px solid var(--line)' }}>
          刪除會同時移除檔案本身、它的切片與向量，以及管理員為它調過的檢索關鍵字。
          <b>標為「尚未索引」的檔案已經存在資料夾裡</b>，執行一次增量更新就會進入檢索範圍。
        </div>
      </div>
    </div>
  )
}

function Errors() {
  const [data, setData] = useState(null)
  useEffect(() => { api.get('/api/admin/errors').then(setData) }, [])
  if (!data) return <div className="empty">載入中…</div>

  const table = (rows) => (
    <table className="tbl">
      <thead><tr><th>檔案</th><th>訊息</th><th>時間</th></tr></thead>
      <tbody>{rows.map((e, i) => (
        <tr key={i}><td>{e.file_name}</td><td>{e.message}</td><td>{e.occurred_at}</td></tr>
      ))}</tbody>
    </table>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* 「缺少視覺模型」與「檔案解析失敗」分開呈現。
          混在一起會讓管理員以為圖片檔壞了，實際上只是少裝一個模型。 */}
      {data.needs_vlm.length > 0 && (
        <div className="panel">
          <div className="phead"><b>📷 等待視覺模型</b><span className="pill">{data.needs_vlm.length}</span></div>
          {table(data.needs_vlm)}
          <div style={{ padding: 12, fontSize: 11.5, color: 'var(--ink-3)' }}>
            <b>這些檔案沒有問題</b>，只是內容在影像裡、需要視覺模型才讀得出來。
            檔名與階段歸類仍然保留，可以用關鍵字找到。
          </div>
        </div>
      )}
      <div className="panel">
        <div className="phead"><b>⚠️ 解析失敗</b><span className="pill">{data.failures.length}</span></div>
        {data.failures.length ? table(data.failures)
          : <div className="empty">沒有解析失敗的檔案</div>}
      </div>
    </div>
  )
}

const TABS = [
  { key: 'index', label: '文件與索引', el: <Indexing /> },
  { key: 'chunks', label: '切片內容', el: <Chunks /> },
  { key: 'errors', label: '解析狀態', el: <Errors /> },
  { key: 'test', label: '檢索測試', el: <SearchTest /> },
]

export default function Library() {
  const [tab, setTab] = useState('chunks')
  return (
    <>
      <header className="top">
        <div><h1>知識庫維護</h1><div className="crumb">僅管理員可存取</div></div>
      </header>
      <div className="tabs">
        {TABS.map((t) => (
          <button key={t.key} className={`tab ${tab === t.key ? 'on' : ''}`} onClick={() => setTab(t.key)}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="page">{TABS.find((t) => t.key === tab).el}</div>
    </>
  )
}
