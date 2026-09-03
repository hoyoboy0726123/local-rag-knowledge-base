import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api, stream } from '../app/api'
import DocActions from '../components/DocActions'

/* 依文件分組，組內維持原本的引註順序。
 * 用 Map 而不是物件：物件的鍵順序在數字型檔名下會被 JS 重排。 */
function groupByFile(sources) {
  const map = new Map()
  for (const s of sources) {
    if (!map.has(s.file_name)) map.set(s.file_name, [])
    map.get(s.file_name).push(s)
  }
  return [...map.entries()]
}

/* 把答案裡的 [1]、[2, 3] 換成可點的引註色塊。
 *
 * **以 chunk_id 為鍵對應來源，不要用顯示序號硬對。**
 * 回答裡的編號是模型寫的，來源面板的排序可能因為去重而不同，
 * 用序號硬對會在多輪對話後錯位。 */
function withCitations(text, onCite) {
  const parts = []
  let last = 0
  const re = /\[(\d+(?:\s*,\s*\d+)*)\]/g
  let m
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index))
    for (const n of m[1].split(',').map((s) => s.trim())) {
      parts.push(
        <span key={`${m.index}-${n}`} className="cite" onClick={() => onCite(Number(n))}>{n}</span>
      )
    }
    last = re.lastIndex
  }
  if (last < text.length) parts.push(text.slice(last))
  return parts
}

/* 防禦性清掉模型偶爾殘留的 LaTeX。
 *
 * system instruction 已明確要求不要用 LaTeX（溫度直接寫 25°C），
 * 但模型偶爾還是會吐出來。
 *
 * **V1 沒發現這件事**，是因為 Streamlit 的 st.markdown 內建 KaTeX 會把它渲染掉——
 * 換成一般的 markdown 渲染器就原形畢露。內容本身其實就不該有 LaTeX：
 * 溫度與型號是規格數值，不是數學式。 */
function stripLatex(text) {
  return text.replace(/\$([^$\n]{0,80}?)\$/g, (_full, inner) =>
    inner
      .replace(/\\text\{([^}]*)\}/g, '$1')
      .replace(/\\mathrm\{([^}]*)\}/g, '$1')
      .replace(/\^\s*\\circ/g, '°')
      .replace(/\\circ/g, '°')
      .replace(/\\times/g, '×')
      .replace(/\\[a-zA-Z]+/g, '')
      .replace(/[{}]/g, '')
      .replace(/\s+/g, ' ')
      .trim()
  )
}

/* 一次檢索的軌跡膠囊。問答進行中與歷史訊息共用同一個元件——
   兩邊各寫一次的下場是改了一邊忘了另一邊。

   **「0 段」有兩種意思，一定要分開講。** 一種是真的沒撈到，另一種是
   「撈到了但這一輪前面已經給過」——後者在追問時是常態，畫面上若都顯示
   「0 段」，使用者只會以為知識庫裡沒東西。

   query 用單行省略：模型有時會產生兩百字的關鍵詞串（實測看過整張分類表被
   當成 query 送出），完整攤開會讓膠囊撐成一大塊，把答案擠到畫面外。
   完整內容留在 title，滑過去看得到。 */
function Trace({ search }) {
  const { query, stage, hits, seen_only: seenOnly } = search
  const count = hits > 0
    ? `${hits} 段`
    : (seenOnly ? '0 段（前面已提供）' : '0 段')
  return (
    <div className="trace" title={query}>
      <span className="k">◐ 檢索</span>
      <span className="q">{query}</span>
      {stage && <span className="st">（{stage}）</span>}
      <b className={hits > 0 ? '' : 'zero'}>· {count}</b>
    </div>
  )
}

function Answer({ text, onCite }) {
  return (
    <div className="ans">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // 只在文字節點做引註替換，才不會破壞表格與清單結構
          p: ({ children }) => <p>{mapChildren(children, onCite)}</p>,
          li: ({ children }) => <li>{mapChildren(children, onCite)}</li>,
          td: ({ children }) => <td>{mapChildren(children, onCite)}</td>,
          h1: ({ children }) => <h1>{mapChildren(children, onCite)}</h1>,
          h2: ({ children }) => <h2>{mapChildren(children, onCite)}</h2>,
          h3: ({ children }) => <h3>{mapChildren(children, onCite)}</h3>,
          h4: ({ children }) => <h4>{mapChildren(children, onCite)}</h4>,
          strong: ({ children }) => <strong>{mapChildren(children, onCite)}</strong>,
        }}
      >
        {stripLatex(text)}
      </ReactMarkdown>
    </div>
  )
}

function mapChildren(children, onCite) {
  return (Array.isArray(children) ? children : [children]).map((c, i) =>
    typeof c === 'string' ? <span key={i}>{withCitations(c, onCite)}</span> : c
  )
}

/* 來源全文。
 *
 * 側欄只有 300px 寬，長切片在裡面讀不了——所以「展開」與「看全文」是兩件事：
 * 卡片內展開適合掃一眼，這個浮層才是真的閱讀。
 * 切片本身是 Markdown 片段（含表格），純文字顯示會讓表格糊成一團。 */
function SourceModal({ src, onClose }) {
  // 'chunk' = AI 實際讀到的那一段；'doc' = 整份文件。
  // 預設停在切片，因為要判斷「AI 為什麼這樣答」時，該看的是它讀到的那段。
  const [view, setView] = useState('chunk')
  const [doc, setDoc] = useState(null)
  const [docError, setDocError] = useState('')

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // 換一份來源時清掉上一份的內容。`_openDoc` 讓卡片能直接跳到完整文件檢視。
  useEffect(() => {
    setDoc(null); setDocError('')
    setView(src?._openDoc ? 'doc' : 'chunk')
  }, [src])

  // 內容只在真的切到 doc 檢視時才抓，而且只抓一次
  useEffect(() => {
    if (view !== 'doc' || !src || doc !== null || docError) return
    api.get(`/api/documents/content?path=${encodeURIComponent(src.file_path)}`)
      .then((d) => setDoc(d.content))
      .catch((e) => setDocError(e.message))
  }, [view, src, doc, docError])

  if (!src) return null
  const showingDoc = view === 'doc'
  return (
    <div className="mask" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="mhead">
          <div style={{ minWidth: 0 }}>
            <b>{src.file_name}</b>
            <div className="smeta" style={{ marginLeft: 0, marginTop: 3 }}>
              {src.stage_code && <span className="tag">{src.stage_code}</span>}
              <span>{showingDoc ? '完整文件' : src.locator}</span>
            </div>
          </div>
          <button className="btn" onClick={onClose}>關閉</button>
        </div>

        <DocActions
          path={src.file_path}
          name={src.file_name}
          onOpenDoc={showingDoc ? null : () => setView('doc')}
        />
        {showingDoc && (
          <button className="lnk backlnk" onClick={() => setView('chunk')}>
            ← 回到 AI 讀到的段落
          </button>
        )}

        <div className="docbody">
          {!showingDoc && <ReactMarkdown remarkPlugins={[remarkGfm]}>{src.content}</ReactMarkdown>}
          {showingDoc && docError && <div className="empty">{docError}</div>}
          {showingDoc && !docError && doc === null && <div className="empty">讀取中…</div>}
          {showingDoc && doc !== null && (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{doc}</ReactMarkdown>
          )}
        </div>

        <div className="mfoot">
          {showingDoc
            ? <>這是<b>整份文件</b>解析後的內容（{doc?.length ?? 0} 字）。要原始排版請下載原始檔。</>
            : <>這是知識庫裡<b>實際被檢索到的切片原文</b>，AI 的回答只能依據這段內容。</>}
        </div>
      </div>
    </div>
  )
}

export default function Chat() {
  // **一開始不建立 session。**
  // 進頁面就先建，會讓每次點開都留下一筆「0 則」的空對話——實測二十筆裡有一半是這樣來的。
  // 改成第一次送出問題時才建。
  const [sessionId, setSessionId] = useState(null)
  const [sessions, setSessions] = useState([])
  const [messages, setMessages] = useState([])
  const [stages, setStages] = useState([])
  const [scope, setScope] = useState('')
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [live, setLive] = useState(null)     // 進行中的回答
  const [selected, setSelected] = useState(null)
  const [opened, setOpened] = useState(null) // 卡片內展開的來源序號
  const [modal, setModal] = useState(null)   // 看全文的來源
  const [rawFor, setRawFor] = useState(null) // 展開原始片段的訊息索引
  const [railOpen, setRailOpen] = useState(true)
  const endRef = useRef(null)

  const loadSessions = () => api.get('/api/chat/sessions').then((d) => setSessions(d.sessions))

  useEffect(() => {
    api.get('/api/stages').then((d) => setStages(d.stages))
    loadSessions()
  }, [])

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, live])

  const newChat = () => {
    setSessionId(null); setMessages([]); setLive(null)
    setSelected(null); setOpened(null)
  }

  /* 開啟舊對話並接續。
   * session_id 一送回後端，`get_messages` 就會把整段前文餵給 Agent，
   * 所以接著追問「還有嗎」是接得住的——不需要前端另外傳 history。 */
  const openSession = async (id) => {
    if (busy) return
    const d = await api.get(`/api/chat/sessions/${id}/messages`)
    setSessionId(id)
    setMessages(d.messages.map((m) => ({
      role: m.role,
      content: m.content,
      sources: m.sources || [],
      // 歷史訊息沒有留檢索軌跡（那是串流當下的過程），只還原答案與來源
      searches: [],
    })))
    setLive(null); setSelected(null); setOpened(null)
  }

  const removeSession = async (id, title) => {
    if (!window.confirm(`刪除「${title}」？此動作無法復原。`)) return
    await api.del(`/api/chat/sessions/${id}`)
    if (id === sessionId) newChat()
    loadSessions()
  }

  /* `wide` = 用更多脈絡重問。
   *
   * 一般模式每個命中只補前後各一段；wide 會擴展到整個結構單元（章節／投影片／
   * 記錄）。**做成使用者按下才啟用**，因為它不是免付費的：脈絡變多會變慢，
   * 而且塞太多反而會讓模型忽略中間的內容。答案不完整時再花這個成本才划算。 */
  const ask = async (question, wide = false) => {
    if (!question.trim() || busy) return
    setInput(''); setBusy(true); setSelected(null); setOpened(null)

    let sid = sessionId
    if (!sid) {
      sid = (await api.post('/api/chat/sessions')).session_id
      setSessionId(sid)
    }

    setMessages((m) => [...m, { role: 'user', content: question, wide }])
    setLive({ searches: [], text: '', sources: [], wide })

    try {
      await stream('/api/chat/ask',
        { session_id: sid, question, stage_code: scope || null, wide },
        (name, data) => {
          if (name === 'search') {
            setLive((l) => ({ ...l, searches: [...l.searches, data] }))
          } else if (name === 'text') {
            setLive((l) => ({ ...l, text: l.text + data.piece }))
          } else if (name === 'error') {
            setLive((l) => ({ ...l, text: `⚠️ ${data.message}` }))
          } else if (name === 'done') {
            // **以 done 的 answer 為準**，不要用串流累積的 buffer：
            // 未檢索前產生的文字不會經由 text 事件送出（後端的防護），
            // 用 buffer 當最終答案等於繞過那層防護。
            setLive((l) => {
              setMessages((m) => [...m, {
                role: 'assistant',
                content: data.answer || l.text,
                sources: data.sources,
                searches: l.searches,
              }])
              return null
            })
          }
        })
    } catch (e) {
      setMessages((m) => [...m, { role: 'assistant', content: `⚠️ ${e.message}`, sources: [] }])
      setLive(null)
    } finally {
      setBusy(false)
      loadSessions()  // 標題由後端以第一句提問產生，這裡把清單同步回來
    }
  }

  // 點答案裡的 [n]：高亮、展開、捲到那張卡
  const gotoCite = (n) => {
    setSelected(n); setOpened(n)
    requestAnimationFrame(() =>
      document.getElementById(`src-${n}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    )
  }

  const lastSources = [...messages].reverse().find((m) => m.role === 'assistant')?.sources || []
  const shown = live ? [] : lastSources

  return (
    <>
      <header className="top">
        <div>
          <h1>AI 問答</h1>
          <div className="crumb">回答僅依據知識庫內容，每項結論均可追溯</div>
        </div>
        <div className="tools">
          <button className="btn ghost" onClick={() => setRailOpen((v) => !v)} title="收合對話清單">
            {railOpen ? '◧ 收合清單' : '◨ 展開清單'}
          </button>
          <select className="sel" value={scope} onChange={(e) => setScope(e.target.value)}>
            <option value="">檢索範圍：全部階段</option>
            {stages.map((s) => <option key={s.code} value={s.code}>{s.code} — {s.name_zh}</option>)}
          </select>
        </div>
      </header>

      <div className={`chatwrap ${railOpen ? '' : 'norail'}`}>
        {railOpen && (
          <aside className="rail">
            <button className="btn pri newconv" onClick={newChat}>＋ 新對話</button>
            <div className="raillbl">我的對話<span>只有你看得到</span></div>
            <div className="convlist">
              {sessions.length === 0 && <div className="empty" style={{ fontSize: 12 }}>還沒有對話</div>}
              {sessions.map((s) => (
                <div key={s.ID} className={`conv ${s.ID === sessionId ? 'on' : ''}`}
                     onClick={() => openSession(s.ID)}>
                  <div className="rinfo">
                    <b>{s.標題}</b>
                    <span>{s.訊息數} 則 · {s.最後更新}</span>
                  </div>
                  <button className="x" title="刪除"
                          onClick={(e) => { e.stopPropagation(); removeSession(s.ID, s.標題) }}>×</button>
                </div>
              ))}
            </div>
          </aside>
        )}

        <div style={{ minWidth: 0 }}>
          {messages.length === 0 && !live && (
            <div className="note accent" style={{ marginBottom: 18 }}>
              <b>可以這樣問：</b> 問完之後直接追問「還有嗎」也聽得懂，不必重複主詞。
              知識庫沒有的內容系統會直接說查無資訊，不會臆測。
            </div>
          )}

          <div className="thread">
            {messages.map((m, i) => m.role === 'user' ? (
              <div className="msg" key={i}>
                <div className="av me">問</div>
                <div className="qtext">
                  {m.content}
                  {m.wide && <span className="widetag">更多脈絡</span>}
                </div>
              </div>
            ) : (
              <div className="msg" key={i}>
                <div className="av ai">◈</div>
                <div className="abody">
                  {(m.searches || []).map((s, j) => (
                    <Trace search={s} key={j} />
                  ))}
                  <Answer text={m.content} onCite={gotoCite} />
                  {/* 答案不完整時的補救。用前一則使用者訊息當問題重問，
                      使用者不必自己再打一次。已經用過更多脈絡的就不再提供，
                      免得重複點下去只是原地重跑。 */}
                  {(() => {
                    const q = messages[i - 1]
                    const hasQ = q && q.role === 'user'
                    const canWide = hasQ && !q.wide
                    const raws = m.sources || []
                    if (!canWide && !raws.length) return null
                    return (
                      <>
                        <div className="widerow">
                          {canWide && <span className="widehint">答案不夠完整？</span>}
                          {canWide && (
                            <button className="lnk" disabled={busy}
                                    onClick={() => ask(q.content, true)}
                                    title="每個來源擴展到整個章節／投影片／記錄——讀得更深">
                              🔍 用更多脈絡重問
                            </button>
                          )}
                          {/* 原始片段。**AI 的敘述再怎麼寫都是改寫**，而規格表、
                              對照表這類內容的原文形式本身就是最好的答案形式——
                              實測「大類／敘述」對照表被改寫成條列後代號欄整欄消失。
                              預設收合：多數問題不需要，需要的人點一下就看得到。 */}
                          {raws.length > 0 && (
                            <button className="lnk"
                                    onClick={() => setRawFor(rawFor === i ? null : i)}
                                    title="顯示檢索到的原始內容，未經 AI 改寫">
                              📄 {rawFor === i ? '收合原始片段' : `顯示原始片段（${raws.length}）`}
                            </button>
                          )}
                        </div>
                        {rawFor === i && (
                          <div className="rawsrc">
                            {raws.map((s) => (
                              <div className="rawitem" key={s.index}>
                                <div className="rawhead">
                                  <span className="num">{s.index}</span>
                                  {s.locator || s.file_name}
                                </div>
                                {/* 用 Markdown 渲染，表格才會是表格。
                                    這裡刻意不套引註色塊：原文裡的數字不是引註。 */}
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                  {s.content}
                                </ReactMarkdown>
                              </div>
                            ))}
                          </div>
                        )}
                      </>
                    )
                  })()}
                </div>
              </div>
            ))}

            {live && (
              <div className="msg">
                <div className="av ai">◈</div>
                <div className="abody">
                  {live.searches.map((s, j) => (
                    <Trace search={s} key={j} />
                  ))}
                  {live.text
                    ? <Answer text={live.text} onCite={() => {}} />
                    : <div style={{ color: 'var(--ink-3)', fontSize: 13 }}>思考中…</div>}
                </div>
              </div>
            )}
            <div ref={endRef} />
          </div>

          <div className="composer">
            <input
              value={input}
              placeholder="輸入你的問題…"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && ask(input)}
              disabled={busy}
            />
            <div className="crow">
              {/* 不放建議問題按鈕：任何預設問句都綁定特定領域，
                  對其他知識庫的使用者是噪音。 */}
              <span />
              <button className="send" onClick={() => ask(input)} disabled={busy}>↑</button>
            </div>
          </div>
        </div>

        <aside className="aside">
          <div className="panel">
            <div className="phead"><b>參考來源</b><span className="pill">{shown.length}</span></div>
            {shown.length === 0 ? (
              <div className="empty" style={{ padding: 22, fontSize: 12 }}>提問後這裡會列出依據</div>
            ) : groupByFile(shown).map(([fileName, items]) => (
              <div key={fileName} className="srcgroup">
                {/* 依文件分組，並保留原本的引註編號。
                    來源混了幾份文件是使用者該看得見的事——不同文件可能是不同標準，
                    把它們平鋪成一串會讓人以為那是同一份規格。 */}
                <div className="srcgrouphead">
                  <b>{fileName}</b>
                  {items[0].stage_code && <span className="tag">{items[0].stage_code}</span>}
                  <span className="pill">{items.length}</span>
                </div>
                {items.map((s) => (
                  <div key={s.index} id={`src-${s.index}`}
                       className={`src ${selected === s.index ? 'sel' : ''}`}
                       onClick={() => {
                         setSelected(s.index)
                         setOpened((o) => (o === s.index ? null : s.index))
                       }}>
                    <div className="srow">
                      <span className="num">{s.index}</span>
                      <b>{s.locator || s.file_name}</b>
                      <span className="caret">{opened === s.index ? '▴' : '▾'}</span>
                    </div>
                    <div className={`snip ${opened === s.index ? 'open' : ''}`}>{s.content}</div>
                    {opened === s.index && (
                      <div className="srcacts" onClick={(e) => e.stopPropagation()}>
                        <button className="btn full" onClick={() => setModal(s)}>
                          ⤢ 看這段全文
                        </button>
                        <button className="btn full" onClick={() => setModal({ ...s, _openDoc: true })}>
                          📄 開啟完整文件
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ))}
          </div>
          {groupByFile(shown).length > 1 && (
            <div className="note amber">
              <b>這個回答引用了 {groupByFile(shown).length} 份不同文件。</b>
              不同文件可能是不同的標準或版本，
              數值請對照各自的來源確認，不要當成同一份規格。
            </div>
          )}
          <div className="note amber">
            <b>為什麼只回答這些？</b> 系統只依據上方來源作答。
            知識庫沒有的內容不會被補完，也不會臆測。
          </div>
        </aside>
      </div>

      <SourceModal src={modal} onClose={() => setModal(null)} />
    </>
  )
}
