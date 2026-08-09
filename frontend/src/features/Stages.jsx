import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from '../app/api'
import DocActions from '../components/DocActions'

/* 文件閱讀器。
 *
 * **不用 os.startfile。** 系統部署在區網伺服器上，那個呼叫會在伺服器上開檔，
 * 遠端使用者點下去什麼也不會發生。內容一律送到瀏覽器渲染 + 提供下載。
 *
 * 以 Markdown 渲染的理由：這些文件多半是規範與查核表，
 * 表格結構正是重點；用純文字會變成一整片管線符號。 */
function DocReader({ path, name }) {
  const [content, setContent] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setContent(null); setError('')
    api.get(`/api/documents/content?path=${encodeURIComponent(path)}`)
      .then((d) => setContent(d.content))
      .catch((e) => setError(e.message))
  }, [path])

  return (
    <div className="panel" style={{ marginTop: 12 }}>
      <div className="phead">
        <b>{name}</b>
      </div>
      <DocActions path={path} name={name} />
      {error && <div className="empty">{error}</div>}
      {!error && content === null && <div className="empty">讀取中…</div>}
      {content !== null && (
        <>
          <div style={{ padding: '9px 15px 0', fontSize: 11.5, color: 'var(--ink-3)' }}>
            以下為系統解析後的內容，與 AI 讀到的文字一致（{content.length} 字）
          </div>
          <div className="docbody">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          </div>
        </>
      )}
    </div>
  )
}

function DocList({ documents, onPick, picked }) {
  if (!documents.length) return <div className="empty">此階段目前沒有已索引的文件</div>
  return documents.map((d) => (
    <div key={d.路徑} className={`row ${picked === d.路徑 ? 'on' : ''}`} onClick={() => onPick(d)}>
      <div className="ft">{d.類型}</div>
      <div className="rinfo">
        <b>{d.文件名稱}</b>
        <span>{d.切片數} 個切片 · {d.索引時間}</span>
      </div>
      {d['VLM 解析'] && <span className="tag kw">VLM</span>}
    </div>
  ))
}

export default function Stages() {
  const [stages, setStages] = useState([])
  const [stats, setStats] = useState(null)
  const [current, setCurrent] = useState(null)
  const [docs, setDocs] = useState([])
  const [common, setCommon] = useState([])
  const [picked, setPicked] = useState(null)

  useEffect(() => {
    api.get('/api/stages').then((d) => { setStages(d.stages); setStats(d.stats) })
    // 未歸屬階段的文件：時間軸只有六個階段，沒有這一區就完全沒有入口
    api.get('/api/documents/unclassified').then((d) => setCommon(d.documents))
  }, [])

  useEffect(() => {
    setPicked(null)
    if (!current) return setDocs([])
    api.get(`/api/stages/${current}/documents`).then((d) => setDocs(d.documents))
  }, [current])

  const stage = stages.find((s) => s.code === current)

  return (
    <>
      <header className="top">
        <div>
          <h1>階段導覽</h1>
          <div className="crumb">
            {stats && `${stats.documents} 份文件 · ${stats.chunks} 個切片 · 最後索引 ${stats.last_indexed}`}
          </div>
        </div>
      </header>

      <div className="page">
        <div className="stages">
          {stages.map((s) => (
            <div key={s.code} className={`stage ${current === s.code ? 'on' : ''}`}
                 onClick={() => setCurrent(current === s.code ? null : s.code)}>
              <b>{s.code}</b><span>{s.name_zh}</span><i>{s.doc_count} 份</i>
            </div>
          ))}
        </div>

        {common.length > 0 && (
          <div className="panel" style={{ marginBottom: 18 }}>
            <div className="phead">
              <b>📁 共通文件（未歸屬特定階段）</b><span className="pill">{common.length}</span>
            </div>
            <div style={{ padding: '9px 13px 0', fontSize: 11.5, color: 'var(--ink-3)' }}>
              這些文件沒有放在階段子資料夾中，因此不屬於任何單一階段，但一樣會被 AI 檢索到。
            </div>
            <DocList documents={common} picked={picked?.路徑} onPick={setPicked} />
          </div>
        )}

        {!current ? (
          <div className="note accent">請由上方選擇一個階段，查看該階段的產出物與掛載文件。</div>
        ) : (
          <div className="grid2">
            <div className="panel">
              <div className="phead"><b>本階段產出物</b></div>
              {(stage?.deliverables || []).length === 0
                ? <div className="empty" style={{ padding: 20, fontSize: 12 }}>尚未定義</div>
                : <div style={{ padding: '10px 13px' }}>
                    {stage.deliverables.map((d, i) => (
                      <label key={i} style={{ display: 'flex', gap: 8, alignItems: 'flex-start', marginBottom: 7, fontSize: 13, color: 'var(--ink-2)' }}>
                        <input type="checkbox" style={{ marginTop: 4 }} />{d}
                      </label>
                    ))}
                    <div style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 8 }}>
                      此處勾選僅供個人備忘，不會寫入資料庫。
                    </div>
                  </div>}
            </div>

            <div>
              <div className="panel">
                <div className="phead"><b>掛載文件</b><span className="pill">{docs.length}</span></div>
                <DocList documents={docs} picked={picked?.路徑} onPick={setPicked} />
              </div>
              {picked && <DocReader path={picked.路徑} name={picked.文件名稱} />}
            </div>
          </div>
        )}
      </div>
    </>
  )
}
