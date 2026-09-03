import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from '../app/api'
import DocActions from '../components/DocActions'

/* 知識庫 = 知識庫根目錄下的一個子資料夾；「通用」是直接放在根目錄的檔案。
   資料夾是唯一的真相來源——用檔案總管建資料夾、搬檔案，跟在這裡操作是同一件事，
   全量重建不會丟失分類。搬移不重新向量化：內容沒變，只同步路徑。 */

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
      <div className="phead"><b>{name}</b></div>
      <DocActions path={path} name={name} />
      {error && <div className="empty">{error}</div>}
      {!error && content === null && <div className="empty">讀取中…</div>}
      {content !== null && (
        <>
          <div style={{ padding: '9px 15px 0', fontSize: 11.5, color: 'var(--ink-3)' }}>
            以下為系統解析後的內容，與 AI 讀到的文字一致（{content.length} 字）
          </div>
          <div className="docbody"><ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown></div>
        </>
      )}
    </div>
  )
}

/* 文件列。管理員多一個「搬到…」下拉，直接換知識庫。
   rel_path 從絕對路徑推：後端 move 端點吃的是相對於根目錄的路徑。 */
function DocRow({ d, picked, onPick, isAdmin, kbs, root, onMoved }) {
  const abs = d['路徑']
  const rel = root && abs.startsWith(root)
    ? abs.slice(root.length).replace(/^[\\/]+/, '').replace(/\\/g, '/')
    : null
  const move = async (e) => {
    e.stopPropagation()
    const kb = e.target.value
    if (kb === '__keep') return
    try {
      const r = await api.post('/api/admin/documents/move', { rel_path: rel, kb: kb || null })
      onMoved(r.message)
    } catch (err) { onMoved(err.message) }
  }
  return (
    <div className={`row ${picked === abs ? 'on' : ''}`} onClick={() => onPick(d)}>
      <div className="ft">{d['類型']}</div>
      <div className="rinfo">
        <b>{d['文件名稱']}</b>
        <span>{d['切片數']} 個切片 · {d['索引時間']}</span>
      </div>
      {d['VLM 解析'] && <span className="tag kw">VLM</span>}
      {isAdmin && rel !== null && (
        <select className="sel" value="__keep" onClick={(e) => e.stopPropagation()} onChange={move}
                style={{ marginLeft: 8, fontSize: 11 }}>
          <option value="__keep">搬到…</option>
          {kbs.filter((k) => k.label !== (d['知識庫'] || '通用')).map((k) => (
            <option key={k.name} value={k.name}>{k.label}</option>
          ))}
        </select>
      )}
    </div>
  )
}

export default function KnowledgeBases() {
  const [kbs, setKbs] = useState([])
  const [stats, setStats] = useState(null)
  const [current, setCurrent] = useState('')      // '' = 通用
  const [docs, setDocs] = useState([])
  const [picked, setPicked] = useState(null)
  const [me, setMe] = useState(null)
  const [root, setRoot] = useState('')
  const [msg, setMsg] = useState('')
  const [newName, setNewName] = useState('')
  const isAdmin = me?.role === 'ADMIN'

  const load = () => api.get('/api/kbs').then((d) => { setKbs(d.kbs); setStats(d.stats) })
  useEffect(() => {
    load()
    api.get('/api/auth/me').then((u) => {
      setMe(u)
      if (u.role === 'ADMIN') api.get('/api/admin/documents').then((d) => setRoot(d.root)).catch(() => {})
    })
  }, [])
  const loadDocs = () => api.get(`/api/kbs/documents?kb=${encodeURIComponent(current)}`).then((d) => setDocs(d.documents))
  useEffect(() => { setPicked(null); loadDocs() }, [current])

  const flash = (m) => { setMsg(m); setTimeout(() => setMsg(''), 4000) }
  const refresh = (m) => { flash(m); load(); loadDocs() }

  const create = async () => {
    if (!newName.trim()) return
    try { const r = await api.post('/api/admin/kbs', { name: newName.trim() }); setNewName(''); refresh(r.message) }
    catch (e) { flash(e.message) }
  }
  const rename = async () => {
    const n = window.prompt('新的知識庫名稱', current)
    if (!n || n === current) return
    try { const r = await api.put(`/api/admin/kbs/${encodeURIComponent(current)}`, { new_name: n }); setCurrent(n); refresh(r.message) }
    catch (e) { flash(e.message) }
  }
  const remove = async () => {
    try { const r = await api.del(`/api/admin/kbs/${encodeURIComponent(current)}`); setCurrent(''); refresh(r.message) }
    catch (e) { flash(e.message) }
  }

  const cur = kbs.find((k) => k.name === current)

  return (
    <>
      <header className="top">
        <div>
          <h1>知識庫</h1>
          <div className="crumb">
            {stats && `${stats.documents} 份文件 · ${stats.chunks} 個切片 · 最後索引 ${stats.last_indexed}`}
          </div>
        </div>
        {msg && <div className="note accent" style={{ fontSize: 12 }}>{msg}</div>}
      </header>

      <div className="page">
        <div className="stages">
          {kbs.map((k) => (
            <div key={k.name} className={`stage ${current === k.name ? 'on' : ''}`} onClick={() => setCurrent(k.name)}>
              <b>{k.label}</b><span>{k.is_general ? '根目錄的檔案' : '子資料夾'}</span><i>{k.doc_count} 份</i>
            </div>
          ))}
          {isAdmin && (
            <div className="stage" style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <input className="inp" placeholder="新知識庫名稱" value={newName}
                     onChange={(e) => setNewName(e.target.value)}
                     onKeyDown={(e) => e.key === 'Enter' && create()} style={{ fontSize: 12, width: 130 }} />
              <button className="btn" onClick={create}>＋ 建立</button>
            </div>
          )}
        </div>

        <div className="panel">
          <div className="phead">
            <b>{cur ? cur.label : '通用'}</b><span className="pill">{docs.length}</span>
            {isAdmin && current && (
              <span style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
                <button className="btn" onClick={rename}>改名</button>
                <button className="btn" onClick={remove} disabled={docs.length > 0}
                        title={docs.length > 0 ? '裡面還有文件，先搬走才能刪' : ''}>刪除</button>
              </span>
            )}
          </div>
          <div style={{ padding: '9px 13px 0', fontSize: 11.5, color: 'var(--ink-3)' }}>
            {current
              ? <>知識庫根目錄下的 <code>{current}/</code> 資料夾。用檔案總管把檔案放進去、再按「增量更新」，也算歸類。</>
              : <>直接放在知識庫根目錄的檔案。不確定該歸哪裡的就放這裡，之後隨時可以搬。</>}
          </div>
          {docs.length === 0
            ? <div className="empty">這個知識庫目前沒有已索引的文件</div>
            : docs.map((d) => (
                <DocRow key={d['路徑']} d={d} picked={picked ? picked['路徑'] : null} onPick={setPicked}
                        isAdmin={isAdmin} kbs={kbs} root={root} onMoved={refresh} />
              ))}
        </div>
        {picked && <DocReader path={picked['路徑']} name={picked['文件名稱']} />}
      </div>
    </>
  )
}
