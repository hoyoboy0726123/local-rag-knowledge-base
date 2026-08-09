import { useState } from 'react'
import { api } from '../app/api'

export default function Profile({ user }) {
  const [old, setOld] = useState('')
  const [neu, setNeu] = useState('')
  const [msg, setMsg] = useState(null)

  const submit = async (e) => {
    e.preventDefault()
    try {
      const r = await api.post('/api/auth/password', { old_password: old, new_password: neu })
      setMsg({ ok: r.message }); setOld(''); setNeu('')
    } catch (e2) { setMsg({ error: e2.message }) }
  }

  return (
    <>
      <header className="top"><div><h1>個人設定</h1></div></header>
      <div className="page" style={{ maxWidth: 520 }}>
        <div className="panel">
          <div className="phead"><b>{user.display_name}　{user.username}</b></div>
          <form style={{ padding: 15 }} onSubmit={submit}>
            <div className="field"><label>目前密碼</label>
              <input type="password" value={old} onChange={(e) => setOld(e.target.value)} /></div>
            <div className="field"><label>新密碼</label>
              <input type="password" value={neu} onChange={(e) => setNeu(e.target.value)} /></div>
            {msg?.ok && <div className="note accent">{msg.ok}</div>}
            {msg?.error && <div className="note danger">{msg.error}</div>}
            <button className="btn pri" style={{ marginTop: 10 }} disabled={!old || !neu}>更新密碼</button>
          </form>
        </div>
      </div>
    </>
  )
}
