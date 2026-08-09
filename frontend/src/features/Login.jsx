import { useState } from 'react'
import { api, setToken } from '../app/api'

export default function Login({ onLogin }) {
  const [u, setU] = useState('')
  const [p, setP] = useState('')
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setErr(''); setBusy(true)
    try {
      const d = await api.post('/api/auth/login', { username: u, password: p })
      setToken(d.token)
      onLogin(d.user)
    } catch (e2) {
      setErr(e2.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="login">
      <form className="card" onSubmit={submit}>
        <div className="mark" style={{ width: 34, height: 34, fontSize: 16 }}>R</div>
        <h2>本機知識庫</h2>
        <p className="sub">所有 AI 運算都在本機完成，研發資料不會離開內網</p>

        <div className="field">
          <label>帳號</label>
          <input value={u} onChange={(e) => setU(e.target.value)} autoFocus placeholder="admin" />
        </div>
        <div className="field">
          <label>密碼</label>
          <input type="password" value={p} onChange={(e) => setP(e.target.value)} />
        </div>
        {err && <div className="note danger" style={{ marginTop: 4 }}>{err}</div>}
        <button className="btn pri" disabled={busy || !u || !p}>{busy ? '登入中…' : '登入'}</button>

        <div className="demo">
          展示帳號（密碼皆為 <code>demo1234</code>）<br />
          <code>admin</code> 知識庫管理員　<code>user01</code> / <code>user02</code> 一般使用者
        </div>
      </form>
    </div>
  )
}
