import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { api, clearToken } from './api'

/* 對話紀錄**不在這裡**——它併在問答頁的左側清單，
   跟一般聊天介面一樣可以直接點回去接續。
   獨立成一頁只能「看」不能「續」，那是把紀錄當成報表在做。 */
const WORK = [
  { to: '/chat', icon: '◈', label: 'AI 問答' },
  { to: '/kbs', icon: '◱', label: '知識庫' },
  { to: '/profile', icon: '⚙', label: '個人設定' },
]
const ADMIN = [
  { to: '/admin/library', icon: '⌸', label: '知識庫維護' },
  { to: '/admin/models', icon: '◎', label: '模型與設定' },
  { to: '/admin/users', icon: '⚇', label: '帳號管理' },
]

/* 收折狀態記在 localStorage 而不是 sessionStorage。
   token 用 sessionStorage 是安全考量（共用電腦不留登入狀態），
   但版面偏好沒有這個顧慮，每次登入都要重收一次才是煩人。 */
const SLIM_KEY = 'kb_side_slim'

export default function Layout({ user, onLogout }) {
  const [engine, setEngine] = useState(null)
  const [slim, setSlim] = useState(() => localStorage.getItem(SLIM_KEY) === '1')
  const navigate = useNavigate()

  useEffect(() => {
    api.get('/api/chat/engine').then(setEngine).catch(() => setEngine({ alive: false, message: '無法連線' }))
  }, [])

  useEffect(() => { localStorage.setItem(SLIM_KEY, slim ? '1' : '0') }, [slim])

  const logout = () => {
    clearToken()
    onLogout()
    navigate('/login')
  }

  const isAdmin = user.role === 'ADMIN'

  // 收折後只剩圖示，滑鼠停留才知道是什麼——所以 title 一定要給
  const link = (i) => (
    <NavLink key={i.to} to={i.to} title={slim ? i.label : undefined}
             className={({ isActive }) => (isActive ? 'on' : '')}>
      <span className="ico">{i.icon}</span>
      <span className="lbl">{i.label}</span>
    </NavLink>
  )

  return (
    <div className={`app ${slim ? 'slim' : ''}`}>
      <aside className="side">
        <div className="brand">
          <div className="mark">R</div>
          <div className="lbl"><b>本機知識庫</b><span>文件不離開內網</span></div>
          <button className="collapse" onClick={() => setSlim((v) => !v)}
                  title={slim ? '展開側邊欄' : '收折側邊欄'}>
            {slim ? '»' : '«'}
          </button>
        </div>

        <div>
          <div className="navlbl">工作區</div>
          <nav className="nav">{WORK.map(link)}</nav>
        </div>

        {/* 管理選單僅管理員可見。
            這只是體驗——後端每個管理端點都另外驗證角色，
            前端隱藏可以被繞過（改 URL 就行）。 */}
        {isAdmin && (
          <div>
            <div className="navlbl">管理</div>
            <nav className="nav">{ADMIN.map(link)}</nav>
          </div>
        )}

        <div className="side-foot">
          {/* 引擎狀態必須隨時看得到，否則使用者不會知道問答為什麼壞了。
              收折後文字沒了，狀態點本身就得帶 title。 */}
          <div className="engine" title={engine?.alive ? '本機引擎就緒' : '引擎未就緒'}>
            <span className={`dot ${engine?.alive ? '' : 'off'}`} />
            <span className="lbl">
              {engine ? (engine.alive ? '本機引擎就緒' : '引擎未就緒') : '檢查中…'}
            </span>
          </div>
          <div className="who" title={`${user.display_name}（${user.username}）`}>
            <div className="av">{user.display_name?.[0] || user.username[0]}</div>
            <div className="lbl" style={{ minWidth: 0, flex: 1 }}>
              <b>{user.display_name}{isAdmin && <span className="badge">ADMIN</span>}</b>
              <span>{user.username}</span>
            </div>
          </div>
          <button className="btn logout" onClick={logout} title="登出">
            <span className="lbl">登出</span>
            <span className="ico-only">⏻</span>
          </button>
        </div>
      </aside>

      <div className="main"><Outlet /></div>
    </div>
  )
}
