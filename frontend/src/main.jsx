import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter, Navigate, Route, Routes } from 'react-router-dom'
import './styles/tokens.css'
import './styles/app.css'
import { api, getToken } from './app/api'
import Layout from './app/Layout'
import Login from './features/Login'
import Chat from './features/Chat'
import KnowledgeBases from './features/KnowledgeBases'
import Profile from './features/Profile'
import Library from './features/Library'
import Models from './features/Models'
import Users from './features/Users'

function App() {
  const [user, setUser] = useState(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    if (!getToken()) return setReady(true)
    api.get('/api/auth/me').then(setUser).catch(() => {}).finally(() => setReady(true))
  }, [])

  if (!ready) return null
  if (!user) return <Login onLogin={setUser} />

  const admin = user.role === 'ADMIN'
  return (
    <Routes>
      <Route element={<Layout user={user} onLogout={() => setUser(null)} />}>
        <Route path="/chat" element={<Chat />} />
        <Route path="/kbs" element={<KnowledgeBases />} />
        {/* 舊的 /history 網址由下方的 * 導回 /chat——對話紀錄已併入問答頁 */}
        <Route path="/profile" element={<Profile user={user} />} />
        {/* 管理路由只在管理員登入時掛載。
            這是體驗層的處理——後端每個管理端點都另外驗證角色。 */}
        {admin && <Route path="/admin/library" element={<Library />} />}
        {admin && <Route path="/admin/models" element={<Models />} />}
        {admin && <Route path="/admin/users" element={<Users />} />}
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Route>
    </Routes>
  )
}

createRoot(document.getElementById('root')).render(
  <React.StrictMode><HashRouter><App /></HashRouter></React.StrictMode>
)
