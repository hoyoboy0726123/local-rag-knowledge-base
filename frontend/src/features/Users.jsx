import { useEffect, useState } from 'react'
import { api } from '../app/api'

export default function Users() {
  const [rows, setRows] = useState([])
  useEffect(() => { api.get('/api/admin/users').then((d) => setRows(d.users)).catch(() => setRows([])) }, [])
  return (
    <>
      <header className="top">
        <div><h1>帳號管理</h1>
          <div className="crumb">管理員可建立與停用帳號，但看不到任何人的對話內容</div></div>
      </header>
      <div className="page" style={{ maxWidth: 820 }}>
        <div className="panel">
          <div className="phead"><b>帳號</b><span className="pill">{rows.length}</span></div>
          {rows.length === 0 ? <div className="empty">載入中或尚無資料</div> : (
            <table className="tbl">
              <thead><tr><th>帳號</th><th>顯示名稱</th><th>角色</th><th>狀態</th><th>最後登入</th></tr></thead>
              <tbody>{rows.map((u) => (
                <tr key={u.ID}>
                  <td>{u.帳號}</td><td>{u.顯示名稱}</td>
                  <td>{u.角色}</td><td>{u.啟用}</td><td>{u.最後登入}</td>
                </tr>
              ))}</tbody>
            </table>
          )}
        </div>
        <div className="note amber" style={{ marginTop: 14 }}>
          <b>管理員也看不到別人的對話內容。</b>
          對話中可能有 PM 對專案的疑慮或未定案的判斷，屬於個人工作紀錄。
        </div>
      </div>
    </>
  )
}
