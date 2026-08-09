import { useEffect, useState } from 'react'
import { api, getToken } from '../app/api'

/* 一份文件可以做的三件事，問答頁與階段導覽頁共用。
 *
 * **為什麼不是單純的 <a href>。** 後端走 JWT，token 存在 sessionStorage，
 * 而瀏覽器發出的原生連結請求不會帶 Authorization 標頭 —— 直接 <a> 會 401。
 * 所以一律 fetch 取回 blob 再觸發動作，外觀維持連結／按鈕的樣子。
 * （另一條路是發簽章 URL，但那要多維護一套 token 過期邏輯，
 *   換來的只有「右鍵另存」，不值得。）
 */

/* 只有這些格式給「新分頁開啟」，後端有同一份白名單把關。
 *
 * 排除 Office 檔：瀏覽器無法原生渲染，給了也只會變下載，讓人以為壞掉。
 * 排除 .svg / .html：blob URL 會**繼承本頁來源**，這兩種格式裡的 <script>
 *   會在應用的 origin 下執行，讀得到 sessionStorage 裡的 JWT。
 * 排除 .md / .txt：blob URL 下瀏覽器常忽略 charset，改用系統預設編碼，
 *   中文會整片變亂碼。這類檔案走「開啟完整文件」才對——那邊有正確渲染。 */
const INLINE_EXT = ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']
const canInline = (name = '') => INLINE_EXT.some((e) => name.toLowerCase().endsWith(e))

/* 後端是否與瀏覽器在同一台機器上。
 *
 * 只問一次就快取起來：這個值在一次工作階段內不會變，
 * 而來源面板每則回答都會重新渲染，每次都打一發沒有意義。 */
let localCache = null
export function useIsLocal() {
  const [isLocal, setIsLocal] = useState(localCache ?? false)
  useEffect(() => {
    if (localCache !== null) return
    api.get('/api/client/is-local')
      .then((d) => { localCache = !!d.is_local; setIsLocal(localCache) })
      .catch(() => { localCache = false })
  }, [])
  return isLocal
}

async function fetchBlob(path, inline) {
  const url = `/api/documents/download?path=${encodeURIComponent(path)}${inline ? '&inline=true' : ''}`
  const res = await fetch(url, { headers: { Authorization: `Bearer ${getToken()}` } })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.detail || `取得檔案失敗（${res.status}）`)
  }
  return res.blob()
}

export default function DocActions({ path, name, onOpenDoc }) {
  const isLocal = useIsLocal()
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')

  const run = async (kind, fn) => {
    setBusy(kind); setError('')
    try { await fn() } catch (e) { setError(e.message) } finally { setBusy('') }
  }

  const download = () => run('dl', async () => {
    const blob = await fetchBlob(path, false)
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = name
    a.click()
    URL.revokeObjectURL(a.href)
  })

  /* 先開空白分頁再塞內容。
   * 若等 await 回來才 window.open()，已經脫離使用者手勢，會被彈出視窗封鎖擋掉。 */
  const openTab = () => {
    const win = window.open('', '_blank')
    run('tab', async () => {
      try {
        const blob = await fetchBlob(path, true)
        const url = URL.createObjectURL(blob)
        if (win) win.location = url
        else window.open(url, '_blank')
        setTimeout(() => URL.revokeObjectURL(url), 60_000)
      } catch (e) {
        if (win) win.close()
        throw e
      }
    })
  }

  const openLocal = () => run('local', async () => {
    await api.post('/api/documents/open', { path })
  })

  return (
    <div className="docacts">
      {onOpenDoc && (
        <button className="lnk" onClick={onOpenDoc} title="在這裡直接顯示整份文件，不會下載">
          📄 開啟完整文件
        </button>
      )}
      {canInline(name) && (
        <button className="lnk" onClick={openTab} disabled={busy === 'tab'}
                title="用瀏覽器內建檢視器開新分頁，不會存檔">
          {busy === 'tab' ? '開啟中…' : '🔗 在新分頁開啟'}
        </button>
      )}
      <button className="lnk" onClick={download} disabled={busy === 'dl'}
              title="把原始檔存到你的電腦">
        {busy === 'dl' ? '下載中…' : '⬇ 下載原始檔'}
      </button>
      {isLocal && (
        <button className="lnk" onClick={openLocal} disabled={busy === 'local'}
                title="用這台電腦的預設程式開啟（例如 PowerPoint、Excel）">
          {busy === 'local' ? '開啟中…' : '🖥 用本機程式開啟'}
        </button>
      )}
      {error && <span className="docacts-err">{error}</span>}
    </div>
  )
}
