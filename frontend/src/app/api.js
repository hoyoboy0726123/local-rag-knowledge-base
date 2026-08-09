/* 與後端溝通的唯一入口。
 *
 * token 存在 sessionStorage：關掉分頁就登出，符合區網共用的情境
 * （共用電腦上不該把登入狀態留給下一個人）。
 */

const TOKEN_KEY = 'kb_token'

export const getToken = () => sessionStorage.getItem(TOKEN_KEY)
export const setToken = (t) => sessionStorage.setItem(TOKEN_KEY, t)
export const clearToken = () => sessionStorage.removeItem(TOKEN_KEY)

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) }
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`
  if (options.body && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
  }

  const res = await fetch(path, { ...options, headers })

  if (res.status === 401) {
    clearToken()
    window.location.hash = '#/login'
    throw new Error('登入已過期，請重新登入')
  }
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.detail || `請求失敗（${res.status}）`)
  }
  return res.status === 204 ? null : res.json()
}

export const api = {
  get: (p) => request(p),
  post: (p, body) => request(p, { method: 'POST', body: body ? JSON.stringify(body) : undefined }),
  put: (p, body) => request(p, { method: 'PUT', body: JSON.stringify(body) }),
  del: (p) => request(p, { method: 'DELETE' }),
  upload: (p, formData) => request(p, { method: 'POST', body: formData }),
}

/* SSE 串流。
 *
 * 用 fetch + ReadableStream 而不是 EventSource，因為 EventSource
 * 只能發 GET 且無法帶 Authorization 標頭。
 *
 * 事件格式：`event: <名稱>\n` + `data: <JSON>\n\n`
 * **結尾兩個換行是分隔符，少一個就整包解析不出來。**
 */
export async function stream(path, body, onEvent) {
  const res = await fetch(path, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${getToken()}`,
    },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`串流建立失敗（${res.status}）`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // 一則事件以空行結束；最後一段可能不完整，留在 buffer 等下一輪
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop()

    for (const block of blocks) {
      let name = 'message'
      let data = ''
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) name = line.slice(7)
        else if (line.startsWith('data: ')) data = line.slice(6)
      }
      if (data) {
        try { onEvent(name, JSON.parse(data)) } catch { /* 忽略不完整的片段 */ }
      }
    }
  }
}
