import { useState, useRef, useEffect } from 'react'
import { chat } from '../api'
import { getLocalChatReply } from '../utils/chatKnowledge'

const SUGGESTIONS = [
  'How do I analyze a chest X-ray?',
  'What is Grad-CAM?',
  'Explain confidence scores',
  'Which modalities are supported?',
]

export default function ChatAssistant({ studyContext = null }) {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        "Hi! I'm your Radiology AI assistant. Ask me about findings, Grad-CAM, modalities, reports, or how to use this platform.",
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, open, loading])

  useEffect(() => {
    if (open) inputRef.current?.focus()
  }, [open])

  const sendMessage = async (text) => {
    const trimmed = text.trim()
    if (!trimmed || loading) return

    setMessages((m) => [...m, { role: 'user', content: trimmed }])
    setInput('')
    setLoading(true)

    let reply = ''

    try {
      const token = localStorage.getItem('token')
      if (token) {
        const { data } = await chat.send(trimmed, studyContext)
        reply = data?.reply?.trim() || ''
      } else {
        const { data } = await chat.sendPublic(trimmed, studyContext)
        reply = data?.reply?.trim() || ''
      }
    } catch (err) {
      const status = err?.response?.status
      if (status === 401) {
        reply = 'Your session expired. Please log out and sign in again. Meanwhile: ' + getLocalChatReply(trimmed, studyContext)
      }
    }

    if (!reply) {
      reply = getLocalChatReply(trimmed, studyContext)
    }

    setMessages((m) => [...m, { role: 'assistant', content: reply }])
    setLoading(false)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    sendMessage(input)
  }

  return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-[90] bg-slate-900/20 md:hidden"
          onClick={() => setOpen(false)}
          aria-hidden="true"
        />
      )}

      <div className="fixed bottom-6 right-6 z-[100] flex flex-col items-end gap-3 pointer-events-none">
        {open && (
          <div className="pointer-events-auto w-[min(100vw-2rem,380px)] h-[min(70vh,520px)] flex flex-col rounded-2xl border border-slate-200 bg-white shadow-2xl shadow-slate-300/50 overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100 bg-gradient-to-r from-cyan-50 to-violet-50">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-500 to-violet-500 flex items-center justify-center text-white text-sm shadow-md">
                  AI
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-900">Radiology Assistant</p>
                  <p className="text-[10px] text-emerald-600 font-medium">● Online</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
                aria-label="Close chat"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-slate-50">
              {messages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div
                    className={`max-w-[90%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed whitespace-pre-wrap break-words ${
                      msg.role === 'user'
                        ? 'bg-gradient-to-br from-cyan-600 to-violet-600 text-white rounded-br-md'
                        : 'bg-white border border-slate-200 text-slate-800 rounded-bl-md shadow-sm'
                    }`}
                  >
                    {msg.content}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
                    <div className="flex items-center gap-1.5 py-0.5">
                      <span className="w-2 h-2 rounded-full bg-cyan-600 opacity-90" />
                      <span className="w-2 h-2 rounded-full bg-cyan-600 opacity-60" />
                      <span className="w-2 h-2 rounded-full bg-cyan-600 opacity-30" />
                    </div>
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>

            {messages.length <= 2 && !loading && (
              <div className="px-3 pb-2 flex flex-wrap gap-1.5 pointer-events-auto">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => sendMessage(s)}
                    className="text-xs px-2.5 py-1 rounded-full border border-slate-200 bg-white text-slate-600 hover:border-cyan-400 hover:text-cyan-700 transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}

            <form onSubmit={handleSubmit} className="p-3 border-t border-slate-100 bg-white flex gap-2 pointer-events-auto">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about radiology AI..."
                className="flex-1 rounded-xl border border-slate-200 px-3 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-100"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="btn-primary px-4 py-2.5 disabled:opacity-40 shrink-0"
                aria-label="Send message"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </form>
          </div>
        )}

        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className={`pointer-events-auto flex items-center gap-2 rounded-full shadow-lg transition-colors ${
            open
              ? 'bg-slate-800 text-white px-4 py-3'
              : 'bg-gradient-to-r from-cyan-600 to-violet-600 text-white px-5 py-3 hover:opacity-95'
          }`}
          aria-label={open ? 'Close assistant' : 'Open AI assistant'}
        >
          {open ? (
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                />
              </svg>
              <span className="text-sm font-semibold">AI Assistant</span>
            </>
          )}
        </button>
      </div>
    </>
  )
}
