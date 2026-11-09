import { createContext, useContext, useState, useEffect } from 'react'

const StudyChatContext = createContext({
  studyContext: null,
  setStudyContext: () => {},
  clearStudyContext: () => {},
})

const STORAGE_KEY = 'radiology_active_study_context'

export function StudyChatProvider({ children }) {
  const [studyContext, setStudyContextState] = useState(() => {
    try {
      const saved = sessionStorage.getItem(STORAGE_KEY)
      return saved ? JSON.parse(saved) : null
    } catch {
      return null
    }
  })

  const setStudyContext = (ctx) => {
    setStudyContextState(ctx)
    try {
      if (ctx) {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(ctx))
      } else {
        sessionStorage.removeItem(STORAGE_KEY)
      }
    } catch {
      // Ignore storage errors
    }
  }

  const clearStudyContext = () => setStudyContext(null)

  return (
    <StudyChatContext.Provider value={{ studyContext, setStudyContext, clearStudyContext }}>
      {children}
    </StudyChatContext.Provider>
  )
}

export function useStudyChat() {
  return useContext(StudyChatContext)
}
