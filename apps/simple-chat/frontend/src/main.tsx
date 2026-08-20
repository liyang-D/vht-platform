import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import HistoryPage from './HistoryPage.tsx'

const basePath = import.meta.env.BASE_URL.replace(/\/$/, '')
const currentPath = window.location.pathname.replace(/\/$/, '')
const page = currentPath === `${basePath}/history` ? <HistoryPage /> : <App />

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {page}
  </StrictMode>,
)
