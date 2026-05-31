import { useChat } from './hooks/useChat'
import { ChatWindow } from './components/ChatWindow'
import { ChatInput } from './components/ChatInput'
import './App.css'

export default function App() {
  const {
    messages, isLoading, error, send, clear,
    temperature, setTemperature,
    topP, setTopP,
  } = useChat()

  return (
    <div className="app">
      <header className="app__header">
        <span className="app__logo">Quantum</span>
        <div className="app__controls">
          <label className="app__slider">
            <span>Temp {temperature.toFixed(1)}</span>
            <input
              type="range" min="0.1" max="2" step="0.1"
              value={temperature}
              onChange={e => setTemperature(parseFloat(e.target.value))}
            />
          </label>
          <label className="app__slider">
            <span>Top-p {topP.toFixed(1)}</span>
            <input
              type="range" min="0.1" max="1" step="0.05"
              value={topP}
              onChange={e => setTopP(parseFloat(e.target.value))}
            />
          </label>
          <button className="app__clear" onClick={clear} disabled={messages.length === 0}>
            Clear
          </button>
        </div>
      </header>

      {error && <div className="app__error">{error}</div>}

      <ChatWindow messages={messages} isLoading={isLoading} />
      <ChatInput onSend={send} isLoading={isLoading} />
    </div>
  )
}