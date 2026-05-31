import { useEffect, useRef } from 'react'
import { ChatMessage } from './ChatMessage'
import type { Message } from '../types'

interface Props {
  messages: Message[]
  isLoading: boolean
}

export function ChatWindow({ messages, isLoading }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  return (
    <div className="chat-window">
      {messages.length === 0 && (
        <div className="chat-window__empty">
          <h2>Quantum</h2>
          <p>An AI built from scratch. Ask me anything.</p>
        </div>
      )}
      {messages.map(msg => (
        <ChatMessage key={msg.id} message={msg} />
      ))}
      {isLoading && (
        <div className="message message--assistant">
          <div className="message__bubble">
            <span className="message__typing">●●●</span>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  )
}