import ReactMarkdown from 'react-markdown'
import type { Message } from '../types'

interface Props {
  message: Message
}

export function ChatMessage({ message }: Props) {
  const isUser = message.role === 'user'

  return (
    <div className={`message ${isUser ? 'message--user' : 'message--assistant'}`}>
      <div className="message__bubble">
        <div className="message__content">
          {isUser
            ? <p>{message.content}</p>
            : <ReactMarkdown>{message.content || '…'}</ReactMarkdown>
          }
        </div>
        <span className="message__time">
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  )
}