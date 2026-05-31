import { useState, useCallback } from 'react'
import type { Message, ChatRequest } from '../types'

async function streamMessage(
  request: ChatRequest,
  onToken: (token: string) => void,
): Promise<void> {
  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  if (!res.body) throw new Error('No response body')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    const chunk = decoder.decode(value, { stream: true })
    const lines = chunk.split('\n')

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const token = line.slice(6)
      if (token === '[DONE]') return
      if (token.startsWith('[ERROR]')) throw new Error(token)
      onToken(token)
    }
  }
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [temperature, setTemperature] = useState(0.8)
  const [topP, setTopP] = useState(0.9)

  const send = useCallback(async (content: string) => {
    if (!content.trim() || isLoading) return

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    }

    const assistantMessage: Message = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage, assistantMessage])
    setIsLoading(true)
    setError(null)

    try {
      await streamMessage(
        { message: content.trim(), temperature, top_p: topP },
        (token) => {
          setMessages(prev => prev.map(m =>
            m.id === assistantMessage.id
              ? { ...m, content: m.content + token }
              : m
          ))
        }
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setMessages(prev => prev.filter(m => m.id !== assistantMessage.id))
    } finally {
      setIsLoading(false)
    }
  }, [isLoading, temperature, topP])

  const clear = useCallback(() => {
    setMessages([])
    setError(null)
  }, [])

  return { messages, isLoading, error, send, clear, temperature, setTemperature, topP, setTopP }
}