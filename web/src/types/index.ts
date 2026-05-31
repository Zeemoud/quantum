export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export interface ChatRequest {
  message: string
  max_new_tokens?: number
  temperature?: number
  top_k?: number
  top_p?: number
  repetition_penalty?: number
}

export interface ChatResponse {
  response: string
  model_loaded: boolean
}