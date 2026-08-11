import { useEffect, useMemo, useRef, useState } from 'react'

type ChatRole = 'user' | 'assistant' | 'system'

type ChatMessage = {
  id: string
  role: ChatRole
  content: string
}

type StreamEvent = {
  type?: string
  message?: string
  [key: string]: unknown
}

const initialPrompt = 'Summarize a chicken dinner with a quick ingredient list.'

function makeId() {
  return crypto.randomUUID()
}

function formatEvent(event: StreamEvent) {
  if (typeof event.message === 'string' && event.message.trim()) {
    return event.message
  }

  return JSON.stringify(event, null, 2)
}

export default function App() {
  const [prompt, setPrompt] = useState(initialPrompt)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [eventLog, setEventLog] = useState<string[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)

  const statusLabel = useMemo(() => {
    if (error) {
      return 'Offline'
    }

    return isStreaming ? 'Streaming' : 'Ready'
  }, [error, isStreaming])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, eventLog])

  useEffect(() => {
    return () => {
      abortRef.current?.abort()
    }
  }, [])

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()

    const trimmedPrompt = prompt.trim()
    if (!trimmedPrompt || isStreaming) {
      return
    }

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    const userMessage: ChatMessage = {
      id: makeId(),
      role: 'user',
      content: trimmedPrompt,
    }
    const assistantId = makeId()

    setError(null)
    setIsStreaming(true)
    setEventLog([])
    setMessages((current) => [
      ...current,
      userMessage,
      { id: assistantId, role: 'assistant', content: '' },
    ])

    try {
      const response = await fetch('/api/recipes/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({ user_request: trimmedPrompt }),
        signal: controller.signal,
      })

      if (!response.ok || !response.body) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) {
          break
        }

        buffer += decoder.decode(value, { stream: true })

        let splitIndex = buffer.indexOf('\n\n')
        while (splitIndex !== -1) {
          const rawEvent = buffer.slice(0, splitIndex).trim()
          buffer = buffer.slice(splitIndex + 2)
          splitIndex = buffer.indexOf('\n\n')

          if (!rawEvent) {
            continue
          }

          const event = parseSseEvent(rawEvent)
          if (!event) {
            continue
          }

          setEventLog((current) => [formatEvent(event), ...current].slice(0, 8))
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    content: appendAssistantContent(message.content, event),
                  }
                : message,
            ),
          )
        }
      }
    } catch (streamError) {
      if (controller.signal.aborted) {
        return
      }

      const message = streamError instanceof Error ? streamError.message : 'Stream failed.'
      setError(message)
      setMessages((current) =>
        current.map((chatMessage) =>
          chatMessage.role === 'assistant' && chatMessage.content === ''
            ? { ...chatMessage, content: `Error: ${message}` }
            : chatMessage,
        ),
      )
    } finally {
      setIsStreaming(false)
      abortRef.current = null
    }
  }

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Remy</p>
          <h1>Streaming recipe chat for the FastAPI agent.</h1>
          <p className="lede">
            Send a prompt and watch the backend stream progress events back into the
            conversation.
          </p>
        </div>

        <div className="status-card">
          <span className={`status ${statusLabel.toLowerCase()}`}>{statusLabel}</span>
          <p>
            Endpoint: <strong>/api/recipes/stream</strong>
          </p>
          <p>Transport: Server-sent events</p>
        </div>
      </section>

      <section className="workspace">
        <div className="chat-panel">
          <div className="messages">
            {messages.length === 0 ? (
              <div className="empty-state">
                <h2>Start the stream</h2>
                <p>Ask for a recipe, a summary, or any ingredient-driven request.</p>
              </div>
            ) : (
              messages.map((message) => (
                <article key={message.id} className={`message ${message.role}`}>
                  <span>{message.role}</span>
                  <p>{message.content || '…'}</p>
                </article>
              ))
            )}
            <div ref={endRef} />
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <label htmlFor="prompt">Prompt</label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              rows={4}
              placeholder="Describe the recipe task you want the agent to stream..."
            />
            <div className="actions">
              <button type="submit" disabled={isStreaming}>
                {isStreaming ? 'Streaming…' : 'Send'}
              </button>
              <button
                type="button"
                className="secondary"
                onClick={() => {
                  abortRef.current?.abort()
                  setIsStreaming(false)
                }}
                disabled={!isStreaming}
              >
                Stop
              </button>
            </div>
            {error ? <p className="error">{error}</p> : null}
          </form>
        </div>

        <aside className="event-panel">
          <h2>Live events</h2>
          <div className="events">
            {eventLog.length === 0 ? (
              <p className="muted">Stream events will appear here.</p>
            ) : (
              eventLog.map((event, index) => (
                <pre key={`${index}-${event.slice(0, 16)}`}>{event}</pre>
              ))
            )}
          </div>
        </aside>
      </section>
    </main>
  )
}

function parseSseEvent(rawEvent: string): StreamEvent | null {
  const lines = rawEvent.split('\n')
  const dataLines = lines
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())

  if (dataLines.length === 0) {
    return null
  }

  const data = dataLines.join('\n')

  try {
    return JSON.parse(data) as StreamEvent
  } catch {
    return { message: data }
  }
}

function appendAssistantContent(currentContent: string, event: StreamEvent) {
  const fragment = formatEvent(event)

  if (!fragment) {
    return currentContent
  }

  return currentContent ? `${currentContent}\n${fragment}` : fragment
}