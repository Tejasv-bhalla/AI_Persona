import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown, { Components } from "react-markdown";
import "./styles.css";

type Role = "user" | "assistant";

type Message = {
  id: string;
  role: Role;
  content: string;
  grounded?: boolean;
  /** True once the backend streamed a regenerated answer over the original one. */
  corrected?: boolean;
  available_slots?: string[];
};

type ChatEvent = {
  type: "token" | "correction" | "done" | "error" | "meta";
  data: string;
  session_id?: string;
  grounded?: boolean;
  available_slots?: string[];
};

/** The backend only consumes the last 6 turns; cap the payload so it stays bounded. */
const MAX_HISTORY_TURNS = 10;

const rawApiUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const apiBaseUrl = rawApiUrl.endsWith("/") ? rawApiUrl.slice(0, -1) : rawApiUrl;

function makeId(): string {
  return crypto.randomUUID();
}

const markdownComponents: Components = {
  pre: ({ children }) => <pre className="code-block">{children}</pre>,
};

function Markdown({ content }: { content: string }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown components={markdownComponents}>{content}</ReactMarkdown>
    </div>
  );
}

function BookingForm({ slotTime, onClose }: { slotTime: string; onClose: () => void }) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [notes, setNotes] = useState("");
  const [status, setStatus] = useState<"idle" | "submitting" | "success" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");

  async function handleBook(e: FormEvent) {
    e.preventDefault();
    if (!name.trim() || !email.trim()) return;
    setStatus("submitting");
    try {
      const res = await fetch(`${apiBaseUrl}/book`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          preferred_time: slotTime,
          attendee_name: name,
          attendee_email: email,
          notes: notes,
        }),
      });
      const data = await res.json();
      if (data.status === "success") {
        setStatus("success");
      } else {
        setStatus("error");
        setErrorMsg(data.message || "Something went wrong.");
      }
    } catch {
      setStatus("error");
      setErrorMsg("Failed to connect to server.");
    }
  }

  const formattedDate = useMemo(() => {
    try {
      return new Date(slotTime).toLocaleString([], {
        weekday: "long",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return slotTime;
    }
  }, [slotTime]);

  if (status === "success") {
    return (
      <div className="booking-status success">
        <p>🎉 Booking confirmed! Thank you.</p>
        <button onClick={onClose} className="close-btn">
          Close
        </button>
      </div>
    );
  }

  return (
    <form onSubmit={handleBook} className="booking-form">
      <h4>Confirm Booking:</h4>
      <p className="selected-slot-time">📅 {formattedDate}</p>
      {status === "error" && <p className="error-text">❌ {errorMsg}</p>}
      <input
        placeholder="Your Name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        required
        disabled={status === "submitting"}
      />
      <input
        type="email"
        placeholder="Your Email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        required
        disabled={status === "submitting"}
      />
      <textarea
        placeholder="Notes (optional)"
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        disabled={status === "submitting"}
      />
      <div className="form-actions">
        <button
          type="button"
          onClick={onClose}
          disabled={status === "submitting"}
          className="cancel-btn"
        >
          Cancel
        </button>
        <button type="submit" disabled={status === "submitting" || !name.trim() || !email.trim()}>
          {status === "submitting" ? "Booking..." : "Confirm"}
        </button>
      </div>
    </form>
  );
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: makeId(),
      role: "assistant",
      content:
        "Hey, I’m Tejasv’s grounded AI persona. Ask me about projects, experience, technical decisions, or schedule a call.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedSlot, setSelectedSlot] = useState<{ slot: string; messageId: string } | null>(
    null
  );
  const sessionId = useMemo(makeId, []);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    fetch(`${apiBaseUrl}/warm`).catch(() => undefined);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text || isStreaming) return;

    const assistantId = makeId();
    const history = messages
      .filter((m) => m.content && m.content !== "Thinking…")
      .slice(-MAX_HISTORY_TURNS)
      .map((m) => ({ role: m.role, content: m.content }));

    const controller = new AbortController();
    abortRef.current = controller;
    let correctionStarted = false;

    setInput("");
    setIsStreaming(true);
    setMessages((current) => [
      ...current,
      { id: makeId(), role: "user", content: text },
      { id: assistantId, role: "assistant", content: "" },
    ]);

    try {
      const response = await fetch(`${apiBaseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          session_id: sessionId,
          conversation_history: history,
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`Request failed with ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const rawEvent of events) {
          const line = rawEvent.split("\n").find((item) => item.startsWith("data: "));
          if (!line) continue;
          const parsed = JSON.parse(line.slice(6)) as ChatEvent;
          if (parsed.type === "token") {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, content: message.content + parsed.data }
                  : message
              )
            );
          } else if (parsed.type === "correction") {
            // The first correction token means grounding failed and a regenerated
            // answer is now streaming: drop what we have and start over.
            const isFirstCorrectionToken = !correctionStarted;
            correctionStarted = true;
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: (isFirstCorrectionToken ? "" : message.content) + parsed.data,
                      corrected: true,
                    }
                  : message
              )
            );
          } else if (parsed.type === "done") {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      grounded: parsed.grounded ?? true,
                      available_slots: parsed.available_slots ?? [],
                    }
                  : message
              )
            );
          } else if (parsed.type === "error") {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, content: parsed.data }
                  : message
              )
            );
          }
        }
      }
    } catch {
      // An intentional stop is not an error: keep the partial answer on screen.
      if (!controller.signal.aborted) {
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  content:
                    "Could not connect to the server. The backend may be booting up or temporarily offline. Please try again in a moment.",
                }
              : message
          )
        );
      }
    } finally {
      abortRef.current = null;
      setIsStreaming(false);
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">RAG-grounded persona</p>
        <h1>Tejasv Bhalla</h1>
        <p>
          A production-style portfolio chatbot grounded in indexed resume, GitHub, changelog,
          and contribution-scope evidence.
        </p>
      </section>

      <section className="chat">
        <div className="messages">
          {messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              <div className="message-header">
                <span>{message.role === "user" ? "You" : "Persona"}</span>
                {message.grounded === false && !message.corrected && (
                  <span
                    className="badge warning"
                    title="The grounding check flagged this answer — it may not be fully grounded in the indexed sources."
                  >
                    ⚠ low confidence
                  </span>
                )}
              </div>
              <div className="message-body">
                {message.content ? (
                  <Markdown content={message.content} />
                ) : (
                  <p className="thinking">Thinking…</p>
                )}
              </div>
              {message.available_slots && message.available_slots.length > 0 && (
                <div className="slots-container">
                  <p className="slots-title">Available booking slots:</p>
                  <div className="slots-grid">
                    {message.available_slots.map((slot) => {
                      const date = new Date(slot);
                      const timeStr = date.toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      });
                      const dateStr = date.toLocaleDateString([], {
                        month: "short",
                        day: "numeric",
                      });
                      return (
                        <button
                          key={slot}
                          className="slot-btn"
                          onClick={() => setSelectedSlot({ slot, messageId: message.id })}
                        >
                          {timeStr} ({dateStr})
                        </button>
                      );
                    })}
                  </div>
                  {selectedSlot && selectedSlot.messageId === message.id && (
                    <BookingForm
                      slotTime={selectedSlot.slot}
                      onClose={() => setSelectedSlot(null)}
                    />
                  )}
                </div>
              )}
            </article>
          ))}
          <div ref={bottomRef} />
        </div>

        <form onSubmit={submit} className="composer">
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask about Tejasv’s projects, timeline, skills, or availability…"
            disabled={isStreaming}
          />
          {isStreaming ? (
            <button type="button" onClick={stop}>
              Stop
            </button>
          ) : (
            <button type="submit" disabled={!input.trim()}>
              Ask
            </button>
          )}
        </form>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
