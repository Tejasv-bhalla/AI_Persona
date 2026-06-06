import React, { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Role = "user" | "assistant";

type Message = {
  id: string;
  role: Role;
  content: string;
  grounded?: boolean;
  available_slots?: string[];
};

type ChatEvent = {
  type: "token" | "done" | "error" | "meta";
  data: string;
  session_id?: string;
  grounded?: boolean;
  available_slots?: string[];
};

const rawApiUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const apiBaseUrl = rawApiUrl.endsWith("/") ? rawApiUrl.slice(0, -1) : rawApiUrl;

function makeId(): string {
  return crypto.randomUUID();
}

function parseInlineMarkdown(text: string): React.ReactNode[] {
  const boldParts = text.split(/\*\*([\s\S]*?)\*\*/g);
  const elements: React.ReactNode[] = [];

  boldParts.forEach((part, boldIndex) => {
    const isBold = boldIndex % 2 === 1;
    const codeParts = part.split(/`([^`]+)`/g);
    codeParts.forEach((codePart, codeIndex) => {
      const isCode = codeIndex % 2 === 1;
      let node: React.ReactNode = codePart;

      if (isCode) {
        node = <code key={`${boldIndex}-${codeIndex}`}>{codePart}</code>;
      }
      if (isBold) {
        node = <strong key={`${boldIndex}-${codeIndex}-bold`}>{node}</strong>;
      }
      elements.push(node);
    });
  });

  return elements;
}

function renderMarkdown(text: string): React.ReactNode {
  if (!text) return null;
  const lines = text.split("\n");
  let inCodeBlock = false;
  let codeBlockLines: string[] = [];
  const elements: React.ReactNode[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (line.startsWith("```")) {
      if (inCodeBlock) {
        elements.push(
          <pre key={`code-${i}`} className="code-block">
            <code>{codeBlockLines.join("\n")}</code>
          </pre>
        );
        codeBlockLines = [];
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
      }
      continue;
    }

    if (inCodeBlock) {
      codeBlockLines.push(line);
      continue;
    }

    if (line.startsWith("- ") || line.startsWith("* ")) {
      const content = line.substring(2);
      elements.push(<li key={`li-${i}`}>{parseInlineMarkdown(content)}</li>);
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      const content = line.replace(/^\d+\.\s+/, "");
      elements.push(
        <li key={`li-num-${i}`} style={{ listStyleType: "decimal", marginLeft: "1.5rem" }}>
          {parseInlineMarkdown(content)}
        </li>
      );
      continue;
    }

    if (line.startsWith("### ")) {
      elements.push(<h3 key={`h3-${i}`}>{parseInlineMarkdown(line.substring(4))}</h3>);
      continue;
    }
    if (line.startsWith("## ")) {
      elements.push(<h2 key={`h2-${i}`}>{parseInlineMarkdown(line.substring(3))}</h2>);
      continue;
    }
    if (line.startsWith("# ")) {
      elements.push(<h1 key={`h1-${i}`}>{parseInlineMarkdown(line.substring(2))}</h1>);
      continue;
    }

    if (line.trim()) {
      elements.push(<p key={`p-${i}`}>{parseInlineMarkdown(line)}</p>);
    } else {
      elements.push(<div key={`br-${i}`} className="paragraph-spacer" />);
    }
  }

  return <div className="markdown-body">{elements}</div>;
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
      .map((m) => ({ role: m.role, content: m.content }));

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
    } finally {
      setIsStreaming(false);
    }
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
                {message.grounded === false && (
                  <span className="badge warning">⚠ response refined</span>
                )}
              </div>
              <div className="message-body">
                {message.content ? (
                  renderMarkdown(message.content)
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
          <button disabled={isStreaming || !input.trim()}>
            {isStreaming ? "Streaming" : "Ask"}
          </button>
        </form>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
