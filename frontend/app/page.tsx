"use client";

import { useEffect, useRef, useState } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
  isError?: boolean;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const EXAMPLES = [
  "How have house prices in the Colombo district changed recently?",
  "What would a 1800 sqft, 3 bedroom, 2 bathroom house in Colombo that is 10 years old cost?",
];

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Keep the newest message in view, including the loading bubble.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(question: string) {
    if (!question || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const data = await res.json().catch(() => null);

      if (!res.ok) {
        // FastAPI puts the reason in `detail` - surfacing it verbatim is far
        // more useful than a generic failure, e.g. the 503 raised when the
        // agent has no LLM provider configured.
        throw new Error(
          typeof data?.detail === "string"
            ? data.detail
            : `Request failed (${res.status})`,
        );
      }

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: String(data.answer) },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            err instanceof Error
              ? err.message
              : `Could not reach the API at ${API_URL}.`,
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="shrink-0 border-b border-black/10 px-4 py-3 dark:border-white/15">
        <h1 className="mx-auto max-w-2xl text-sm font-semibold">
          House Price Intelligence Assistant
        </h1>
      </header>

      <main className="min-h-0 flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto flex max-w-2xl flex-col gap-4">
          {messages.length === 0 && (
            <div className="mt-8 text-center text-sm opacity-60">
              <p className="mb-4">Ask about Sri Lankan house prices.</p>
              <div className="flex flex-col items-center gap-2">
                {EXAMPLES.map((example) => (
                  <button
                    key={example}
                    onClick={() => send(example)}
                    className="max-w-md rounded-lg border border-black/10 px-3 py-2 text-left text-xs hover:bg-black/5 dark:border-white/15 dark:hover:bg-white/10"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message, i) => (
            <div
              key={i}
              className={
                message.role === "user" ? "flex justify-end" : "flex justify-start"
              }
            >
              <div
                className={[
                  "max-w-[85%] rounded-2xl px-4 py-2 text-sm whitespace-pre-wrap",
                  message.role === "user"
                    ? "bg-blue-600 text-white"
                    : message.isError
                      ? "bg-red-100 text-red-900 dark:bg-red-950 dark:text-red-200"
                      : "bg-black/5 dark:bg-white/10",
                ].join(" ")}
              >
                {message.content}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="rounded-2xl bg-black/5 px-4 py-2 text-sm opacity-70 dark:bg-white/10">
                Thinking...
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </main>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input.trim());
        }}
        className="shrink-0 border-t border-black/10 px-4 py-3 dark:border-white/15"
      >
        <div className="mx-auto flex max-w-2xl gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
            placeholder="Ask a question..."
            className="flex-1 rounded-full border border-black/15 bg-transparent px-4 py-2 text-sm outline-none focus:border-blue-600 disabled:opacity-50 dark:border-white/20"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="rounded-full bg-blue-600 px-5 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40"
          >
            {loading ? "..." : "Send"}
          </button>
        </div>
      </form>
    </div>
  );
}
