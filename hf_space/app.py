"""
P11 · Streaming LLM API + Real-time UX — HuggingFace Space
Token-by-token streaming with TTFT tracking, cancellation, and rate limiting.
gradio==5.29.0 + audioop-lts for Python 3.13 compatibility.
"""

import os
import sys
import uuid

import gradio as gr
from transformers import pipeline

sys.path.insert(0, os.path.dirname(__file__))
from src.streamer import stream_response, cancel_stream, rate_limiter
from src.metrics import metrics_store

# ── Load model ────────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading {MODEL}...")
pipe = pipeline(
    "text-generation",
    model=MODEL,
    max_new_tokens=300,
    temperature=0.7,
    do_sample=True,
    device_map="cpu",
)
print("Model loaded.")

# ── Sample SRE queries ────────────────────────────────────────────────────────
SAMPLE_QUERIES = [
    "What steps should I take for a CrashLoopBackOff pod?",
    "How do I calculate error budget for a 99.9% SLO?",
    "What is the on-call handoff checklist?",
    "How do I debug high API latency?",
    "What is a burn rate alert?",
    "How do I safely roll back a Kubernetes deployment?",
    "What metrics should I collect for a microservice?",
]


def get_metrics_summary() -> str:
    s = metrics_store.summary()
    if s.get("completed", 0) == 0:
        return "_No requests yet — ask a question to see metrics._"

    lines = [
        "### 📊 Session Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total requests | {s['total_requests']} |",
        f"| Completed | {s['completed']} |",
        f"| Cancelled | {s.get('cancelled', 0)} |",
        f"| Errors | {s.get('errors', 0)} |",
    ]
    if s.get("avg_ttft_ms"):
        lines.append(f"| Avg TTFT | {s['avg_ttft_ms']}ms |")
    if s.get("p95_ttft_ms"):
        lines.append(f"| p95 TTFT | {s['p95_ttft_ms']}ms |")
    if s.get("avg_total_ms"):
        lines.append(f"| Avg total | {s['avg_total_ms']}ms |")
    if s.get("avg_tokens_per_sec"):
        lines.append(f"| Avg throughput | {s['avg_tokens_per_sec']} tok/s |")

    lines += [
        "",
        "**SRE note:** In production, TTFT p95 < 500ms would be the SLO.",
        "Current model runs on CPU — expect higher latency than GPU.",
    ]
    return "\n".join(lines)


def chat_stream(message: str, history: list, session_id: str):
    """Stream response token by token."""
    if not message.strip():
        yield history, "_Please enter a question._", get_metrics_summary()
        return

    # Add user message to history
    history = history + [[message, ""]]

    # Stream tokens
    for partial_text, metrics_line in stream_response(
        pipe=pipe,
        prompt=message,
        session_id=session_id,
        user_id=session_id,
    ):
        history[-1][1] = partial_text
        yield history, metrics_line, get_metrics_summary()


def stop_stream(session_id: str):
    """Cancel the current stream."""
    cancel_stream(session_id)
    return "🚫 Stream cancelled"


def clear_chat():
    return [], "", get_metrics_summary()


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="P11 · Streaming LLM", theme=gr.themes.Soft()) as demo:

    # Session ID — unique per browser session
    session_id = gr.State(lambda: str(uuid.uuid4())[:8])

    gr.Markdown("""
    # ⚡ P11 · Streaming LLM API + Real-time UX
    **Staff SRE + AI Engineer Portfolio**

    Token-by-token streaming with **TTFT tracking**, **cancellation**, and **rate limiting**.
    Ask any SRE question and watch the response stream in real-time.

    Model: **Qwen2.5-0.5B-Instruct** · running locally · no external API calls
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="SRE Streaming Assistant",
                height=420,
                show_copy_button=True,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your question",
                    placeholder="What steps should I take for a CrashLoopBackOff pod?",
                    scale=4,
                )
                send_btn = gr.Button("▶ Send", variant="primary", scale=1)
                stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)

            status_line = gr.Markdown("_Ready_")

            gr.Markdown("**Sample queries:**")
            for q in SAMPLE_QUERIES:
                btn = gr.Button(q, size="sm")
                btn.click(fn=lambda x=q: x, outputs=msg_input)

        with gr.Column(scale=2):
            metrics_panel = gr.Markdown(get_metrics_summary())
            refresh_metrics_btn = gr.Button("🔄 Refresh Metrics")

            with gr.Accordion("📖 What this demonstrates", open=False):
                gr.Markdown("""
                ## Streaming implementation

                **Token-by-token streaming:**
                Words appear progressively as generated — same UX as ChatGPT.

                **TTFT (Time To First Token):**
                The key latency metric for streaming UX. Users perceive
                responsiveness from TTFT, not total response time.
                In production: SLO p95 TTFT < 500ms.

                **Cancellation:**
                Click ⏹ Stop to cancel mid-stream. Uses a cancellation token
                checked between each token — standard pattern for async streams.

                **Rate limiting:**
                10 requests/minute per session. Returns retry-after header.
                Prevents runaway costs in production.

                **Backpressure:**
                Generator pattern yields control between tokens — prevents
                memory buildup if consumer is slower than producer.

                **SRE additions:**
                - TTFT + throughput tracked per request
                - p95 TTFT displayed in metrics panel
                - Rate limiter with per-user buckets
                - Graceful error handling — stream errors don't crash the server
                """)

    gr.Markdown("""
    ---
    [GitHub](https://github.com/amarshiv86/p11-streaming) ·
    [Staff SRE Portfolio](https://github.com/amarshiv86)
    """)

    # ── Event handlers ────────────────────────────────────────────────────────
    send_btn.click(
        fn=chat_stream,
        inputs=[msg_input, chatbot, session_id],
        outputs=[chatbot, status_line, metrics_panel],
    ).then(fn=lambda: "", outputs=msg_input)

    msg_input.submit(
        fn=chat_stream,
        inputs=[msg_input, chatbot, session_id],
        outputs=[chatbot, status_line, metrics_panel],
    ).then(fn=lambda: "", outputs=msg_input)

    stop_btn.click(
        fn=stop_stream,
        inputs=[session_id],
        outputs=[status_line],
    )

    refresh_metrics_btn.click(
        fn=get_metrics_summary,
        outputs=[metrics_panel],
    )

    clear_btn = gr.Button("🗑 Clear Chat")
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, status_line, metrics_panel],
    )

demo.launch()
