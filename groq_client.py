import os
from typing import Optional

import gradio as gr
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment variables.")

client = Groq(api_key=GROQ_API_KEY)

def get_response(query: str, rag_context: str, web_context: str) -> str:
    system_prompt = """You are Safe — a clinical assistant for crisis hotline counselors.
You only respond based on the context provided. Never make up information.
Always respond in this exact format:

TECHNIQUE: [name of the intervention technique]
PHRASE: [exact phrase the counselor can say right now]
RISK: [Low / Medium / High]
REASON: [one sentence explaining the risk level]"""

    user_prompt = f"""A crisis counselor needs help with this situation:
{query}

CLINICAL CONTEXT FROM DOCUMENTS:
{rag_context}

LIVE WEB RESOURCES:
{web_context}

Based only on the above context, provide guidance for the counselor."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()

def run_safe_assistant(query: str, rag_context: str, web_context: str) -> str:
    try:
        cleaned_query = query.strip()
        cleaned_rag = rag_context.strip()
        cleaned_web = web_context.strip()

        if not cleaned_query:
            return "Please enter a counselor query."

        return get_response(cleaned_query, cleaned_rag, cleaned_web)
    except Exception as exc:
        return f"Error: {exc}"

# Cleaned up CSS (removed theme-toggle specific styles)
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg-primary:
        radial-gradient(circle at top left, rgba(14,165,233,0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(20,184,166,0.12), transparent 22%),
        linear-gradient(180deg, #f8fbff 0%, #f4f7fb 50%, #eef3f9 100%);
    --text-main: #0f172a;
    --text-soft: #475569;
    --text-muted: #64748b;
    --surface: rgba(255, 255, 255, 0.9);
    --surface-strong: rgba(255, 255, 255, 0.97);
    --border: rgba(148, 163, 184, 0.22);
    --shadow-lg: 0 20px 60px rgba(0, 0, 0, 0.08);
    --shadow-md: 0 10px 30px rgba(0, 0, 0, 0.05);
    --input-bg: rgba(255, 255, 255, 0.92);
    --input-border: #cbd5e1;
    --button-text: #ffffff;
    --hero-title: #0f172a;
    --ring: rgba(14,165,233,0.18);
}

html, body, .gradio-container {
    margin: 0;
    padding: 0;
    min-height: 100%;
    font-family: 'Inter', sans-serif;
    font-size: 16.5px;
    color: var(--text-main);
    background: var(--bg-primary);
}

.gradio-container {
    max-width: 100% !important;
    padding: 28px 24px 40px !important;
}

.app-shell {
    width: 100%;
    max-width: 1380px;
    margin: 0 auto;
}

.hero {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 28px;
    padding: 32px;
    box-shadow: var(--shadow-lg);
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

.hero h1 {
    font-size: clamp(2rem, 3vw, 3rem);
    font-weight: 800;
    margin: 0;
    color: var(--hero-title);
}

.hero p {
    font-size: 1rem;
    margin-top: 12px;
    color: var(--text-soft);
}

.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 18px;
    box-shadow: var(--shadow-md);
    height: 100%;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

.section-title {
    font-size: 1.05rem;
    font-weight: 800;
    margin-bottom: 14px;
    color: var(--text-main);
}

.gradio-container .gr-textbox label {
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: var(--text-main) !important;
}

.gradio-container textarea,
.gradio-container input {
    font-size: 16.5px !important;
    line-height: 1.6 !important;
    padding: 12px 14px !important;
    border-radius: 14px !important;
    background: var(--input-bg) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--input-border) !important;
    box-shadow: none !important;
}

.gradio-container textarea:focus,
.gradio-container input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 4px var(--ring) !important;
}

.gradio-container button.primary {
    font-size: 1rem;
    font-weight: 800;
    padding: 12px;
    border-radius: 14px;
    background: linear-gradient(135deg, #0ea5e9, #14b8a6) !important;
    color: var(--button-text) !important;
    border: none !important;
    box-shadow: 0 12px 24px rgba(14, 165, 233, 0.20);
}

.footer-note {
    font-size: 0.95rem;
    color: var(--text-muted);
}
"""

with gr.Blocks(
    title="Safe Clinical Assistant",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
) as demo:
    with gr.Column(elem_classes="app-shell"):
        # Removed THEME_TOGGLE_HTML
        gr.HTML("""
        <div class="hero">
            <h1>Support crisis counselors with structured guidance.</h1>
            <p>Provide context → get actionable response instantly.</p>
        </div>
        """)

        with gr.Row(equal_height=True):
            with gr.Column(scale=6):
                with gr.Group(elem_classes="panel"):
                    gr.HTML('<div class="section-title">Input</div>')

                    query_input = gr.Textbox(
                        label="Counselor Query",
                        lines=4,
                        value="caller feels numb and disconnected",
                    )

                    rag_input = gr.Textbox(
                        label="Clinical Context",
                        lines=7,
                        value="Grounding techniques help reconnect to the present moment.",
                    )

                    web_input = gr.Textbox(
                        label="Web Context",
                        lines=7,
                        value="5-4-3-2-1 grounding technique engages senses.",
                    )

                    submit_btn = gr.Button("Generate Response", variant="primary")

            with gr.Column(scale=5):
                with gr.Group(elem_classes="panel"):
                    gr.HTML('<div class="section-title">Output</div>')

                    output_box = gr.Textbox(
                        lines=18,
                        interactive=False,
                        elem_classes="output-box",
                    )

                    gr.HTML('<div class="footer-note">TECHNIQUE • PHRASE • RISK • REASON</div>')

        submit_btn.click(
            fn=run_safe_assistant,
            inputs=[query_input, rag_input, web_input],
            outputs=output_box,
        )

if __name__ == "__main__":
    demo.launch()