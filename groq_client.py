import os
from typing import Optional

import gradio as gr
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment variables.")

client = Groq(api_key=GROQ_API_KEY)


def get_response(query: str, rag_context: str, web_context: str) -> str:
    """
    Generate a counselor-facing response using only the provided context.
    """
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
    """
    Wrapper for Gradio with basic error handling.
    """
    try:
        cleaned_query = query.strip()
        cleaned_rag = rag_context.strip()
        cleaned_web = web_context.strip()

        if not cleaned_query:
            return "Please enter a counselor query."

        return get_response(
            query=cleaned_query,
            rag_context=cleaned_rag,
            web_context=cleaned_web,
        )
    except Exception as exc:
        return f"Error: {exc}"


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .gradio-container {
    margin: 0;
    padding: 0;
    min-height: 100%;
    font-family: 'Inter', sans-serif;
    background:
        radial-gradient(circle at top left, #1e293b 0%, transparent 30%),
        radial-gradient(circle at top right, #0f766e 0%, transparent 25%),
        linear-gradient(180deg, #0b1020 0%, #111827 100%);
    color: #e5e7eb;
}

.gradio-container {
    max-width: 100% !important;
    padding: 32px 20px 40px !important;
}

.app-shell {
    max-width: 1280px;
    margin: 0 auto;
}

.hero {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(17, 24, 39, 0.78));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 28px 28px 22px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}

.hero-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(45, 212, 191, 0.12);
    color: #99f6e4;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 14px;
    border: 1px solid rgba(45, 212, 191, 0.25);
}

.hero h1 {
    margin: 0;
    font-size: 2.4rem;
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #f8fafc;
}

.hero p {
    margin-top: 12px;
    margin-bottom: 0;
    max-width: 760px;
    font-size: 1rem;
    line-height: 1.7;
    color: #cbd5e1;
}

.panel {
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 18px;
    box-shadow: 0 18px 50px rgba(0,0,0,0.25);
    backdrop-filter: blur(10px);
}

.section-title {
    margin: 4px 0 14px 2px;
    font-size: 0.95rem;
    font-weight: 700;
    color: #cbd5e1;
    letter-spacing: 0.01em;
}

.gr-box,
.gr-textbox,
.gr-button,
.gr-markdown {
    border-radius: 16px !important;
}

textarea,
input {
    font-size: 15px !important;
}

.gradio-container .gr-textbox,
.gradio-container .gr-textbox textarea,
.gradio-container .gr-textbox input {
    background: rgba(15, 23, 42, 0.72) !important;
    color: #f8fafc !important;
    border-color: rgba(148, 163, 184, 0.22) !important;
}

.gradio-container .gr-textbox label,
.gradio-container .gr-textbox .wrap label {
    color: #dbeafe !important;
    font-weight: 600 !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: #94a3b8 !important;
}

.gradio-container button.primary,
.gradio-container .gr-button-primary {
    background: linear-gradient(135deg, #14b8a6, #0ea5e9) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 10px 25px rgba(14, 165, 233, 0.25);
}

.gradio-container button.primary:hover,
.gradio-container .gr-button-primary:hover {
    filter: brightness(1.05);
    transform: translateY(-1px);
    transition: all 0.18s ease;
}

.output-box textarea {
    font-family: "Inter", sans-serif !important;
    line-height: 1.65 !important;
}

.footer-note {
    margin-top: 14px;
    color: #94a3b8;
    font-size: 0.9rem;
}

@media (max-width: 900px) {
    .hero h1 {
        font-size: 1.9rem;
    }

    .gradio-container {
        padding: 18px 12px 24px !important;
    }

    .hero,
    .panel {
        border-radius: 18px;
    }
}
"""


with gr.Blocks(
    title="Safe Clinical Assistant",
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
    css=CUSTOM_CSS,
) as demo:
    with gr.Column(elem_classes="app-shell"):
        gr.HTML(
            """
            <div class="hero">
                <div class="hero-badge">Safe • Clinical Assistant</div>
                <h1>Support crisis counselors with structured, context-grounded guidance.</h1>
                <p>
                    Enter the counselor’s situation, supporting document context, and any live web
                    context. The assistant will return a concise response in a strict clinical format.
                </p>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=6):
                with gr.Group(elem_classes="panel"):
                    gr.HTML('<div class="section-title">Input</div>')

                    query_input = gr.Textbox(
                        label="Counselor Query",
                        placeholder="Example: Caller feels numb, detached, and says they are not fully present.",
                        lines=4,
                        value="caller feels numb and disconnected",
                    )

                    rag_input = gr.Textbox(
                        label="Clinical Context from Documents (RAG)",
                        placeholder="Paste grounded document context here...",
                        lines=7,
                        value="Grounding techniques help patients reconnect with the present moment.",
                    )

                    web_input = gr.Textbox(
                        label="Live Web Resources",
                        placeholder="Paste relevant live web findings here...",
                        lines=7,
                        value="5-4-3-2-1 technique helps with dissociation by engaging the senses.",
                    )

                    submit_btn = gr.Button("Generate Response", variant="primary")

            with gr.Column(scale=5):
                with gr.Group(elem_classes="panel"):
                    gr.HTML('<div class="section-title">Assistant Output</div>')

                    output_box = gr.Textbox(
                        label="Response",
                        lines=18,
                        interactive=False,
                        elem_classes="output-box",
                        placeholder="The structured clinical response will appear here...",
                    )

                    gr.HTML(
                        """
                        <div class="footer-note">
                            Output format: TECHNIQUE • PHRASE • RISK • REASON
                        </div>
                        """
                    )

        submit_btn.click(
            fn=run_safe_assistant,
            inputs=[query_input, rag_input, web_input],
            outputs=output_box,
        )

if __name__ == "__main__":
    demo.launch()