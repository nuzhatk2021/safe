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
        if not query.strip():
            return "Please enter a query."

        return get_response(
            query=query.strip(),
            rag_context=rag_context.strip(),
            web_context=web_context.strip(),
        )
    except Exception as exc:
        return f"Error: {exc}"


with gr.Blocks(title="Safe Clinical Assistant") as demo:
    gr.Markdown("# Safe Clinical Assistant")
    gr.Markdown(
        "Enter a crisis counseling scenario along with RAG and web context, then generate a structured response."
    )

    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="Counselor Query",
                placeholder="Example: caller feels numb and disconnected",
                lines=3,
                value="caller feels numb and disconnected",
            )

            rag_input = gr.Textbox(
                label="Clinical Context from Documents (RAG)",
                lines=5,
                value="Grounding techniques help patients reconnect with the present moment.",
            )

            web_input = gr.Textbox(
                label="Live Web Resources",
                lines=5,
                value="5-4-3-2-1 technique helps with dissociation by engaging the senses.",
            )

            submit_btn = gr.Button("Generate Response")

        with gr.Column():
            output_box = gr.Textbox(
                label="Assistant Output",
                lines=12,
                interactive=False,
            )

    submit_btn.click(
        fn=run_safe_assistant,
        inputs=[query_input, rag_input, web_input],
        outputs=output_box,
    )

if __name__ == "__main__":
    demo.launch()