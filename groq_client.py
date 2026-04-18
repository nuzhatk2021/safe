import os
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

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