import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_response(query, rag_context, web_context):
    
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
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=300
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    test_rag = "Grounding techniques help patients reconnect with the present moment."
    test_web = "5-4-3-2-1 technique helps with dissociation by engaging the senses."
    result = get_response("caller feels numb and disconnected", test_rag, test_web)
    print(result)
