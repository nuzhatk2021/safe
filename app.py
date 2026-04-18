import os
from typing import Optional
import gradio as gr
from dotenv import load_dotenv
from groq import Groq
from rag_pipeline import build_index, retrieve
from tavily_search import search_web
from voice_output import speak

load_dotenv()

GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment variables.")

client = Groq(api_key=GROQ_API_KEY)

print("Building Safe index...")
build_index()
print("Safe is ready.")

conversation_history = []
current_situation = ""
current_guidance = ""


def get_guidance(query, rag_context, web_context):
    system_prompt = """You are Safe — a clinical assistant for crisis hotline counselors.
You only respond based on the context provided. Never make up information.
Always respond in this exact format:

TECHNIQUE: [name of the intervention technique]
PHRASE: [exact phrase the counselor can say right now — warm, human, natural]
RISK: [Low / Medium / High]
REASON: [one sentence explaining the risk level]
NEXT STEPS:
- [action 1]
- [action 2]
- [action 3]
QUESTION 1: [first question to ask the caller]
QUESTION 2: [second question to ask the caller]
QUESTION 3: [third question to ask the caller]"""

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
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


def get_followup_response(question, situation, previous):
    system_prompt = """You are Safe — a clinical assistant for crisis hotline counselors.
Answer the counselor's follow-up question concisely and practically.
End with one suggested phrase the counselor can say right now.
Keep it under 4 sentences."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Situation: {situation}\nPrevious guidance: {previous}\nFollow-up: {question}"},
        ],
        max_tokens=250,
    )
    return response.choices[0].message.content.strip()


def generate_summary():
    global conversation_history
    if not conversation_history:
        return "No session to summarize yet. Describe a situation first."

    system_prompt = """You are Safe. Generate a concise clinical session note.
Format exactly:
CALLER SUMMARY: [brief description]
INTERVENTION USED: [technique applied]
RISK ASSESSMENT: [risk level and reasoning]
COUNSELOR ACTIONS: [what was done]
FOLLOW-UP NEEDED: [Yes/No and why]
REFERRALS SUGGESTED: [any referrals]"""

    history_text = "\n\n".join([
        f"Situation: {item.get('situation','')}\nGuidance: {item.get('guidance','')}"
        if 'situation' in item
        else f"Q: {item.get('followup','')}\nA: {item.get('answer','')}"
        for item in conversation_history
    ])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Session:\n{history_text}"},
        ],
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


def run_safe(query):
    global conversation_history, current_situation, current_guidance
    conversation_history = []

    if not query.strip():
        return "", "", "", "", "", "", ""

    rag_results = retrieve(query)
    rag_context = "\n\n".join(rag_results)
    web_context = search_web(query)
    response = get_guidance(query.strip(), rag_context, web_context)

    current_situation = query
    current_guidance = response
    conversation_history.append({"situation": query, "guidance": response})

    technique, phrase, risk_level, reason = "", "", "", ""
    next_steps = []
    questions = []

    for line in response.split("\n"):
        l = line.strip()
        if l.startswith("TECHNIQUE:"):
            technique = l.replace("TECHNIQUE:", "").strip()
        elif l.startswith("PHRASE:"):
            phrase = l.replace("PHRASE:", "").strip().strip('"')
        elif l.startswith("RISK:"):
            risk_level = l.replace("RISK:", "").strip()
        elif l.startswith("REASON:"):
            reason = l.replace("REASON:", "").strip()
        elif l.startswith("- ") and technique and not questions:
            next_steps.append(l[2:])
        elif l.startswith("QUESTION 1:"):
            questions.append(l.replace("QUESTION 1:", "").strip())
        elif l.startswith("QUESTION 2:"):
            questions.append(l.replace("QUESTION 2:", "").strip())
        elif l.startswith("QUESTION 3:"):
            questions.append(l.replace("QUESTION 3:", "").strip())

    # Risk HTML
    if "High" in risk_level:
        risk_html = """<div style='background:#fff0f0;border:2px solid #ef4444;border-radius:12px;
        padding:14px 20px;color:#b91c1c;font-weight:800;font-size:1.1rem;
        letter-spacing:1px;margin-bottom:16px;text-align:center;'>
        🚨 HIGH RISK — Escalate immediately</div>"""
    elif "Medium" in risk_level:
        risk_html = """<div style='background:#fffbeb;border:2px solid #f59e0b;border-radius:12px;
        padding:14px 20px;color:#b45309;font-weight:800;font-size:1.1rem;
        letter-spacing:1px;margin-bottom:16px;text-align:center;'>
        ⚠️ MEDIUM RISK — Monitor closely</div>"""
    else:
        risk_html = """<div style='background:#f0fdf4;border:2px solid #22c55e;border-radius:12px;
        padding:14px 20px;color:#15803d;font-weight:800;font-size:1.1rem;
        letter-spacing:1px;margin-bottom:16px;text-align:center;'>
        ✓ LOW RISK</div>"""

    # Phrase HTML
    phrase_html = f"""
    <div style='background:rgba(124,58,237,0.08);border:2px solid rgba(124,58,237,0.2);
    border-radius:14px;padding:20px 22px;margin-bottom:16px;'>
        <div style='font-size:0.62rem;font-weight:800;letter-spacing:2px;color:#7c3aed;
        margin-bottom:10px;text-transform:uppercase;'>💬 Say this right now</div>
        <div style='font-size:1.15rem;font-weight:700;color:#1e1b4b;line-height:1.6;font-style:italic;'>
        "{phrase}"</div>
    </div>
    <div style='display:flex;gap:16px;margin-bottom:4px;'>
        <div style='font-size:0.85rem;color:#374151;'><strong style='color:#1e1b4b;'>Technique:</strong> {technique}</div>
    </div>
    <div style='font-size:0.85rem;color:#374151;'><strong style='color:#1e1b4b;'>Why:</strong> {reason}</div>
    """

    # Next steps HTML
    steps_html = "".join([
        f"<div style='display:flex;gap:10px;align-items:flex-start;margin-bottom:8px;'>"
        f"<div style='min-width:22px;height:22px;border-radius:50%;"
        f"background:linear-gradient(135deg,#7c3aed,#a855f7);"
        f"color:white;font-size:0.68rem;font-weight:800;"
        f"display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px;'>{i+1}</div>"
        f"<div style='font-size:0.88rem;color:#1e1b4b;line-height:1.6;'>{s}</div></div>"
        for i, s in enumerate(next_steps[:3])
    ])

    # Questions HTML — now goes to right column
    q_chips = ""
    for i, q in enumerate(questions[:3]):
        q_chips += f"""
        <div class="q-chip" onclick="handleQuestion(this, '{q.replace("'", "\\'")}')">
            💬 {q}
        </div>"""

    questions_html = f"""
    <div style='font-size:0.62rem;font-weight:800;letter-spacing:2px;color:#a5b4fc;
    text-transform:uppercase;margin-bottom:12px;'>Tap a question — Safe answers instantly</div>
    {q_chips}
    """ if q_chips else "<div style='color:#94a3b8;font-size:0.85rem;'>Questions will appear after you get guidance...</div>"

    return risk_html, phrase_html, steps_html, questions_html, "", "", ""


def ask_followup(question):
    global conversation_history, current_situation, current_guidance
    if not question.strip():
        return "Please type a follow-up question."
    if not current_situation:
        return "Please describe a situation first."
    answer = get_followup_response(question, current_situation, current_guidance)
    conversation_history.append({"followup": question, "answer": answer})
    return answer


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body {
    margin:0; padding:0; font-family:'Inter',sans-serif;
    min-height:100vh;
    background:linear-gradient(135deg,#ede9fe 0%,#e0e7ff 40%,#fce7f3 100%);
}
.gradio-container {
    max-width:100% !important; padding:0 !important;
    background:transparent !important; min-height:100vh;
}
.gradio-container > .main > .wrap { padding:0 !important; min-height:100vh; }
.app-shell { width:100%; min-height:100vh; padding:18px; box-sizing:border-box; }

.hero {
    border-radius:18px; padding:18px 26px; margin-bottom:14px;
    backdrop-filter:blur(16px); display:flex;
    align-items:center; justify-content:space-between; gap:20px;
    background:rgba(255,255,255,0.9);
    border:2px solid rgba(167,139,250,0.3);
    box-shadow:0 10px 32px rgba(124,58,237,0.08);
}

.safe-logo {
    font-size:clamp(2rem,4vw,3.2rem); font-weight:900; letter-spacing:-1px;
    background:linear-gradient(135deg,#7c3aed,#a855f7,#ec4899);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin:0; line-height:1;
}

.hero-badge {
    display:inline-block; color:white; font-size:0.6rem; font-weight:700;
    padding:3px 10px; border-radius:999px; margin-bottom:6px;
    letter-spacing:1.5px; text-transform:uppercase;
    background:linear-gradient(135deg,#7c3aed,#a855f7);
}

.hero-sub { font-size:0.82rem; margin:4px 0 0 0; color:rgba(30,27,75,0.6); }

.theme-dots { display:flex; gap:8px; align-items:center; }
.theme-dot {
    width:22px; height:22px; border-radius:50%; cursor:pointer;
    border:2px solid rgba(255,255,255,0.5); transition:transform 0.2s;
}
.theme-dot:hover { transform:scale(1.3); }

.panel {
    border-radius:16px; padding:18px; backdrop-filter:blur(12px);
    background:rgba(255,255,255,0.9);
    border:2px solid rgba(167,139,250,0.25);
    box-shadow:0 6px 20px rgba(124,58,237,0.06);
    height:100%;
}

.section-label {
    font-size:0.6rem; font-weight:800; letter-spacing:2.5px;
    text-transform:uppercase; margin-bottom:12px; color:#7c3aed;
}

.divider { height:1px; margin:14px 0; background:rgba(167,139,250,0.2); }

.q-chip {
    display:block; padding:12px 16px; margin:8px 0;
    border-radius:12px; background:rgba(124,58,237,0.07);
    border:1.5px solid rgba(124,58,237,0.2);
    color:#3b0764; font-size:0.88rem; font-weight:600;
    cursor:pointer; transition:all 0.2s ease;
    font-family:'Inter',sans-serif; line-height:1.4;
}
.q-chip:hover {
    background:rgba(124,58,237,0.14);
    border-color:rgba(124,58,237,0.4);
    transform:translateX(4px);
    color:#1e1b4b;
}
.q-chip.active {
    background:linear-gradient(135deg,#7c3aed,#a855f7);
    color:white; border-color:transparent;
}

.affirmation {
    margin-top:12px; padding:12px 14px; border-radius:12px;
    display:flex; align-items:center; gap:10px;
    background:linear-gradient(135deg,#7c3aed,#a855f7);
}
.affirmation-text { font-size:0.8rem; font-weight:600; color:white; line-height:1.5; }
.affirmation-text span { font-weight:400; opacity:0.85; }

.gradio-container textarea,
.gradio-container input[type=text] {
    font-family:'Inter',sans-serif !important; font-size:14.5px !important;
    line-height:1.65 !important; padding:11px 14px !important;
    border-radius:10px !important; box-shadow:none !important;
    transition:all 0.2s !important;
    background:rgba(255,255,255,0.85) !important;
    border:1.5px solid rgba(167,139,250,0.25) !important;
    color:#1e1b4b !important;
}
.gradio-container textarea:focus,
.gradio-container input[type=text]:focus {
    border-color:#7c3aed !important;
    box-shadow:0 0 0 3px rgba(124,58,237,0.1) !important;
    background:#fff !important;
}
.gradio-container label span {
    font-size:0.82rem !important; font-weight:700 !important; color:#1e1b4b !important;
}

.gradio-container button.primary {
    font-family:'Inter',sans-serif !important; font-size:0.9rem !important;
    font-weight:700 !important; padding:12px 14px !important;
    border-radius:10px !important; color:white !important; border:none !important;
    width:100% !important; transition:all 0.2s ease !important;
    background:linear-gradient(135deg,#7c3aed,#a855f7) !important;
    box-shadow:0 6px 16px rgba(124,58,237,0.25) !important;
}
.gradio-container button.primary:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 10px 22px rgba(124,58,237,0.35) !important;
}
.gradio-container button.secondary {
    font-family:'Inter',sans-serif !important; font-size:0.85rem !important;
    font-weight:600 !important; padding:10px 14px !important;
    border-radius:10px !important; width:100% !important; transition:all 0.2s !important;
    background:rgba(124,58,237,0.08) !important; color:#5b21b6 !important;
    border:1.5px solid rgba(124,58,237,0.2) !important;
}
.gradio-container button.secondary:hover {
    background:rgba(124,58,237,0.15) !important;
}

.footer {
    text-align:center; padding:14px; color:#a5b4fc;
    font-size:0.65rem; letter-spacing:1.5px;
}
"""

THEME_JS = """
const THEMES = {
  dreamy: {
    bg:'linear-gradient(135deg,#ede9fe 0%,#e0e7ff 40%,#fce7f3 100%)',
    panelBg:'rgba(255,255,255,0.9)', panelBorder:'rgba(167,139,250,0.25)',
    logoGrad:'linear-gradient(135deg,#7c3aed,#a855f7,#ec4899)',
    heroSub:'rgba(30,27,75,0.6)', sectionLabel:'#7c3aed',
    inputBg:'rgba(255,255,255,0.85)', inputBorder:'rgba(167,139,250,0.25)',
    inputColor:'#1e1b4b', labelColor:'#1e1b4b',
    btnBg:'linear-gradient(135deg,#7c3aed,#a855f7)',
    btn2Bg:'rgba(124,58,237,0.08)', btn2Color:'#5b21b6', btn2Border:'rgba(124,58,237,0.2)',
    affirmBg:'linear-gradient(135deg,#7c3aed,#a855f7)',
    badgeBg:'linear-gradient(135deg,#7c3aed,#a855f7)',
    qBg:'rgba(124,58,237,0.07)', qBorder:'rgba(124,58,237,0.2)', qColor:'#3b0764',
    footerColor:'#a5b4fc',
  },
  ocean: {
    bg:'linear-gradient(135deg,#ecfdf5 0%,#e0f2fe 40%,#f0f9ff 100%)',
    panelBg:'rgba(255,255,255,0.9)', panelBorder:'rgba(20,184,166,0.25)',
    logoGrad:'linear-gradient(135deg,#0ea5e9,#14b8a6)',
    heroSub:'rgba(15,42,42,0.6)', sectionLabel:'#0ea5e9',
    inputBg:'rgba(255,255,255,0.85)', inputBorder:'rgba(20,184,166,0.25)',
    inputColor:'#0f2a2a', labelColor:'#0f2a2a',
    btnBg:'linear-gradient(135deg,#0ea5e9,#14b8a6)',
    btn2Bg:'rgba(14,165,233,0.07)', btn2Color:'#0369a1', btn2Border:'rgba(14,165,233,0.2)',
    affirmBg:'linear-gradient(135deg,#0ea5e9,#14b8a6)',
    badgeBg:'linear-gradient(135deg,#0ea5e9,#14b8a6)',
    qBg:'rgba(14,165,233,0.07)', qBorder:'rgba(14,165,233,0.2)', qColor:'#0369a1',
    footerColor:'#5eead4',
  },
  midnight: {
    bg:'linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#0f2942 100%)',
    panelBg:'rgba(255,255,255,0.06)', panelBorder:'rgba(255,255,255,0.09)',
    logoGrad:'linear-gradient(135deg,#818cf8,#a5b4fc)',
    heroSub:'rgba(226,232,240,0.6)', sectionLabel:'#818cf8',
    inputBg:'rgba(255,255,255,0.08)', inputBorder:'rgba(255,255,255,0.12)',
    inputColor:'#e2e8f0', labelColor:'#e2e8f0',
    btnBg:'linear-gradient(135deg,#6366f1,#8b5cf6)',
    btn2Bg:'rgba(99,102,241,0.1)', btn2Color:'#a5b4fc', btn2Border:'rgba(99,102,241,0.25)',
    affirmBg:'linear-gradient(135deg,#6366f1,#8b5cf6)',
    badgeBg:'linear-gradient(135deg,#6366f1,#8b5cf6)',
    qBg:'rgba(99,102,241,0.1)', qBorder:'rgba(99,102,241,0.25)', qColor:'#c7d2fe',
    footerColor:'#6366f1',
  },
  void: {
    bg:'#000',
    panelBg:'rgba(255,255,255,0.05)', panelBorder:'rgba(255,255,255,0.07)',
    logoGrad:'linear-gradient(135deg,#64748b,#94a3b8)',
    heroSub:'rgba(241,245,249,0.5)', sectionLabel:'#64748b',
    inputBg:'rgba(255,255,255,0.06)', inputBorder:'rgba(255,255,255,0.08)',
    inputColor:'#f1f5f9', labelColor:'#f1f5f9',
    btnBg:'linear-gradient(135deg,#1e293b,#334155)',
    btn2Bg:'rgba(255,255,255,0.05)', btn2Color:'#94a3b8', btn2Border:'rgba(255,255,255,0.08)',
    affirmBg:'linear-gradient(135deg,#1e293b,#0f172a)',
    badgeBg:'linear-gradient(135deg,#1e293b,#334155)',
    qBg:'rgba(255,255,255,0.05)', qBorder:'rgba(255,255,255,0.08)', qColor:'#cbd5e1',
    footerColor:'#1e293b',
  }
};

function applyTheme(name) {
  const t = THEMES[name];
  if (!t) return;
  document.body.style.background = t.bg;
  document.body.style.minHeight = '100vh';
  document.querySelectorAll('.hero,.panel').forEach(el => {
    el.style.background = t.panelBg;
    el.style.border = '2px solid ' + t.panelBorder;
  });
  document.querySelectorAll('.safe-logo').forEach(el => el.style.backgroundImage = t.logoGrad);
  document.querySelectorAll('.hero-badge').forEach(el => el.style.background = t.badgeBg);
  document.querySelectorAll('.hero-sub').forEach(el => el.style.color = t.heroSub);
  document.querySelectorAll('.section-label').forEach(el => el.style.color = t.sectionLabel);
  document.querySelectorAll('.divider').forEach(el => el.style.background = 'rgba(127,127,127,0.15)');
  document.querySelectorAll('.affirmation').forEach(el => el.style.background = t.affirmBg);
  document.querySelectorAll('.footer').forEach(el => el.style.color = t.footerColor);
  document.querySelectorAll('.q-chip:not(.active)').forEach(el => {
    el.style.background = t.qBg;
    el.style.border = '1.5px solid ' + t.qBorder;
    el.style.color = t.qColor;
  });
  document.querySelectorAll('.gradio-container textarea,.gradio-container input[type=text]').forEach(el => {
    el.style.background = t.inputBg;
    el.style.border = '1.5px solid ' + t.inputBorder;
    el.style.color = t.inputColor;
  });
  document.querySelectorAll('.gradio-container label span').forEach(el => el.style.color = t.labelColor);
  document.querySelectorAll('.gradio-container button.primary').forEach(el => el.style.background = t.btnBg);
  document.querySelectorAll('.gradio-container button.secondary').forEach(el => {
    el.style.background = t.btn2Bg;
    el.style.color = t.btn2Color;
    el.style.border = '1.5px solid ' + t.btn2Border;
  });
}

function handleQuestion(el, question) {
  document.querySelectorAll('.q-chip').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  const textareas = document.querySelectorAll('.gradio-container textarea');
  textareas.forEach(ta => {
    if (ta.placeholder && ta.placeholder.toLowerCase().includes('refuses')) {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
      setter.call(ta, question);
      ta.dispatchEvent(new Event('input', { bubbles: true }));
      setTimeout(() => {
        document.querySelectorAll('.gradio-container button').forEach(btn => {
          if (btn.textContent.trim() === 'Ask \u2192') btn.click();
        });
      }, 200);
    }
  });
}

window.addEventListener('load', () => setTimeout(() => applyTheme('dreamy'), 400));
"""

with gr.Blocks(
    title="Safe — Clinical Co-Pilot",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
    head=f"<script>{THEME_JS}</script>",
) as demo:
    with gr.Column(elem_classes="app-shell"):

        gr.HTML("""
        <div class="hero">
            <div>
                <div class="hero-badge">Hack Brooklyn 2026</div>
                <p class="safe-logo">SAFE</p>
                <p class="hero-sub">Real-time clinical guidance — built for the counselor, not the patient.</p>
            </div>
            <div style="text-align:right;">
                <div style='font-size:0.55rem;font-weight:800;letter-spacing:2px;margin-bottom:8px;opacity:0.5;text-transform:uppercase;'>Theme</div>
                <div class="theme-dots">
                    <div class="theme-dot" title="Dreamy" style="background:linear-gradient(135deg,#7c3aed,#a855f7);" onclick="applyTheme('dreamy')"></div>
                    <div class="theme-dot" title="Ocean" style="background:linear-gradient(135deg,#0ea5e9,#14b8a6);" onclick="applyTheme('ocean')"></div>
                    <div class="theme-dot" title="Midnight" style="background:linear-gradient(135deg,#1e1b4b,#0f2942);" onclick="applyTheme('midnight')"></div>
                    <div class="theme-dot" title="Void" style="background:#111;border-color:#333;" onclick="applyTheme('void')"></div>
                </div>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── Column 1: Input ──
            with gr.Column(scale=4):
                with gr.Group(elem_classes="panel"):
                    gr.HTML('<div class="section-label">Counselor Input</div>')
                    situation_input = gr.Textbox(
                        label="What is happening on this call?",
                        placeholder="e.g. Caller is a 19yo male, first time calling, says he feels completely numb and disconnected from reality...",
                        lines=7,
                    )
                    get_guidance_btn = gr.Button("Get Guidance →", variant="primary")
                    gr.HTML('<div class="divider"></div>')
                    gr.HTML('<div class="section-label">Ask Safe a Follow-up</div>')
                    followup_input = gr.Textbox(
                        label="Your question",
                        placeholder="e.g. What if the caller refuses to engage?",
                        lines=3,
                    )
                    followup_btn = gr.Button("Ask →", variant="secondary")
                    gr.HTML("""
                    <div class="affirmation" style="margin-top:12px;">
                        <div style="font-size:1.1rem;">💙</div>
                        <div class="affirmation-text">
                            <strong>Helping someone could change a life.</strong><br>
                            <span>Take a deep breath — you're making a difference.</span>
                        </div>
                    </div>
                    """)

            # ── Column 2: Main Response ──
            with gr.Column(scale=5):
                with gr.Group(elem_classes="panel"):
                    risk_display = gr.HTML(value="<div style='height:4px'></div>")
                    phrase_display = gr.HTML(
                        value="""<div style='background:rgba(124,58,237,0.06);border:2px solid rgba(124,58,237,0.15);
                        border-radius:14px;padding:20px 22px;'>
                        <div style='font-size:0.62rem;font-weight:800;letter-spacing:2px;color:#7c3aed;
                        margin-bottom:10px;text-transform:uppercase;'>💬 Say this right now</div>
                        <div style='font-size:0.95rem;color:#94a3b8;font-style:italic;'>
                        Your phrase will appear here after you get guidance...</div></div>"""
                    )
                    gr.HTML('<div class="divider"></div>')
                    gr.HTML('<div class="section-label">Next Steps</div>')
                    next_steps_display = gr.HTML(
                        value="<div style='color:#94a3b8;font-size:0.85rem;'>Next steps will appear here...</div>"
                    )

            # ── Column 3: Questions + Follow-up + Summary ──
            with gr.Column(scale=4):
                with gr.Group(elem_classes="panel"):
                    gr.HTML('<div class="section-label">Suggested Questions</div>')
                    questions_display = gr.HTML(
                        value="<div style='color:#94a3b8;font-size:0.85rem;'>Questions will appear after you get guidance — tap one to get Safe's instant answer.</div>"
                    )
                    gr.HTML('<div class="divider"></div>')
                    gr.HTML('<div class="section-label">Follow-up Answer</div>')
                    followup_output = gr.Textbox(
                        lines=6, interactive=False, label="",
                        placeholder="Tap a question or type your own — Safe answers instantly...",
                    )
                    gr.HTML('<div class="divider"></div>')
                    gr.HTML('<div class="section-label">End of Call — Session Notes</div>')
                    summarize_btn = gr.Button("📋 Generate Session Notes", variant="secondary")
                    summary_output = gr.Textbox(
                        lines=7, interactive=False, label="",
                        placeholder="Click after the call ends to generate clinical notes...",
                    )

        gr.HTML('<div class="footer">🛡️ Safe · Built at Hack Brooklyn · Tavily · ElevenLabs · AWS · Groq LLaMA 3.3</div>')

    get_guidance_btn.click(
        fn=run_safe,
        inputs=[situation_input],
        outputs=[risk_display, phrase_display, next_steps_display, questions_display, followup_input, followup_output, summary_output],
    )
    followup_btn.click(
        fn=ask_followup,
        inputs=[followup_input],
        outputs=[followup_output],
    )
    summarize_btn.click(
        fn=generate_summary,
        inputs=[],
        outputs=[summary_output],
    )

if __name__ == "__main__":
    demo.launch(share=True)