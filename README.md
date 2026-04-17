# 🛡️ Safe

**Real-time clinical guidance for crisis counselors.**
*So no one is ever alone on the hardest calls.*

---

## Why I Built This

Every mental health AI out there is built for the person in crisis. Chatbots, mood trackers, guided journaling apps — all patient-facing.

But what about the counselor on the other end of the line?

They pick up calls with zero support. No guidance. No safety net. Just them and a person in their most vulnerable moment. They're expected to recall every intervention technique, every grounding exercise, every de-escalation protocol — under pressure, in real time.

Nobody has built anything for them. So I did.

---

## What Safe Does

Safe is a RAG-powered AI co-pilot built for crisis hotline counselors — not patients.

The counselor types a quick situation summary mid-call. Safe:

1. Searches vetted clinical literature stored in S3
2. Pulls live mental health resources via Tavily
3. Synthesizes a calm, structured response via Claude Haiku
4. Reads the suggested phrase aloud via ElevenLabs — so the counselor never has to look away

The model only responds from retrieved context. It cannot hallucinate clinical advice. If it's not in the documents, Safe won't say it.

---

## How It Feels to Use It

A counselor types:

> *"Caller is a 19 year old male, first time calling, says he feels completely numb and disconnected"*

Safe comes back with the right grounding technique pulled from clinical literature, a phrase they can say out loud right now, and a risk level. If the situation looks like it could escalate, a red banner fires immediately.

At the end of the call, one click drafts the session notes. No paperwork at midnight.

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| LLM | Claude Haiku 4.5 | Chosen for accuracy, not agreeability |
| RAG Framework | LlamaIndex | Clinical document ingestion + retrieval |
| Live Search | **Tavily API** ✅ *sponsor* | Real-time web search for current resources |
| Embeddings | HuggingFace MiniLM | Local semantic embeddings, zero API cost |
| Vector Store | FAISS | Fast similarity search across clinical chunks |
| Voice Output | **ElevenLabs API** ✅ *sponsor* | Reads suggestions aloud, hands-free |
| UI | Gradio | Clean chat interface |
| Document Store | AWS S3 | Stores clinical PDFs + FAISS index |
| Hosting | AWS EC2 t2.micro | Live public URL, free tier |

---

## 🗂️ Project Structure

```
safe/
├── app.py              # Gradio UI — chat, risk flag banner, audio playback
├── rag_pipeline.py     # LlamaIndex ingestion, FAISS indexing, retrieval
├── tavily_search.py    # Live web search via Tavily API
├── claude_client.py    # Anthropic API wrapper, prompt construction, parsing
├── voice_output.py     # ElevenLabs text-to-speech integration
├── risk_flags.py       # Escalation pattern detection
├── s3_loader.py        # Pulls PDFs and FAISS index from S3 on startup
├── requirements.txt    # All dependencies pinned
├── .env.example        # Environment variable template
└── docs/               # Clinical PDFs (not committed — stored in S3)
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- AWS account (free tier works)
- API keys for Claude, Tavily, and ElevenLabs

### 1. Clone the repo
```bash
git clone https://github.com/nuzhatk2021/safe.git
cd safe
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
ANTHROPIC_API_KEY=your_anthropic_key
TAVILY_API_KEY=your_tavily_key
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_voice_id
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_bucket_name
```

### 4. Upload clinical documents to S3
```bash
aws s3 sync docs/ s3://your-bucket-name/docs/
```

### 5. Build the FAISS index
```bash
python rag_pipeline.py --build
```

### 6. Run the app
```bash
python app.py
# Opens at http://localhost:7860
```

---

## ☁️ Deploy on AWS EC2

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Clone, install, and run
git clone https://github.com/nuzhatk2021/safe.git
cd safe
pip install -r requirements.txt
cp .env.example .env

# Run persistently with tmux
tmux new -s safe
python app.py
# Ctrl+B then D to detach — app keeps running after you disconnect
```

App runs on port `7860`. Open `http://your-ec2-ip:7860` in any browser.

---

## 🧠 How It Works

1. **On startup** — S3 loader pulls clinical PDFs and the serialized FAISS index from your bucket
2. **Query comes in** — risk flags scan for escalation patterns and raise a banner if triggered
3. **Retrieval** — MiniLM embeds the query, FAISS finds the top matching clinical chunks
4. **Live search** — Tavily fetches current web results alongside the local retrieval
5. **Generation** — Claude constructs a response grounded only in the retrieved context
6. **Voice** — ElevenLabs reads the suggested phrase aloud through the Gradio UI

---

## 📄 Clinical Documents

Safe retrieves from publicly available sources:

- SAMHSA Crisis Counseling Program Guidelines
- SAMHSA Treatment Improvement Protocol — Brief Interventions
- Beck Institute CBT Resource Library
- DBT Skills Training Manual (public summary)
- AFSP Safe Messaging Guidelines for Mental Health Professionals
- NIMH Suicide Prevention Resource Center fact sheets

---

## ⚠️ A Note on Risk Flagging

Safe detects escalation language and surfaces a warning banner before the response is shown. This is a pattern-matching layer — not a replacement for clinical judgment. Always follow your organization's crisis protocol.

---

## 🏆 Built At

**Hack Brooklyn** — built in under 24 hours

Sponsors integrated: **Tavily** (live search) · **ElevenLabs** (voice output)

---

*Built with care for the people who answer the hardest calls.*