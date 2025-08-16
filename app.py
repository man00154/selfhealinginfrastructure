# app.py
import os
import requests
import streamlit as st
from typing import List

# ---------- Gemini Config ----------
MODEL_NAME = "gemini-2.0-flash-lite"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

def get_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))

# ---------- Tiny RAG (Knowledge Base) ----------
class TinyRAG:
    def __init__(self):
        self.docs: List[str] = [
            "High CPU usage may indicate process bottlenecks.",
            "Disk space alerts can lead to service crashes.",
            "Network latency spikes may degrade performance.",
            "Memory leaks can cause application instability.",
            "Automatic remediation can restart services or free resources."
        ]

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        query_words = query.lower().split()
        scored = []
        for doc in self.docs:
            score = sum(1 for w in query_words if w in doc.lower())
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[:k]]

# ---------- Gemini Client ----------
def gemini_generate(api_key: str, prompt: str, max_tokens: int = 300) -> str:
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens}
    }
    try:
        resp = requests.post(API_URL, headers=headers, params=params, json=payload, timeout=30)
        data = resp.json()
    except Exception as e:
        return f"⚠️ Network/Request error: {e}"

    if "candidates" in data and data["candidates"]:
        cand = data["candidates"][0]
        parts = cand.get("content", {}).get("parts", [])
        for p in parts:
            if "text" in p and p["text"]:
                return p["text"].strip()
    return "⚠️ Gemini API returned no usable text."

# ---------- Agentic AI Pipeline ----------
def agentic_self_healing(api_key: str, incident: str, rag: TinyRAG) -> str:
    context = " ".join(rag.retrieve(incident))
    prompt = f"""
You are an expert Data Center AI assistant for self-healing infrastructure.
Analyze the following incident or system log and provide a proactive remediation plan.

Incident:
{incident}

Knowledge Base:
{context}

Tasks:
1. Predict potential issues
2. Generate an automatic remediation plan
3. Suggest monitoring improvements
4. Provide concise explanation for each step

Return a structured, actionable report.
"""
    return gemini_generate(api_key, prompt, max_tokens=350)

# ---------- Optional LangGraph ----------
try:
    import langgraph as lg
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Self-Healing Infrastructure AI", layout="wide")
st.title("🤖 MANISH - DATA CENTER - Self-Healing Infrastructure Co-Pilot")

incident_input = st.text_area(
    "Paste recent system logs, alerts, or incident summary:",
    height=200,
    placeholder="Example: CPU spikes to 95%, Disk space < 5%, Service X failed..."
)

if st.button("Run Self-Healing Analysis"):
    api_key = get_api_key()
    if not api_key:
        st.error("GEMINI_API_KEY not found. Set environment variable or Streamlit secrets.")
    elif not incident_input.strip():
        st.warning("Please enter system logs or incidents.")
    else:
        rag = TinyRAG()
        if LANGGRAPH_AVAILABLE:
            g = lg.Graph()
            g.add_node("incident_input", {"length": len(incident_input)})
        with st.spinner("Analyzing and generating remediation plan..."):
            report = agentic_self_healing(api_key, incident_input.strip(), rag)
        st.subheader("📝 Self-Healing Report")
        st.markdown(report)
        if LANGGRAPH_AVAILABLE:
            g.add_node("self_healing_output", {"length": len(report)})
            g.add_edge("incident_input", "self_healing_output")
            st.info("Pipeline recorded in LangGraph (local).")
