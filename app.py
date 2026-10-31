import time
import streamlit as st
import chromadb
from google import genai

st.set_page_config(page_title="FAQ Assistant")
st.title("FAQ चैटबॉट")
st.caption("यहाँ दी गई जानकारी के आधार पर जवाब मिलेगा")

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

threshold = st.sidebar.slider("दूरी की सीमा", 0.0, 1.0, 0.5, 0.05)
debug = st.sidebar.checkbox("दूरी दिखाओ")

@st.cache_data
def load_faq():
    with open("faq.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

faq = load_faq()

def embed(texts):
    for n in range(3):
        try:
            r = client.models.embed_content(model="gemini-embedding-001", contents=texts)
            return [e.values for e in r.embeddings]
        except Exception:
            time.sleep(3 * (n + 1))
    raise RuntimeError("embedding service unavailable")

@st.cache_resource
def build_index():
    db = chromadb.Client()
    col = db.get_or_create_collection("faq", metadata={"hnsw:space": "cosine"})
    col.upsert(
        ids=[f"c{i}" for i in range(len(faq))],
        documents=faq,
        embeddings=embed(faq),
    )
    return col

collection = build_index()

def answer(question):
    r = collection.query(query_embeddings=embed([question]), n_results=3)
    docs = r["documents"][0]
    dist = r["distances"][0][0]
    if dist > threshold:
        return "मुझे नहीं पता", dist
    prompt = f"""नीचे दिए संदर्भ के आधार पर ही सवाल का जवाब दो।
अगर संदर्भ में जवाब न हो तो कहो "मुझे नहीं पता"।

संदर्भ:
{chr(10).join(docs)}

सवाल: {question}"""
    try:
        resp = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
        return resp.text, dist
    except Exception:
        return "(अभी जवाब बनाने वाला सर्वर व्यस्त है) सबसे करीब की जानकारी: " + docs[0], dist

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)

question = st.chat_input("अपना सवाल पूछिए")
if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            text, dist = answer(question)
        except Exception:
            text, dist = "अभी सेवा उपलब्ध नहीं है, कुछ देर बाद कोशिश करें।", None
        st.write(text)
        if debug and dist is not None:
            st.caption(f"दूरी: {dist:.3f}")
    st.session_state.history.append(("assistant", text))
