import streamlit as st
import requests
import pandas as pd
import os
import pyttsx3
import speech_recognition as sr

from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

llm = OllamaLLM(base_url="http://127.0.0.1:11434", model="llama3")
# ---------------- Page Config ----------------
st.set_page_config(page_title="GK AI Assistant", page_icon="🤖", layout="wide")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)
    st.markdown("## 🤖 GK AI Assistant")
    st.markdown("""
        Welcome to **GK AI**, your intelligent assistant powered by **DeepSeek-R1**.
        🧠 Ask questions, get explanations, or see real-time data.
    """)

    st.divider()
    theme = st.radio("🎨 Choose Theme", ["Light", "Dark"], horizontal=True)
    model_name = st.selectbox("⚙️ Model", ["deepseek-r1", "llama3", "mistral"])
    st.caption("💡 DeepSeek-R1 gives strong reasoning and coding support.")

    st.divider()
    st.markdown("### 🎤 Voice Options")
    use_voice_input = st.checkbox("🎙️ Enable Voice Input")
    use_voice_output = st.checkbox("🗣️ Enable Voice Output")

    if st.button("🧹 Clear Chat History"):
        st.session_state["messages"] = []
        st.experimental_rerun()

# ---------------- Styling ----------------
if theme == "Dark":
    st.markdown("""
    <style>
    body {background-color: #0e1117; color: white;}
    .stTextInput > div > div > input {background-color: #262730; color: white;}
    </style>
    """, unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown("<h1 style='text-align: center;'>💬 GK AI</h1>", unsafe_allow_html=True)
st.write("Ask or speak your query — try ‘weather’, ‘stock’, ‘crypto’, or ‘news’ 👇")

# ---------------- Session Memory ----------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---------------- Display Chat History ----------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Setup LLM ----------------
model = OllamaLLM(model=model_name)

# ---------------- Vector DB ----------------
def load_vector_db():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = "./vector_memory"
    os.makedirs(persist_dir, exist_ok=True)

    if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    else:
        with open("./data/knowledge.txt", "r", encoding="utf-8") as f:
            content = f.read()
        docs = [Document(page_content=content)]
        vectordb = Chroma.from_documents(docs, embedding_model, persist_directory=persist_dir)
        vectordb.persist()
    return vectordb

vectordb = load_vector_db()
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff")

# ---------------- Voice Functions ----------------
def listen_to_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... please speak now.")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"🗣️ You said: {text}")
        return text
    except Exception as e:
        st.error(f"Voice recognition error: {e}")
        return ""

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ---------------- Data Functions ----------------
def get_weather(city):
    try:
        data = requests.get(f"https://wttr.in/{city}?format=j1").json()
        current = data["current_condition"][0]
        return {
            "City": city.title(),
            "Temperature (°C)": current["temp_C"],
            "Feels Like (°C)": current["FeelsLikeC"],
            "Humidity (%)": current["humidity"],
            "Weather": current["weatherDesc"][0]["value"]
        }
    except Exception as e:
        return {"Error": str(e)}

def get_stock(symbol):
    try:
        data = requests.get(f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}").json()
        quote = data["quoteResponse"]["result"][0]
        return {
            "Symbol": symbol.upper(),
            "Current Price": quote["regularMarketPrice"],
            "Change": quote["regularMarketChange"],
            "Percent Change": f"{quote['regularMarketChangePercent']:.2f}%",
            "Market Time": quote["regularMarketTime"]
        }
    except Exception as e:
        return {"Error": str(e)}

def get_crypto(symbol="bitcoin"):
    try:
        data = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd").json()
        return {"Crypto": symbol.title(), "Price (USD)": data[symbol]["usd"]}
    except Exception as e:
        return {"Error": str(e)}

def get_news():
    try:
        data = requests.get("https://inshortsapi.vercel.app/news?category=technology").json()
        articles = data.get("data", [])[:5]
        df = pd.DataFrame([
            {"Title": a["title"], "Source": a["source"], "Date": a["date"], "URL": a["url"]}
            for a in articles
        ])
        return df
    except Exception as e:
        return pd.DataFrame([{"Error": str(e)}])

# ---------------- Input Section ----------------
question = ""

if use_voice_input:
    if st.button("🎙️ Speak"):
        question = listen_to_voice()
else:
    question = st.chat_input("Ask your question...")

if question:
    st.chat_message("user").markdown(question)
    st.session_state["messages"].append({"role": "user", "content": question})

    with st.spinner("Thinking... 🤔"):
        q = question.lower()

        if "weather" in q:
            city = q.split("in")[-1].strip() or "Delhi"
            data = get_weather(city)
            st.markdown(f"### 🌤️ Live Weather in {city.title()}")
            st.dataframe(pd.DataFrame([data]))
            answer = f"Here’s the real-time weather data for **{city.title()}**."

        elif "stock" in q:
            symbol = q.split()[-1].upper()
            data = get_stock(symbol)
            st.markdown(f"### 📈 Stock Price for {symbol}")
            st.dataframe(pd.DataFrame([data]))
            answer = f"Here’s the live stock update for **{symbol}**."

        elif "crypto" in q:
            symbol = q.split()[-1].lower()
            data = get_crypto(symbol)
            st.markdown(f"### 💰 Cryptocurrency Price")
            st.dataframe(pd.DataFrame([data]))
            answer = f"Here’s the live price for **{symbol.title()}**."

        elif "news" in q:
            df = get_news()
            st.markdown("### 📰 Latest Tech News")
            st.dataframe(df)
            answer = "Here are the top 5 latest technology news headlines."

        else:
            answer = qa_chain.run(question)

    st.chat_message("assistant").markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})

    if use_voice_output:
        speak_text(answer)
