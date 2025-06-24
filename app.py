import streamlit as st
import pandas as pd
import json
import os
import requests
import sqlite3
from datetime import datetime

# --- Config ---
st.set_page_config(page_title="Fraoula Company Assistant", layout="wide")

DB_FILE = "companies.db"
DATA_STORE = "knowledge_data.json"
DEV_PASSWORD = "fraoula123"
BATCH_SIZE = 1000

# --- API Key (add yours in .streamlit/secrets.toml) ---
API_KEY = st.secrets["openrouter"]["api_key"]
API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- Styling ---
PRIMARY_COLOR = "#9400D3"
SECONDARY_COLOR = "#C779D9"
BACKGROUND_COLOR = "#1E003E"
TEXT_COLOR = "#FFFFFF"

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
    }}
    .stTextInput > div > div > input {{
        background-color: #2a004f;
        color: {TEXT_COLOR};
    }}
    .stButton > button {{
        background-color: {PRIMARY_COLOR};
        color: white;
    }}
    .stButton > button:hover {{
        background-color: {SECONDARY_COLOR};
    }}
    .user-message {{
        background-color: {SECONDARY_COLOR};
        padding: 10px;
        border-radius: 12px 12px 0 12px;
        margin: 8px 0;
        text-align: right;
    }}
    .bot-message {{
        background-color: #3b0070;
        padding: 10px;
        border-radius: 12px 12px 12px 0;
        margin: 8px 0;
        text-align: left;
    }}

    /* --- FIX TABS VISIBILITY --- */
    [data-testid="stTabs"] > div > button {{
        color: white !important;
        background-color: #2a004f !important;
        border: none;
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        margin-right: 4px;
        transition: all 0.3s ease;
        opacity: 1 !important;
    }}
    [data-testid="stTabs"] > div > button[aria-selected="true"] {{
        color: #ffffff !important;
        background-color: #3b0070 !important;
        border-bottom: 3px solid {SECONDARY_COLOR};
        font-weight: bold;
    }}
    [data-testid="stTabs"] > div > button:hover {{
        background-color: #4a007f !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- Database Setup ---
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS companies (
    CIN TEXT PRIMARY KEY,
    Name TEXT,
    State TEXT,
    Email TEXT
)
""")
conn.commit()

# --- Helpers ---
def normalize_columns(df):
    mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if "cin" in col_lower:
            mapping[col] = "CIN"
        elif "name" in col_lower:
            mapping[col] = "Name"
        elif "state" in col_lower:
            mapping[col] = "State"
        elif "email" in col_lower:
            mapping[col] = "Email"
    df = df.rename(columns=mapping)
    return df[[c for c in ["CIN", "Name", "State", "Email"] if c in df.columns]]

def chunk_text(text, max_chars=500):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def save_data(file_name, chunks):
    entry = {
        "file_name": file_name,
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "context": chunks
    }
    existing = []
    if os.path.exists(DATA_STORE):
        with open(DATA_STORE, "r") as f:
            existing = json.load(f)
    with open(DATA_STORE, "w") as f:
        json.dump(existing + [entry], f, indent=2)

def load_chunks():
    if not os.path.exists(DATA_STORE):
        return []
    with open(DATA_STORE, "r") as f:
        records = json.load(f)
    return [chunk for r in records for chunk in r["context"]]

def keyword_search(query, chunks, top_k=3):
    ranked = sorted(chunks, key=lambda x: sum(1 for w in query.lower().split() if w in x.lower()), reverse=True)
    return ranked[:top_k] if ranked else []

# --- Tabs ---
tab1, tab2 = st.tabs(["Dev", "Chatbot"])

# --- Dev Tab ---
with tab1:
    st.header("Developer Panel")

    if "dev_auth" not in st.session_state:
        st.session_state.dev_auth = False

    if not st.session_state.dev_auth:
        password = st.text_input("Enter Developer Password", type="password")
        if st.button("Login"):
            if password == DEV_PASSWORD:
                st.session_state.dev_auth = True
                st.success("Access granted")
            else:
                st.error("Incorrect password.")
    else:
        uploaded_file = st.file_uploader("Upload CSV, JSON, TXT, or Excel", type=["csv", "json", "txt", "xlsx"])
        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1].lower()
            raw_text = ""
            df_preview = None
            try:
                if file_type == "csv":
                    df_preview = pd.read_csv(uploaded_file)
                    raw_text = df_preview.to_string(index=False)
                elif file_type == "json":
                    data = json.load(uploaded_file)
                    raw_text = json.dumps(data, indent=2)
                    df_preview = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
                elif file_type == "txt":
                    raw_text = uploaded_file.read().decode("utf-8")
                elif file_type == "xlsx":
                    df_preview = pd.read_excel(uploaded_file)
                    raw_text = df_preview.to_string(index=False)

                chunks = chunk_text(raw_text)
                save_data(uploaded_file.name, chunks)

                if df_preview is not None and "cin" in [c.lower() for c in df_preview.columns]:
                    df = normalize_columns(df_preview)
                    df['CIN'] = df['CIN'].astype(str).str.strip().str.upper()
                    df['Name'] = df.get('Name', "").astype(str).str.lower()
                    df['State'] = df.get('State', "").astype(str)
                    df['Email'] = df.get('Email', "").astype(str)
                    df = df.drop_duplicates(subset="CIN")

                    cursor.execute("SELECT CIN FROM companies")
                    existing_cins = set(row[0].strip().upper() for row in cursor.fetchall())

                    inserted = 0
                    for _, row in df.iterrows():
                        if row["CIN"] in existing_cins:
                            continue
                        cursor.execute("INSERT INTO companies (CIN, Name, State, Email) VALUES (?, ?, ?, ?)",
                                       (row["CIN"], row.get("Name"), row.get("State"), row.get("Email")))
                        inserted += 1
                        existing_cins.add(row["CIN"])
                    conn.commit()
                    st.success(f"Uploaded {inserted} new companies to database.")

                if df_preview is not None:
                    st.dataframe(df_preview)
                else:
                    st.text_area("Preview", raw_text, height=200)
            except Exception as e:
                st.error(f"Failed to read file: {e}")

        if os.path.exists(DATA_STORE):
            with open(DATA_STORE, "r") as f:
                data_log = json.load(f)
            st.download_button(" Download Upload Log (JSON)",
                               data=json.dumps(data_log, indent=2),
                               file_name="upload_log.json",
                               mime="application/json")

# --- Chatbot Tab ---
with tab2:
    st.title("Fraoula Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chunks = load_chunks()
    for msg in st.session_state.chat_history:
        cls = "user-message" if msg["role"] == "user" else "bot-message"
        st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([8, 2])
        user_input = col1.text_input("You:", placeholder="Ask anything...")
        if col2.form_submit_button("Send") and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            matches = keyword_search(user_input, chunks)
            context = "\n---\n".join(matches)
            messages = []
            if context:
                messages.append({"role": "system", "content": f"Use this info:\n{context}"})
            messages += [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]

            payload = {
                "model": "meta-llama/llama-3.3-8b-instruct:free",
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.7
            }
            try:
                res = requests.post(API_URL, headers=HEADERS, json=payload)
                res.raise_for_status()
                reply = res.json()["choices"][0]["message"]["content"]
            except Exception as e:
                reply = f"Error: {e}"

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()
