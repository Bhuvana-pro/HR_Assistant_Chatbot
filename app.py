import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# Correct imports for LangChain 0.3.x+
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval import RetrievalQA
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import datetime


# --- Google Sheets Auth ---
creds = Credentials.from_service_account_info(st.secrets["google_service_account"])
client = gspread.authorize(creds)

# Open your HR Assistant sheet by URL
spreadsheet = client.open_by_url(
    "https://docs.google.com/spreadsheets/d/13CpT3bevuQoPb2nw-woGmzuZ9nopZfA-pbdwr4XILZ0/edit#gid=0"
)

# --- Load Policies ---
policy_data = spreadsheet.worksheet("Policies").get_all_records()
policy_docs = [f"Policy: {row['Title']} ({row['Section']}) → {row['Content']}" for row in policy_data]

# --- Load Benefits ---
benefit_data = spreadsheet.worksheet("Benefits").get_all_records()
benefit_docs = [
    f"Benefit: {row['BenefitName']} → {row['Description']} | Eligibility: {row['Eligibility']} | Notes: {row['Notes']}"
    for row in benefit_data
]

# --- Load LeaveBalances ---
leave_data = spreadsheet.worksheet("LeaveBalances").get_all_records()
leave_docs = [
    f"Leave balance for {row['EmployeeEmail']}: Casual={row['CasualLeave']}, Sick={row['SickLeave']} (Last updated {row['LastUpdated']})"
    for row in leave_data
]

# --- Combine all HR docs ---
docs = policy_docs + benefit_docs + leave_docs

# --- Build ChromaDB ---
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents(docs)

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai"]["api_key"])
db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")

retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", api_key=st.secrets["openai"]["api_key"]),
    retriever=retriever,
    return_source_documents=True
)

# --- Streamlit UI ---
st.title("HR Assistant 🤖")
st.write("Ask me about HR policies, benefits, or your leave balance!")

query = st.text_input("Enter your question:")
if query:
    result = qa(query)
    answer = result["result"]

    # Human-like response
    st.write("### Answer:")
    st.write(f"{answer}\n\nI hope that clarifies your query!")

    # --- Log Q&A ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        log_sheet = spreadsheet.worksheet("Logs")
        log_sheet.append_row([timestamp, query, answer])
        st.success("Interaction logged to Google Sheet ✅")
    except Exception:
        st.warning("Could not log interaction. Make sure you have a 'Logs' tab in your sheet.")
