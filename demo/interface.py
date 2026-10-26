import sys, os, io, hashlib
sys.dont_write_bytecode = True

import time
from dotenv import load_dotenv

import pandas as pd
import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from embeddings import create_embeddings

from llm_agent import ChatBot
from ingest_data import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

print(DATA_PATH)
print(FAISS_PATH)

welcome_message = """
  #### Introduction 🚀

  The system is a RAG pipeline designed to assist hiring managers in searching for the most suitable candidates out of thousands of resumes more effectively. ⚡

  The idea is to use a similarity retriever to identify the most suitable applicants with job descriptions.
  This data is then augmented into an LLM generator for downstream tasks such as analysis, summarization, and decision-making. 

  #### Getting started 🛠️

  1. Choose Ollama, OpenAI, Groq, or Hugging Face in the sidebar.
  2. Enter an API key for a hosted provider, then type a job description. 💬

  Hint: The knowledge base of the LLM has been loaded with a pre-existing vectorstore of [resumes](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/blob/main/data/main-data/synthetic-resumes.csv) to be used right away. 
  In addition, you may also find example job descriptions to test [here](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/blob/main/data/supplementary-data/job_title_des.csv).

  Please make sure to check the sidebar for more useful information. 💡
"""

info_message = """
  # Information

  ### 1. What if I want to use my own resumes?

  If you want to load in your own resumes file, simply use the uploading button above. 
  Please make sure to have the following column names: `Resume` and `ID`. 

  Keep in mind that the indexing process can take **quite some time** to complete. ⌛

  ### 2. What if I want to set my own parameters?

  You can change the RAG mode, provider, and model using the sidebar options above.

  About the other parameters such as the generator's *temperature* or retriever's *top-K*, I don't want to allow modifying them for the time being to avoid certain problems. 
  FYI, the temperature is currently set at `0.1` and the top-K is set at `5`.  

  ### 3. Is my uploaded data safe? 

  Your data is not being stored anyhow by the program. Everything is recorded in a Streamlit session state and will be removed once you refresh the app. 

  Uploaded data is processed by the selected AI provider. Do not upload sensitive data
  to a hosted provider unless your organization's policy permits it.

  ### 4. How does the chatbot work? 

  The Chatbot works a bit differently to the original structure proposed in the paper so that it is more usable in practical use cases.

  For example, the system classifies the intent of every single user prompt to know whether it is appropriate to toggle RAG retrieval on/off. 
  The system also records the chat history and chooses to use it in certain cases, allowing users to ask follow-up questions or tasks on the retrieved resumes.
"""

about_message = """
  # About

  This small program is a prototype designed out of pure interest as additional work for the author's Bachelor's thesis project. 
  The aim of the project is to propose and prove the effectiveness of RAG-based models in resume screening, thus inspiring more research into this field.

  The program is very much a work in progress. I really appreciate any contribution or feedback on [GitHub](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline).

  If you are interested, please don't hesitate to give me a star. ⭐
"""


st.set_page_config(page_title="Resume Screening GPT")
st.title("Resume Screening GPT")

if "chat_history" not in st.session_state:
  st.session_state.chat_history = [AIMessage(content=welcome_message)]

if "df" not in st.session_state:
  st.session_state.df = pd.read_csv(DATA_PATH)

if "embedding_model" not in st.session_state:
  st.session_state.embedding_model = create_embeddings()

if "rag_pipeline" not in st.session_state:
  vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
  st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

if "resume_list" not in st.session_state:
  st.session_state.resume_list = []



def clear_message():
  st.session_state.resume_list = []
  st.session_state.chat_history = [AIMessage(content=welcome_message)]



user_query = st.chat_input("Type your message here...")

with st.sidebar:
  st.markdown("# Control Panel")

  st.selectbox("RAG Mode", ["Generic RAG", "RAG Fusion"], placeholder="Generic RAG", key="rag_selection")
  st.selectbox("AI Provider", ["Ollama", "OpenAI", "Groq", "Hugging Face"], key="provider_selection")
  if st.session_state.provider_selection == "Ollama":
    st.text_input("Ollama Model", OLLAMA_LLM_MODEL, key="ollama_model")
    st.text_input("Ollama URL", OLLAMA_BASE_URL, key="ollama_url")
    st.caption("Ollama runs locally and normally does not require an API key.")
  elif st.session_state.provider_selection == "OpenAI":
    st.text_input("OpenAI API Key", type="password", key="api_key")
    st.text_input("OpenAI Model", "gpt-4o-mini", key="openai_model")
  elif st.session_state.provider_selection == "Groq":
    st.text_input("Groq API Key", value=GROQ_API_KEY, type="password", key="api_key")
    st.text_input("Groq Model", "llama-3.3-70b-versatile", key="groq_model")
    st.caption("Groq uses an OpenAI-compatible API.")
  else:
    st.text_input("Hugging Face API Key", type="password", key="api_key")
    st.text_input("Hugging Face Model", "Qwen/Qwen2.5-72B-Instruct", key="huggingface_model")
    st.caption("Hugging Face automatically selects an enabled provider for this model.")
  uploaded_file = st.file_uploader("Upload resumes", type=["csv"], key="uploaded_file")
  st.button("Clear conversation", on_click=clear_message)

  st.divider()
  st.markdown(info_message)

  st.divider()
  st.markdown(about_message)
  st.markdown("Made by [Hungreeee](https://github.com/Hungreeee)")

if uploaded_file is not None:
  uploaded_bytes = uploaded_file.getvalue()
  upload_fingerprint = hashlib.sha256(uploaded_bytes).hexdigest()
  if st.session_state.get("loaded_upload_fingerprint") != upload_fingerprint:
    try:
      uploaded_df = pd.read_csv(io.BytesIO(uploaded_bytes))
    except Exception as error:
      st.error(f"Could not read the uploaded CSV: {error}")
      st.stop()

    required_columns = {"ID", "Resume"}
    missing_columns = required_columns.difference(uploaded_df.columns)
    if missing_columns:
      st.error(f"Uploaded CSV is missing required columns: {', '.join(sorted(missing_columns))}.")
      st.stop()
    if uploaded_df.empty:
      st.error("The uploaded CSV does not contain any resume rows.")
      st.stop()

    with st.spinner(f"Indexing {len(uploaded_df):,} uploaded resumes..."):
      vectordb = ingest(uploaded_df, "Resume", st.session_state.embedding_model)
    st.session_state.df = uploaded_df
    st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, uploaded_df)
    st.session_state.loaded_upload_fingerprint = upload_fingerprint
    st.session_state.loaded_dataset_name = uploaded_file.name
    st.success(f"Loaded {len(uploaded_df):,} resumes from `{uploaded_file.name}`.")

active_dataset_name = st.session_state.get("loaded_dataset_name", os.path.basename(DATA_PATH))
active_dataset_rows = len(st.session_state.df)
st.info(f"Active resume dataset: `{active_dataset_name}` ({active_dataset_rows:,} resumes)")

for message in st.session_state.chat_history:
  if isinstance(message, AIMessage):
    with st.chat_message("AI"):
      st.write(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message("Human"):
      st.write(message.content)
  else:
    with st.chat_message("AI"):
      message[0].render(*message[1:])


retriever = st.session_state.rag_pipeline

if st.session_state.provider_selection != "Ollama" and not st.session_state.get("api_key", "").strip():
  st.info("Enter your provider API key in the sidebar to use a hosted model.")
  st.stop()

model_key_by_provider = {
  "Ollama": "ollama_model",
  "OpenAI": "openai_model",
  "Groq": "groq_model",
  "Hugging Face": "huggingface_model",
}
selected_model = st.session_state.get(model_key_by_provider[st.session_state.provider_selection])
if not selected_model:
  st.error(f"Enter a model name for {st.session_state.provider_selection}.")
  st.stop()

llm = ChatBot(
  provider=st.session_state.provider_selection,
  model=selected_model,
  api_key=st.session_state.get("api_key", ""),
  base_url=st.session_state.get("ollama_url", "") if st.session_state.provider_selection == "Ollama" else (
    "https://api.groq.com/openai/v1" if st.session_state.provider_selection == "Groq" else ""
  ),
)

if user_query is not None and user_query != "":
  with st.chat_message("Human"):
    st.markdown(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))

  with st.chat_message("AI"):
    start = time.time()
    try:
      with st.spinner("Generating answers..."):
        document_list = retriever.retrieve_docs(user_query, llm, st.session_state.rag_selection)
        query_type = retriever.meta_data["query_type"]
        st.session_state.resume_list = document_list
        stream_message = llm.generate_message_stream(user_query, document_list, [], query_type)
      end = time.time()
      response = st.write_stream(stream_message)
    except Exception as error:
      st.error(f"{st.session_state.provider_selection} request failed: {error}")
      st.stop()
    
    retriever_message = chatbot_verbosity
    retriever_message.render(document_list, retriever.meta_data, end-start)

    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.chat_history.append((retriever_message, document_list, retriever.meta_data, end-start))