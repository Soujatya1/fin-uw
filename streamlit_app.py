import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd

st.set_page_config(
    page_title="Financial Underwriting Assistant",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Financial Underwriting Assistant")

template = """
Hello, AI Financial Underwriting Assistant. You are a specialized AI agent with expertise in financial underwriting for insurance products. Your role is to analyze customer financial documents and assess their financial viability for insurance policies based on the provided underwriting guidelines.

Your analysis should focus on:

**Financial Document Analysis:**
- Extract and analyze key financial information from salary slips, ITR documents, mutual fund statements, bank statements, etc.
- Calculate income stability, debt-to-income ratios, and financial capacity
- Assess financial history and patterns

**Risk Assessment Parameters:**
- Annual Income and Income Stability
- Employment Status and Duration
- Debt Obligations and Financial Commitments
- Investment Portfolio and Assets
- Financial History and Credit Profile
- Premium Affordability Analysis

**Financial Viability Determination:**
- Determine if the customer can afford the proposed insurance premium
- Assess long-term financial sustainability
- Calculate recommended coverage amounts based on financial capacity
- Identify any financial red flags or concerns

**Detailed Financial Report:**
Create a comprehensive tabular analysis covering:
| Parameter | Customer Value | Guideline Reference | Risk Assessment | Comments |
|-----------|---------------|-------------------|-----------------|----------|

**Financial Scoring:**
Provide scores in the following format:
- Income Stability Score: (0-100)
- Debt Management Score: (0-100)
- Premium Affordability Score: (0-100)
- Overall Financial Risk Score: High Risk (0-30), Medium Risk (31-60), Low Risk (61-100)

**Recommendation:**
- Policy Eligibility: Approved/Conditional/Declined
- Recommended Coverage Amount
- Premium Payment Frequency Recommendation
- Any additional financial requirements or conditions

Question: {question}
Context from Guidelines: {guidelines_context}
Customer Financial Documents: {customer_context}
Answer:
"""

guidelines_directory = '.github/guidelines/'
customer_docs_directory = '.github/customer_docs/'

os.makedirs(guidelines_directory, exist_ok=True)
os.makedirs(customer_docs_directory, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
guidelines_vector_store = InMemoryVectorStore(embeddings)
customer_docs_vector_store = InMemoryVectorStore(embeddings)

model = ChatGroq(
    groq_api_key="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", 
    model_name="meta-llama/llama-4-maverick-17b-128e-instruct", 
    temperature=0.3
)

def upload_pdf(file, directory):
    file_path = directory + file.name
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def extract_financial_info(documents):
    financial_keywords = [
        "salary", "income", "annual income", "monthly income", 
        "basic pay", "gross salary", "net salary", "CTC",
        "ITR", "income tax return", "form 16", "tax",
        "mutual fund", "SIP", "investment", "portfolio",
        "bank statement", "account balance", "savings",
        "loan", "EMI", "debt", "liability", "credit",
        "bonus", "incentive", "allowance", "deduction"
    ]
    
    relevant_chunks = []
    for doc in documents:
        content_lower = doc.page_content.lower()
        if any(keyword in content_lower for keyword in financial_keywords):
            relevant_chunks.append(doc)
    
    return relevant_chunks

def analyze_customer_finances(question, guidelines_docs, customer_docs):
    guidelines_context = "\n\n".join([doc.page_content for doc in guidelines_docs])
    
    financial_docs = extract_financial_info(customer_docs)
    customer_context = "\n\n".join([doc.page_content for doc in financial_docs])
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    response = chain.invoke({
        "question": question,
        "guidelines_context": guidelines_context,
        "customer_context": customer_context
    })
    
    return response.content

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "guidelines_loaded" not in st.session_state:
    st.session_state.guidelines_loaded = False
if "customer_docs_loaded" not in st.session_state:
    st.session_state.customer_docs_loaded = False

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Upload Financial Underwriting Guidelines")    
    guidelines_files = st.file_uploader(
        "Choose Guidelines PDF files",
        type="pdf",
        accept_multiple_files=True,
        key="guidelines_uploader"
    )
    
    if guidelines_files and not st.session_state.guidelines_loaded:
        with st.spinner("Processing guidelines..."):
            all_guideline_docs = []
            for file in guidelines_files:
                file_path = upload_pdf(file, guidelines_directory)
                documents = load_pdf(file_path)
                chunked_documents = split_text(documents)
                all_guideline_docs.extend(chunked_documents)
            
            guidelines_vector_store.add_documents(all_guideline_docs)
            st.session_state.guidelines_loaded = True
            st.success(f"✅ {len(guidelines_files)} guideline document(s) processed successfully!")

with col2:
    st.markdown("### 💼 Upload Customer Financial Documents")
    
    customer_files = st.file_uploader(
        "Choose Customer Financial Documents",
        type="pdf",
        accept_multiple_files=True,
        key="customer_uploader"
    )
    
    if customer_files:
        with st.spinner("Processing customer documents..."):
            all_customer_docs = []
            for file in customer_files:
                file_path = upload_pdf(file, customer_docs_directory)
                documents = load_pdf(file_path)
                chunked_documents = split_text(documents)
                all_customer_docs.extend(chunked_documents)
            
            customer_docs_vector_store.add_documents(all_customer_docs)
            st.session_state.customer_docs_loaded = True
            st.success(f"✅ {len(customer_files)} customer document(s) processed successfully!")

if st.session_state.guidelines_loaded and st.session_state.customer_docs_loaded:
    st.success("🎉 All documents loaded! Ready for financial analysis.")
elif st.session_state.guidelines_loaded:
    st.warning("Guidelines loaded. Please upload customer financial documents.")
elif st.session_state.customer_docs_loaded:
    st.warning("Customer documents loaded. Please upload underwriting guidelines.")
else:
    st.info("📤 Please upload both guidelines and customer financial documents to begin analysis.")

if st.session_state.guidelines_loaded and st.session_state.customer_docs_loaded:
    st.markdown("---")
    st.markdown("### 🔍 Financial Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💰 Income Analysis", use_container_width=True):
            st.session_state.conversation_history.append({
                "role": "user", 
                "content": "Analyze the customer's income sources, stability, and adequacy for insurance premium payments."
            })
    
    with col2:
        if st.button("📊 Financial Capacity", use_container_width=True):
            st.session_state.conversation_history.append({
                "role": "user", 
                "content": "Assess the customer's overall financial capacity and recommend appropriate coverage amount."
            })
    
    with col3:
        if st.button("⚖️ Risk Assessment", use_container_width=True):
            st.session_state.conversation_history.append({
                "role": "user", 
                "content": "Provide a comprehensive financial risk assessment and policy eligibility recommendation."
            })
    
    question = st.chat_input("Ask about financial underwriting analysis...")
    
    if question:
        st.session_state.conversation_history.append({"role": "user", "content": question})
    
    if st.session_state.conversation_history and st.session_state.conversation_history[-1]["role"] == "user":
        with st.spinner("Analyzing financial documents..."):
            latest_question = st.session_state.conversation_history[-1]["content"]
            guidelines_docs = guidelines_vector_store.similarity_search(latest_question, k=5)
            customer_docs = customer_docs_vector_store.similarity_search(latest_question, k=10)
            
            answer = analyze_customer_finances(latest_question, guidelines_docs, customer_docs)
            
            st.session_state.conversation_history.append({"role": "assistant", "content": answer})
    
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])

with st.sidebar:
    if st.button("🗑️ Clear Analysis History"):
        st.session_state.conversation_history = []
        st.rerun()
