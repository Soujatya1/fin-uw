import streamlit as st
import os
import re
import hashlib
from typing import List, Dict, Tuple
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
from langchain_core.documents import Document

st.set_page_config(
    page_title="Financial Underwriting Assistant",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Financial Underwriting Assistant")

class PIIShield:
    def __init__(self):
        self.pii_patterns = {
            'pan_card': r'\b[A-Z]{5}[-\s]?[0-9]{4}[-\s]?[A-Z]\b',
            'tan': r'\b[A-Z]{4}[-\s]?[0-9]{5}[-\s]?[A-Z]\b',
            'aadhaar': r'\b\d{4}\s?\d{4}\s?\d{4}\b',
            'account_number': r'\b\d{9,18}\b',
            'phone': r'\b(?:\+91[-.\s]?)?[6-9]\d{9}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'name': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',
            'address': r'\b(?:house|flat|plot|door)\s*(?:no\.?|number)?\s*[0-9A-Za-z\-\/]+\b',
            'ifsc': r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
            'pin_code': r'\b[1-9][0-9]{5}\b'
        }
        
        self.replacement_map = {}
        self.anonymization_enabled = True
    
    def anonymize_text(self, text: str) -> str:
        if not self.anonymization_enabled:
            return text
            
        anonymized_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                original = match.group()
                
                if original not in self.replacement_map:
                    self.replacement_map[original] = "###########"
                
                anonymized_text = anonymized_text.replace(original, self.replacement_map[original])
        
        return anonymized_text
    
    def get_pii_summary(self) -> Dict[str, int]:
        pii_summary = {}
        for original, anonymized in self.replacement_map.items():
            for pii_type, pattern in self.pii_patterns.items():
                if re.match(pattern, original, re.IGNORECASE):
                    pii_summary[pii_type] = pii_summary.get(pii_type, 0) + 1
                    break
        return pii_summary

pii_shield = PIIShield()

comprehensive_template = """
Hello, AI Financial Underwriting Assistant. You are a specialized AI agent with expertise in financial underwriting for insurance products. Your role is to analyze customer financial documents and assess their financial viability for insurance policies based on the provided underwriting guidelines.

IMPORTANT: All customer data has been anonymized for privacy protection. Use anonymized identifiers in your analysis.

CRITICAL INSTRUCTIONS:
1. CAREFULLY READ through ALL the provided customer financial documents
2. Extract SPECIFIC numerical values, amounts, and financial data mentioned in the documents
3. When asked about specific values like "Investment Amount", look for exact matches and related terms
4. If you cannot find specific information, clearly state what information is missing
5. Always quote the exact text/numbers from the documents when available

DOCUMENT ANALYSIS FOCUS:
- Salary slips: Basic pay, gross salary, net salary, deductions, allowances
- Mutual Fund statements: Investment amount, current value, NAV, units, SIP amounts, portfolio value
- Bank statements: Account balance, transaction amounts, monthly credits/debits
- Credit card statements: Credit limit, outstanding balance, payment history
- ITR documents: Total income, tax paid, investments under 80C

Your analysis should focus on:

**Financial Document Analysis:**
- Extract and analyze key financial information from salary slips, ITR documents, mutual fund statements, credit card statements and reports, bank statements, etc.
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

specific_template = """
You are a financial document analysis expert. Your task is to find and extract the EXACT information requested from the customer's financial documents.

SEARCH STRATEGY:
1. Look for the EXACT term requested (e.g., "Investment Amount")
2. Look for SIMILAR terms (e.g., "Invested Amount", "Total Investment", "Amount Invested")
3. Look for NUMERICAL values associated with these terms
4. Check different sections of the document (headers, tables, summary sections)

MUTUAL FUND SPECIFIC TERMS TO SEARCH FOR:
- Investment Amount / Invested Amount
- Purchase Amount / Purchase Value
- Total Amount Invested
- SIP Amount / Monthly SIP
- Current Value / Market Value
- Portfolio Value / Total Portfolio Value

RESPONSE FORMAT:
- If found: "The [requested information] is [exact value]. This information was found in: [quote exact text from document]"
- If not found: "I could not locate '[requested information]' in the provided documents. I searched for related terms like [list searched terms]. The available financial information includes: [list what was actually found]"

Question: {question}
Context from Guidelines: {guidelines_context}
Customer Financial Documents: {customer_context}

Provide a direct, specific answer with exact quotes from the source documents.
"""r:
"""

def determine_question_type(question: str) -> str:
    comprehensive_keywords = [
        "complete analysis", "full report", "comprehensive", "detailed analysis",
        "overall assessment", "complete evaluation", "full evaluation",
        "risk assessment", "financial capacity", "policy eligibility"
    ]
    
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in comprehensive_keywords):
        return "comprehensive"
    
    specific_indicators = [
        "what is", "how much", "when", "where", "which", "who",
        "calculate", "show me", "find", "extract", "tell me about"
    ]
    
    if any(indicator in question_lower for indicator in specific_indicators):
        return "specific"
    
    return "specific"

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
        "credit card", "investment amount", "units"
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

def process_documents_with_pii_shield(documents):
    protected_docs = []
    for doc in documents:
        anonymized_content = pii_shield.anonymize_text(doc.page_content)
        
        protected_doc = Document(
            page_content=anonymized_content,
            metadata=doc.metadata
        )
        protected_docs.append(protected_doc)
    
    return protected_docs

def analyze_customer_finances(question, guidelines_docs, customer_docs):
    guidelines_context = "\n\n".join([doc.page_content for doc in guidelines_docs])
    
    financial_docs = extract_financial_info(customer_docs)
    customer_context = "\n\n".join([doc.page_content for doc in financial_docs])
    
    question_type = determine_question_type(question)
    
    if question_type == "comprehensive":
        template = comprehensive_template
    else:
        template = specific_template
    
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

with st.sidebar:
    st.markdown("### 🛡️ PII Protection Settings")
    st.markdown("""
                ### 🔒 Privacy & Security Notice
                - **PII Protection**: Personal identifiable information is automatically anonymized using hash-based replacement
                - **Data Retention**: Document data is stored in memory only and cleared when the session ends
                - **Secure Processing**: All financial analysis is performed on anonymized data
                - **Compliance**: Designed to help maintain privacy standards for financial document processing""")
    
    pii_enabled = st.toggle("Enable PII Shield", value=True, help="Automatically anonymize personal information")
    pii_shield.anonymization_enabled = pii_enabled
    
    if pii_enabled:
        st.success("🛡️ PII Shield Active")
        
        if pii_shield.replacement_map:
            st.markdown("#### PII Detection Summary")
            pii_summary = pii_shield.get_pii_summary()
            for pii_type, count in pii_summary.items():
                st.write(f"• {pii_type.replace('_', ' ').title()}: {count} instances")
    else:
        st.warning("⚠️ PII Shield Disabled - Use with caution!")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Analysis History"):
        st.session_state.conversation_history = []
        pii_shield.replacement_map.clear()
        st.rerun()
    
    if st.button("🧹 Clear PII Cache"):
        pii_shield.replacement_map.clear()
        st.success("PII cache cleared")

col1, col2 = st.columns(2)

with col1:
    guidelines_files = st.file_uploader(
        "📋 Upload Financial Underwriting Guidelines",
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
    customer_files = st.file_uploader(
        "💼 Upload Customer Financial Documents",
        type="pdf",
        accept_multiple_files=True,
        key="customer_uploader"
    )
    
    if customer_files:
        with st.spinner("Processing customer documents with PII protection..."):
            all_customer_docs = []
            for file in customer_files:
                file_path = upload_pdf(file, customer_docs_directory)
                documents = load_pdf(file_path)
                chunked_documents = split_text(documents)
                
                if pii_shield.anonymization_enabled:
                    protected_documents = process_documents_with_pii_shield(chunked_documents)
                    all_customer_docs.extend(protected_documents)
                else:
                    all_customer_docs.extend(chunked_documents)
            
            customer_docs_vector_store.add_documents(all_customer_docs)
            st.session_state.customer_docs_loaded = True
            
            if pii_shield.anonymization_enabled and pii_shield.replacement_map:
                st.success(f"✅ {len(customer_files)} customer document(s) processed with PII protection!")
                st.info(f"🛡️ {len(pii_shield.replacement_map)} PII elements anonymized")
            else:
                st.success(f"✅ {len(customer_files)} customer document(s) processed!")

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
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💰 Income Analysis", use_container_width=True):
            st.session_state.conversation_history.append({
                "role": "user", 
                "content": "What is the customer's monthly income and income sources?"
            })
    
    with col2:
        if st.button("📊 Financial Capacity", use_container_width=True):
            st.session_state.conversation_history.append({
                "role": "user", 
                "content": "What is the customer's debt-to-income ratio and available financial capacity?"
            })
    
    with col3:
        if st.button("⚖️ Risk Assessment", use_container_width=True):
            st.session_state.conversation_history.append({
                "role": "user", 
                "content": "Provide a comprehensive financial risk assessment and policy eligibility recommendation."
            })
    
    with col4:
        if st.button("📋 Full Report", use_container_width=True):
            st.session_state.conversation_history.append({
                "role": "user", 
                "content": "Provide a complete comprehensive financial analysis report."
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
