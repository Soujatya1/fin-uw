import streamlit as st
import os
import re
import hashlib
from typing import List, Dict, Tuple
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
from langchain_core.documents import Document

# New imports for Google Vision API
from google.cloud import vision
import io
from PIL import Image
import pdf2image
import base64

st.set_page_config(
    page_title="Financial Underwriting Assistant",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Financial Underwriting Assistant with OCR")

class VisionOCRProcessor:
    def __init__(self, api_key: str = None):
        """
        Initialize Google Vision API client
        Args:
            api_key: Google Cloud Vision API key (optional if using service account)
        """
        self.api_key = api_key
        
        if api_key:
            # Set the API key as environment variable
            os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'] = api_key
            
        try:
            self.client = vision.ImageAnnotatorClient()
            self.ocr_available = True
        except Exception as e:
            st.warning(f"Google Vision API not available: {str(e)}")
            self.ocr_available = False
    
    def extract_text_from_image(self, image_bytes: bytes) -> str:
        """Extract text from image using Google Vision API"""
        if not self.ocr_available:
            return ""
        
        try:
            image = vision.Image(content=image_bytes)
            response = self.client.text_detection(image=image)
            
            if response.error.message:
                raise Exception(f"Vision API error: {response.error.message}")
            
            texts = response.text_annotations
            
            if texts:
                return texts[0].description
            return ""
            
        except Exception as e:
            st.error(f"OCR extraction failed: {str(e)}")
            return ""
    
    def detect_tables_in_image(self, image_bytes: bytes) -> List[Dict]:
        """Detect and extract table structure from image"""
        if not self.ocr_available:
            return []
        
        try:
            image = vision.Image(content=image_bytes)
            
            # Use document text detection for better table structure
            response = self.client.document_text_detection(image=image)
            
            if response.error.message:
                raise Exception(f"Vision API error: {response.error.message}")
            
            document = response.full_text_annotation
            tables = []
            
            # Extract text blocks that might represent tables
            if document:
                # Simple table detection based on text layout
                text_blocks = []
                for page in document.pages:
                    for block in page.blocks:
                        block_text = ""
                        for paragraph in block.paragraphs:
                            for word in paragraph.words:
                                word_text = ''.join([symbol.text for symbol in word.symbols])
                                block_text += word_text + " "
                        
                        if block_text.strip():
                            # Get bounding box
                            vertices = block.bounding_box.vertices
                            bbox = {
                                'x1': min([v.x for v in vertices]),
                                'y1': min([v.y for v in vertices]),
                                'x2': max([v.x for v in vertices]),
                                'y2': max([v.y for v in vertices])
                            }
                            
                            text_blocks.append({
                                'text': block_text.strip(),
                                'bbox': bbox
                            })
                
                # Identify potential table structures
                tables = self._identify_table_structures(text_blocks)
            
            return tables
            
        except Exception as e:
            st.error(f"Table detection failed: {str(e)}")
            return []
    
    def _identify_table_structures(self, text_blocks: List[Dict]) -> List[Dict]:
        """Identify table structures from text blocks"""
        tables = []
        
        # Simple heuristic: look for blocks with numbers and consistent spacing
        for block in text_blocks:
            text = block['text']
            
            # Check if block contains tabular data patterns
            lines = text.split('\n')
            numeric_lines = sum(1 for line in lines if re.search(r'\d+', line))
            
            if numeric_lines >= 2:  # Potential table if multiple lines contain numbers
                # Try to parse as table
                table_data = []
                for line in lines:
                    if line.strip():
                        # Split by multiple spaces or tabs
                        cells = re.split(r'\s{2,}|\t', line.strip())
                        if len(cells) > 1:
                            table_data.append(cells)
                
                if len(table_data) >= 2:  # At least header + 1 row
                    tables.append({
                        'data': table_data,
                        'bbox': block['bbox'],
                        'type': 'detected_table'
                    })
        
        return tables

class PIIShield:
    def __init__(self):
        self.pii_patterns = {
            'pan_card': r'\b[A-Z]{5}[-\s]?[0-9]{4}[-\s]?[A-Z]\b',
            'tan': r'\b[A-Z]{4}[-\s]?[0-9]{5}[-\s]?[A-Z]\b',
            'aadhaar': r'\b\d{4}\s?\d{4}\s?\d{4}\b',
            'account_number': r'\b\d{9,18}\b',
            'phone': r'\b(?:\+91[-.\s]?)?[6-9]\d{9}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'address': r'\b(?:house|flat|plot|door)\s*(?:no\.?|number)?\s*[0-9A-Za-z\-\/]+\b',
            'ifsc': r'\b[A-Z]{4}0[A-Z0-9]{6}\b'
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

# Initialize OCR processor - will be set up in sidebar
vision_processor = None
pii_shield = PIIShield()

def is_scanned_pdf(file_path: str) -> bool:
    """Check if PDF contains scanned images rather than text"""
    try:
        with pdfplumber.open(file_path) as pdf:
            total_text_length = 0
            for page in pdf.pages:
                text = page.extract_text() or ""
                total_text_length += len(text.strip())
            
            # If very little text is extractable, likely scanned
            return total_text_length < 100
    except:
        return True

def convert_pdf_to_images(file_path: str) -> List[Image.Image]:
    """Convert PDF pages to images"""
    try:
        images = pdf2image.convert_from_path(file_path, dpi=300)
        return images
    except Exception as e:
        st.error(f"Failed to convert PDF to images: {str(e)}")
        return []

def extract_tables_from_pdf_with_ocr(file_path: str):
    """Enhanced PDF extraction with OCR fallback for scanned documents"""
    document_content = []
    
    # First try regular PDF text extraction
    try:
        with pdfplumber.open(file_path) as pdf:
            text_extracted = False
            
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                
                if text.strip() and len(text.strip()) > 50:  # Sufficient text found
                    text_extracted = True
                    document_content.append({
                        "content": text,
                        "page": page_num + 1,
                        "type": "text",
                        "source": "pdfplumber"
                    })
                    
                    # Extract tables normally
                    tables = page.extract_tables()
                    for table_num, table in enumerate(tables):
                        if table:
                            df = pd.DataFrame(table)
                            
                            if not df.empty:
                                # Process headers
                                headers = []
                                if len(df.columns) > 0:
                                    if not pd.isna(df.iloc[0]).all() and not all(x is None for x in df.iloc[0]):
                                        headers = [str(h).strip() if h is not None else f"Column_{i}" 
                                                  for i, h in enumerate(df.iloc[0])]
                                        df = df.iloc[1:]
                                    else:
                                        headers = [f"Column_{i}" for i in range(len(df.columns))]
                                
                                # Handle duplicate headers
                                unique_headers = []
                                header_counts = {}
                                
                                for h in headers:
                                    if h in header_counts:
                                        header_counts[h] += 1
                                        unique_headers.append(f"{h}_{header_counts[h]}")
                                    else:
                                        header_counts[h] = 0
                                        unique_headers.append(h)
                                
                                df.columns = unique_headers
                            
                            document_content.append({
                                "page": page_num + 1,
                                "type": "table",
                                "table_number": table_num + 1,
                                "dataframe": df,
                                "source": "pdfplumber"
                            })
            
            # If insufficient text was extracted, use OCR
            if not text_extracted and vision_processor and vision_processor.ocr_available:
                st.info("🔍 Scanned document detected. Using OCR...")
                document_content = extract_with_ocr(file_path)
    
    except Exception as e:
        st.error(f"PDF extraction failed: {str(e)}")
        # Fallback to OCR if available
        if vision_processor and vision_processor.ocr_available:
            st.info("📄 Falling back to OCR extraction...")
            document_content = extract_with_ocr(file_path)
    
    return document_content

def extract_with_ocr(file_path: str) -> List[Dict]:
    """Extract content using OCR for scanned documents"""
    document_content = []
    
    if not vision_processor or not vision_processor.ocr_available:
        st.error("OCR not available. Please configure Google Vision API.")
        return document_content
    
    # Convert PDF to images
    images = convert_pdf_to_images(file_path)
    
    if not images:
        return document_content
    
    for page_num, image in enumerate(images):
        try:
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # Extract text using OCR
            ocr_text = vision_processor.extract_text_from_image(img_bytes)
            
            if ocr_text.strip():
                document_content.append({
                    "content": ocr_text,
                    "page": page_num + 1,
                    "type": "text",
                    "source": "ocr"
                })
            
            # Try to detect tables in the image
            detected_tables = vision_processor.detect_tables_in_image(img_bytes)
            
            for table_num, table_info in enumerate(detected_tables):
                if table_info['data']:
                    df = pd.DataFrame(table_info['data'])
                    
                    # Set first row as headers if it looks like headers
                    if len(df) > 1:
                        first_row = df.iloc[0]
                        if not any(str(cell).replace('.', '').replace(',', '').isdigit() for cell in first_row if pd.notna(cell)):
                            df.columns = [str(col) if pd.notna(col) else f"Column_{i}" for i, col in enumerate(first_row)]
                            df = df.iloc[1:].reset_index(drop=True)
                    
                    document_content.append({
                        "page": page_num + 1,
                        "type": "table",
                        "table_number": table_num + 1,
                        "dataframe": df,
                        "source": "ocr_table_detection"
                    })
        
        except Exception as e:
            st.error(f"OCR processing failed for page {page_num + 1}: {str(e)}")
            continue
    
    return document_content

def format_table_for_llm(df: pd.DataFrame, table_info: dict) -> str:
    if df.empty:
        return f"Empty table on page {table_info['page']} (Source: {table_info.get('source', 'unknown')})"
    
    source_info = f" - Source: {table_info.get('source', 'unknown')}"
    table_text = f"\n--- TABLE {table_info['table_number']} (Page {table_info['page']}{source_info}) ---\n"
    
    table_text += df.to_string(index=False, na_rep='') + "\n"
    
    table_text += "\nKey Financial Data from this table:\n"
    for col in df.columns:
        non_null_values = df[col].dropna()
        if not non_null_values.empty:
            numeric_values = []
            for val in non_null_values:
                if isinstance(val, (int, float)) or (isinstance(val, str) and any(char.isdigit() for char in str(val))):
                    numeric_values.append(str(val))
            
            if numeric_values:
                table_text += f"- {col}: {', '.join(numeric_values[:5])}\n"
    
    table_text += "--- END TABLE ---\n"
    return table_text

def load_pdf_with_tables_and_ocr(file_path):
    """Load PDF with enhanced OCR capabilities"""
    document_content = extract_tables_from_pdf_with_ocr(file_path)
    documents = []
    
    for content in document_content:
        if content["type"] == "text":
            doc = Document(
                page_content=content["content"],
                metadata={
                    "source": file_path,
                    "page": content["page"],
                    "type": "text",
                    "extraction_method": content.get("source", "unknown")
                }
            )
            documents.append(doc)
            
        elif content["type"] == "table":
            table_text = format_table_for_llm(content["dataframe"], content)
            doc = Document(
                page_content=table_text,
                metadata={
                    "source": file_path,
                    "page": content["page"],
                    "type": "table",
                    "table_number": content["table_number"],
                    "extraction_method": content.get("source", "unknown")
                }
            )
            documents.append(doc)
    
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        add_start_index=True
    )
    
    chunked_docs = []
    for doc in documents:
        if doc.metadata.get("type") == "table":
            chunked_docs.append(doc)
        else:
            chunks = text_splitter.split_documents([doc])
            chunked_docs.extend(chunks)
    
    return chunked_docs

def extract_financial_info(documents):
    financial_keywords = [
        "salary", "income", "annual income", "monthly income", 
        "basic pay", "gross salary", "net salary", "CTC",
        "ITR", "income tax return", "form 16", "tax",
        "mutual fund", "SIP", "investment", "portfolio",
        "credit card", "investment amount", "units", "NAV",
        "bank statement", "account balance", "savings",
        "loan", "EMI", "debt", "liability", "credit",
        "bonus", "incentive", "allowance", "deduction",
        "amount", "balance", "value", "total", "sum"
    ]
    
    relevant_chunks = []
    for doc in documents:
        content_lower = doc.page_content.lower()
        
        if doc.metadata.get("type") == "table":
            relevant_chunks.append(doc)
        elif any(keyword in content_lower for keyword in financial_keywords):
            relevant_chunks.append(doc)
    
    return relevant_chunks

comprehensive_template = """
Hello, AI Financial Underwriting Assistant. You are a specialized AI agent with expertise in financial underwriting for insurance products. Your role is to analyze customer financial documents and assess their financial viability for insurance policies based on the provided underwriting guidelines.

IMPORTANT: All customer data has been anonymized for privacy protection. Use anonymized identifiers in your analysis.

DOCUMENT FORMAT NOTICE: The customer documents contain both text content and structured TABLE data. Tables are clearly marked with "--- TABLE X (Page Y) ---" headers. Pay special attention to tabular data as it often contains key financial figures.

CRITICAL INSTRUCTIONS:
1. CAREFULLY READ through ALL the provided customer financial documents including both text and tables
2. Extract SPECIFIC numerical values, amounts, and financial data mentioned in the documents
3. When analyzing tables, look for columns with financial data like amounts, balances, dates, etc.
4. When asked about specific values like "Investment Amount", look for exact matches in both text and table format
5. If you cannot find specific information, clearly state what information is missing
6. Always quote the exact text/numbers from the documents when available
7. For tabular data, reference the table number and page for traceability

DOCUMENT ANALYSIS FOCUS:
- Salary slips: Basic pay, gross salary, net salary, deductions, allowances (often in tabular format)
- Mutual Fund statements: Investment amount, current value, NAV, units, SIP amounts, portfolio value (usually tabular)
- Bank statements: Account balance, transaction amounts, monthly credits/debits (tabular transaction data)
- Credit card statements: Credit limit, outstanding balance, payment history (tabular)
- ITR documents: Total income, tax paid, investments under 80C (may include tabular schedules)

Your analysis should focus on:

**Financial Document Analysis:**
- Extract and analyze key financial information from salary slips, ITR documents, mutual fund statements, credit card statements and reports, bank statements, etc.
- Pay special attention to tabular data which often contains precise financial figures
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
| Parameter | Customer Value | Source (Text/Table Page) | Risk Assessment | Comments |
|-----------|---------------|------------------------|-----------------|----------|

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
You are a financial underwriting expert. Answer the specific question asked based on the customer's financial documents and underwriting guidelines. 

IMPORTANT: The documents contain both TEXT and TABLE data. Tables are marked with "--- TABLE X (Page Y) ---" headers. Look for specific values in both formats.

CRITICAL: Please search carefully in both text content and tabular data. Financial documents often have key information in table format.

Be concise and specific. Only provide the information directly relevant to the question asked. If you find the information in a table, mention the table number and page.

Question: {question}
Customer Financial Documents: {customer_context}
Answer:
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

# Directory setup
guidelines_directory = '.github/guidelines/'
customer_docs_directory = '.github/customer_docs/'

os.makedirs(guidelines_directory, exist_ok=True)
os.makedirs(customer_docs_directory, exist_ok=True)

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
guidelines_vector_store = InMemoryVectorStore(embeddings)
customer_docs_vector_store = InMemoryVectorStore(embeddings)

model = ChatGroq(
    groq_api_key="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", 
    model_name="meta-llama/llama-4-scout-17b-16e-instruct", 
    temperature=0.3
)

def upload_pdf(file, directory):
    """Upload PDF file to specified directory"""
    file_path = directory + file.name
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def process_documents_with_pii_shield(documents):
    """Process documents through PII anonymization"""
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
    """Analyze customer finances using LLM"""
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

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "guidelines_loaded" not in st.session_state:
    st.session_state.guidelines_loaded = False
if "customer_docs_loaded" not in st.session_state:
    st.session_state.customer_docs_loaded = False
if "table_stats" not in st.session_state:
    st.session_state.table_stats = {"total_tables": 0, "tables_by_page": {}}

# Sidebar configuration with OCR setup
with st.sidebar:
    st.markdown("### 🔍 OCR Configuration")
    
    # Google Vision API setup
    st.markdown("#### Google Vision API Setup")
    api_key_method = st.radio(
        "API Key Method:",
        ["Service Account JSON", "API Key String"],
        help="Choose how to authenticate with Google Vision API"
    )
    
    if api_key_method == "Service Account JSON":
        uploaded_json = st.file_uploader(
            "Upload Service Account JSON",
            type="json",
            help="Upload your Google Cloud service account JSON file"
        )
        
        if uploaded_json:
            try:
                # Save the JSON file temporarily
                json_path = "temp_service_account.json"
                with open(json_path, "wb") as f:
                    f.write(uploaded_json.getbuffer())
                
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json_path
                vision_processor = VisionOCRProcessor()
                
                if vision_processor.ocr_available:
                    st.success("✅ Google Vision API configured successfully!")
                else:
                    st.error("❌ Failed to configure Google Vision API")
                    
            except Exception as e:
                st.error(f"Error setting up Vision API: {str(e)}")
    
    else:
        api_key = st.text_input(
            "Google Vision API Key:",
            type="password",
            help="Enter your Google Cloud Vision API key"
        )
        
        if api_key:
            try:
                vision_processor = VisionOCRProcessor(api_key)
                if vision_processor.ocr_available:
                    st.success("✅ Google Vision API configured!")
                else:
                    st.error("❌ Invalid API key or configuration error")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if not vision_processor or not vision_processor.ocr_available:
        st.warning("⚠️ OCR not available. Scanned documents won't be processed.")
        st.markdown("### 🛡️ PII Protection Settings")
    st.markdown("""
                ### 🔒 Privacy & Security Notice
                - **PII Protection**: Personal identifiable information is automatically anonymized using hash-based replacement
                - **Data Retention**: Document data is stored in memory only and cleared when the session ends
                - **Secure Processing**: All financial analysis is performed on anonymized data
                - **Table Extraction**: Enhanced parsing preserves tabular financial data structure
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
    
    if st.button("🗑️ Clear Analysis History"):
        st.session_state.conversation_history = []
        st.session_state.table_stats = {"total_tables": 0, "tables_by_page": {}}
        pii_shield.replacement_map.clear()
        st.rerun()
    
    if st.button("🧹 Clear PII Cache"):
        pii_shield.replacement_map.clear()
        st.success("PII cache cleared")

# Main document processing section
col1, col2 = st.columns(2)

with col1:
    guidelines_files = st.file_uploader(
        "📋 Upload Financial Underwriting Guidelines",
        type="pdf",
        accept_multiple_files=True,
        key="guidelines_uploader"
    )
    
    if guidelines_files and not st.session_state.get("guidelines_loaded", False):
        with st.spinner("Processing guidelines with OCR support..."):
            all_guideline_docs = []
            for file in guidelines_files:
                file_path = upload_pdf(file, guidelines_directory)
                
                # Use OCR-enhanced processing
                documents = load_pdf_with_tables_and_ocr(file_path)
                chunked_documents = split_text(documents)
                all_guideline_docs.extend(chunked_documents)
            
            guidelines_vector_store.add_documents(all_guideline_docs)
            st.session_state.guidelines_loaded = True
            st.success(f"✅ {len(guidelines_files)} guideline document(s) processed with OCR support!")

with col2:    
    customer_files = st.file_uploader(
        "💼 Upload Customer Financial Documents",
        type="pdf",
        accept_multiple_files=True,
        key="customer_uploader"
    )
    
    if customer_files:
        with st.spinner("Processing customer documents with OCR and table extraction..."):
            all_customer_docs = []
            table_count = 0
            tables_by_page = {}
            ocr_used_count = 0
            
            for file in customer_files:
                file_path = upload_pdf(file, customer_docs_directory)
                
                # Check if document is scanned
                if is_scanned_pdf(file_path):
                    st.info(f"📄 Detected scanned document: {file.name}")
                    ocr_used_count += 1
                
                # Use OCR-enhanced processing
                documents = load_pdf_with_tables_and_ocr(file_path)
                
                for doc in documents:
                    if doc.metadata.get("type") == "table":
                        table_count += 1
                        page = doc.metadata.get("page", 0)
                        tables_by_page[page] = tables_by_page.get(page, 0) + 1
                
                chunked_documents = split_text(documents)
                
                if pii_shield.anonymization_enabled:
                    protected_documents = process_documents_with_pii_shield(chunked_documents)
                    all_customer_docs.extend(protected_documents)
                else:
                    all_customer_docs.extend(chunked_documents)
            
            st.session_state.table_stats = {
                "total_tables": table_count,
                "tables_by_page": tables_by_page
            }
            
            customer_docs_vector_store.add_documents(all_customer_docs)
            st.session_state.customer_docs_loaded = True
            
            success_msg = f"✅ {len(customer_files)} document(s) processed!"
            if table_count > 0:
                success_msg += f" 📊 {table_count} tables extracted!"
            if ocr_used_count > 0:
                success_msg += f" 🔍 {ocr_used_count} scanned documents processed with OCR!"
            
            if pii_shield.anonymization_enabled and pii_shield.replacement_map:
                success_msg += f" 🛡️ {len(pii_shield.replacement_map)} PII elements anonymized!"
            
            st.success(success_msg)

# Status display
if st.session_state.guidelines_loaded and st.session_state.customer_docs_loaded:
    st.success("🎉 All documents loaded! Enhanced OCR and table extraction ready for financial analysis.")
elif st.session_state.guidelines_loaded:
    st.warning("Guidelines loaded. Please upload customer financial documents.")
elif st.session_state.customer_docs_loaded:
    st.warning("Customer documents loaded. Please upload underwriting guidelines.")
else:
    st.info("📤 Please upload both guidelines and customer financial documents to begin analysis.")

# Chat interface and analysis
if st.session_state.guidelines_loaded and st.session_state.customer_docs_loaded:
    st.markdown("---")
    st.markdown("### 🔍 Enhanced Financial Analysis with OCR & Table Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💰 Income Analysis", use_container_width=True):
            st.session_state.conversation_history.append({
                "role": "user", 
                "content": "What is the customer's monthly income and income sources? Look for both text and tabular data."
            })
    
    with col2:
        if st.button("📊 Investment Analysis", use_container_width=True):
            st.session_state.conversation_history.append({
                "role": "user", 
                "content": "Analyze the customer's investment portfolio from mutual fund statements and investment tables."
            })
    
    with col3:
        if st.button("⚖️ Risk Assessment", use_container_width=True):
            st.session_state.conversation_history.append({
                "role": "user", 
                "content": "Provide a comprehensive financial risk assessment using all available text and tabular data."
            })
    
    with col4:
        if st.button("📋 Full Report", use_container_width=True):
            st.session_state.conversation_history.append({
                "role": "user", 
                "content": "Provide a complete comprehensive financial analysis report using all text and table data."
            })
    
    question = st.chat_input("Please ask a question")
    
    if question:
        st.session_state.conversation_history.append({"role": "user", "content": question})
    
    if st.session_state.conversation_history and st.session_state.conversation_history[-1]["role"] == "user":
        with st.spinner("Analyzing financial documents with OCR and table data..."):
            latest_question = st.session_state.conversation_history[-1]["content"]
            guidelines_docs = guidelines_vector_store.similarity_search(latest_question, k=5)
            customer_docs = customer_docs_vector_store.similarity_search(latest_question, k=15)  # Increased k to capture more table data
            
            answer = analyze_customer_finances(latest_question, guidelines_docs, customer_docs)
            
            st.session_state.conversation_history.append({"role": "assistant", "content": answer})
    
    # Display conversation history
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])
