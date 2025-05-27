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

# Rest of your existing code (split_text, extract_financial_info, templates, etc.) remains the same...
# [Include all the remaining functions from your original code]

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
        st.markdown("""
        **To enable OCR:**
        1. Create a Google Cloud Project
        2. Enable Vision API
        3. Create service account or API key
        4. Upload credentials above
        """)
    
    st.markdown("---")
    st.markdown("### 🛡️ PII Protection Settings")
    # [Rest of your PII settings code...]

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

# [Include the rest of your chat interface and analysis code...]
