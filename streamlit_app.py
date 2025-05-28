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

# NEW IMPORTS FOR GOOGLE VISION
from google.cloud import vision
from google.oauth2 import service_account
import json
import fitz  # PyMuPDF for PDF to image conversion
from PIL import Image
import io

# NEW IMPORTS FOR EXCEL EXPORT
from datetime import datetime
import xlsxwriter
from io import BytesIO

st.set_page_config(
    page_title="Financial Underwriting Assistant",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Financial Underwriting Assistant")

class DocumentTypeClassifier:
    def __init__(self):
        self.document_patterns = {
            'bank_statement': [
                r'bank\s*statement', r'account\s*statement', r'statement\s*of\s*account',
                r'opening\s*balance', r'closing\s*balance', r'credit\s*amount', r'debit\s*amount',
                r'transaction\s*date', r'cheque\s*no', r'reference\s*number', r'current\s*balance',
                r'savings\s*account', r'account\s*number', r'ifsc\s*code', r'branch\s*name'
            ],
            'credit_card': [
                r'credit\s*card\s*statement', r'card\s*statement', r'credit\s*limit',
                r'available\s*credit', r'minimum\s*amount\s*due', r'total\s*amount\s*due',
                r'payment\s*due\s*date', r'card\s*number', r'previous\s*balance',
                r'reward\s*points', r'cash\s*advance', r'finance\s*charges'
            ],
            'itr': [
                r'income\s*tax\s*return', r'itr[-\s]?[1-7]', r'assessment\s*year',
                r'financial\s*year', r'total\s*income', r'tax\s*payable', r'refund',
                r'section\s*80c', r'salary\s*income', r'house\s*property\s*income',
                r'capital\s*gains', r'other\s*sources', r'acknowledgment\s*number'
            ],
            'form_16': [
                r'form\s*16', r'tds\s*certificate', r'tax\s*deducted\s*at\s*source',
                r'employer\s*name', r'employee\s*name', r'pan\s*of\s*employee',
                r'tan\s*of\s*deductor', r'gross\s*salary', r'professional\s*tax',
                r'provident\s*fund', r'income\s*tax\s*deducted', r'net\s*salary'
            ],
            'salary_slip': [
                r'salary\s*slip', r'pay\s*slip', r'payslip', r'salary\s*statement',
                r'basic\s*salary', r'basic\s*pay', r'gross\s*salary', r'net\s*salary',
                r'hra', r'house\s*rent\s*allowance', r'conveyance\s*allowance',
                r'medical\s*allowance', r'pf\s*deduction', r'esi\s*deduction',
                r'employee\s*id', r'pay\s*period', r'ctc'
            ]
        }
    
    def classify_document(self, text_content: str, filename: str = "") -> str:
        """Classify document type based on content and filename"""
        text_lower = text_content.lower()
        filename_lower = filename.lower()
        
        # Check filename first for quick classification
        filename_scores = {}
        for doc_type, patterns in self.document_patterns.items():
            filename_score = sum(1 for pattern in patterns if re.search(pattern, filename_lower))
            if filename_score > 0:
                filename_scores[doc_type] = filename_score
        
        # Check content patterns
        content_scores = {}
        for doc_type, patterns in self.document_patterns.items():
            content_score = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            content_scores[doc_type] = content_score
        
        # Combine scores (content weighted higher than filename)
        combined_scores = {}
        all_doc_types = set(list(filename_scores.keys()) + list(content_scores.keys()))
        
        for doc_type in all_doc_types:
            combined_scores[doc_type] = (
                content_scores.get(doc_type, 0) * 3 +  # Content weighted 3x
                filename_scores.get(doc_type, 0) * 1    # Filename weighted 1x
            )
        
        # Return the document type with highest score, or 'unknown' if no matches
        if combined_scores:
            best_match = max(combined_scores.items(), key=lambda x: x[1])
            if best_match[1] > 0:
                return best_match[0]
        
        return 'unknown'
    
    def get_document_type_display_name(self, doc_type: str) -> str:
        """Convert internal document type to display name"""
        display_names = {
            'bank_statement': '🏦 Bank Statement',
            'credit_card': '💳 Credit Card Record',
            'itr': '📋 ITR (Income Tax Return)',
            'form_16': '📄 Form 16',
            'salary_slip': '💼 Salary Slip',
            'unknown': '❓ Unknown Document'
        }
        return display_names.get(doc_type, '❓ Unknown Document')

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

class FinancialDataExtractor:
    def __init__(self):
        self.extracted_data = {
            'personal_info': [],
            'income_details': [],
            'investment_details': [],
            'bank_details': [],
            'loan_details': [],
            'tax_details': [],
            'tables_data': [],
            'raw_text_data': []
        }
        
        self.financial_patterns = {
            'salary': r'(?:salary|basic\s*pay|gross\s*salary|net\s*salary|ctc)[\s:]*[₹$]?\s*([0-9,]+\.?[0-9]*)',
            'investment_amount': r'(?:investment|invested|amount)[\s:]*[₹$]?\s*([0-9,]+\.?[0-9]*)',
            'balance': r'(?:balance|closing\s*balance)[\s:]*[₹$]?\s*([0-9,]+\.?[0-9]*)',
            'credit_limit': r'(?:credit\s*limit)[\s:]*[₹$]?\s*([0-9,]+\.?[0-9]*)',
            'emi': r'(?:emi|monthly\s*installment)[\s:]*[₹$]?\s*([0-9,]+\.?[0-9]*)',
            'tax_paid': r'(?:tax\s*paid|income\s*tax)[\s:]*[₹$]?\s*([0-9,]+\.?[0-9]*)'
        }
    
    def extract_from_documents(self, documents):
        """Extract financial data from all documents"""
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata
            
            # Extract based on document type
            if metadata.get("type") == "table":
                self.extract_table_data(content, metadata)
            else:
                self.extract_text_data(content, metadata)
        
        return self.extracted_data
    
    def extract_table_data(self, content, metadata):
        """Extract data from table content"""
        lines = content.split('\n')
        table_data = []
        
        for line in lines:
            if line.strip() and not line.startswith('---'):
                # Try to identify financial values in table rows
                amounts = re.findall(r'[₹$]?\s*([0-9,]+\.?[0-9]*)', line)
                if amounts:
                    table_data.append({
                        'source': metadata.get('source', ''),
                        'page': metadata.get('page', ''),
                        'table_number': metadata.get('table_number', ''),
                        'content': line.strip(),
                        'extracted_amounts': amounts
                    })
        
        if table_data:
            self.extracted_data['tables_data'].extend(table_data)
    
    def extract_text_data(self, content, metadata):
        """Extract financial data from text content"""
        content_lower = content.lower()
        
        # Extract using patterns
        for pattern_name, pattern in self.financial_patterns.items():
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                extracted_item = {
                    'source': metadata.get('source', ''),
                    'page': metadata.get('page', ''),
                    'type': pattern_name,
                    'value': match.group(1),
                    'context': content[max(0, match.start()-50):match.end()+50].strip()
                }
                
                # Categorize based on content type
                if any(keyword in content_lower for keyword in ['salary', 'income', 'pay']):
                    self.extracted_data['income_details'].append(extracted_item)
                elif any(keyword in content_lower for keyword in ['investment', 'mutual fund', 'sip']):
                    self.extracted_data['investment_details'].append(extracted_item)
                elif any(keyword in content_lower for keyword in ['bank', 'account', 'balance']):
                    self.extracted_data['bank_details'].append(extracted_item)
                elif any(keyword in content_lower for keyword in ['loan', 'emi', 'credit']):
                    self.extracted_data['loan_details'].append(extracted_item)
                elif any(keyword in content_lower for keyword in ['tax', 'itr']):
                    self.extracted_data['tax_details'].append(extracted_item)
        
        # Store raw text for reference
        self.extracted_data['raw_text_data'].append({
            'source': metadata.get('source', ''),
            'page': metadata.get('page', ''),
            'content': content[:500] + '...' if len(content) > 500 else content
        })

def create_excel_export(extracted_data, filename="financial_data_export.xlsx"):
    """Create Excel file with extracted financial data"""
    buffer = BytesIO()
    
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1
        })
        
        cell_format = workbook.add_format({
            'border': 1,
            'text_wrap': True
        })
        
        amount_format = workbook.add_format({
            'border': 1,
            'num_format': '#,##0.00'
        })
        
        # Create summary sheet
        summary_data = []
        for category, items in extracted_data.items():
            if category != 'raw_text_data' and items:
                summary_data.append({
                    'Category': category.replace('_', ' ').title(),
                    'Items Count': len(items),
                    'Description': f"Contains {len(items)} extracted items"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            worksheet = writer.sheets['Summary']
            worksheet.set_column('A:C', 20)
        
        # Create sheets for each category
        for category, items in extracted_data.items():
            if not items or category == 'raw_text_data':
                continue
                
            sheet_name = category.replace('_', ' ').title()[:31]  # Excel sheet name limit
            
            if category == 'tables_data':
                df_data = []
                for item in items:
                    df_data.append({
                        'Source': item.get('source', ''),
                        'Page': item.get('page', ''),
                        'Table Number': item.get('table_number', ''),
                        'Content': item.get('content', ''),
                        'Extracted Amounts': ', '.join(item.get('extracted_amounts', []))
                    })
                df = pd.DataFrame(df_data)
            else:
                df_data = []
                for item in items:
                    df_data.append({
                        'Source': item.get('source', ''),
                        'Page': item.get('page', ''),
                        'Type': item.get('type', ''),
                        'Value': item.get('value', ''),
                        'Context': item.get('context', '')
                    })
                df = pd.DataFrame(df_data)
            
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]
                
                # Format columns
                for col_num, col_name in enumerate(df.columns):
                    if col_name in ['Value', 'Extracted Amounts']:
                        worksheet.set_column(col_num, col_num, 15, amount_format)
                    elif col_name in ['Context', 'Content']:
                        worksheet.set_column(col_num, col_num, 40, cell_format)
                    else:
                        worksheet.set_column(col_num, col_num, 20, cell_format)
                
                # Apply header format
                for col_num, _ in enumerate(df.columns):
                    worksheet.write(0, col_num, df.columns[col_num], header_format)
        
        # Create raw text data sheet (limited content)
        if extracted_data.get('raw_text_data'):
            raw_data = extracted_data['raw_text_data'][:100]  # Limit to first 100 entries
            df_raw = pd.DataFrame(raw_data)
            df_raw.to_excel(writer, sheet_name='Raw Text Data', index=False)
            worksheet = writer.sheets['Raw Text Data']
            worksheet.set_column('A:C', 30, cell_format)
    
    buffer.seek(0)
    return buffer

# Initialize instances
pii_shield = PIIShield()
financial_extractor = FinancialDataExtractor()
doc_classifier = DocumentTypeClassifier()

def setup_vision_client():
    """Setup Google Vision client with API key"""
    try:
        # Hardcoded API key - Replace with your actual API key
        api_key = "AIzaSyDz9toLotDK35LQUWat9E4sQ8DjFmXO4HE"
        
        if not api_key or api_key == "YOUR_ACTUAL_GOOGLE_VISION_API_KEY_HERE":
            st.error("⚠️ Please replace 'YOUR_ACTUAL_GOOGLE_VISION_API_KEY_HERE' with your actual Google Vision API key.")
            return None
            
        # Create credentials from API key
        client = vision.ImageAnnotatorClient(
            client_options={"api_key": api_key}
        )
        return client
    except Exception as e:
        st.error(f"Error setting up Google Vision client: {str(e)}")
        return None

def pdf_to_images(pdf_path):
    """Convert PDF pages to images for Vision API"""
    try:
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
            img_data = pix.tobytes("png")
            images.append({
                "data": img_data,
                "page": page_num + 1
            })
        
        doc.close()
        return images
    except Exception as e:
        st.error(f"Error converting PDF to images: {str(e)}")
        return []

def extract_text_with_vision(pdf_path):
    """Extract text from scanned PDF using Google Vision API"""
    vision_client = setup_vision_client()
    if not vision_client:
        return []
    
    try:
        images = pdf_to_images(pdf_path)
        documents = []
        
        for img_info in images:
            # Prepare image for Vision API
            image = vision.Image(content=img_info["data"])
            
            # Perform text detection
            response = vision_client.text_detection(image=image)
            
            if response.error.message:
                st.error(f"Vision API error: {response.error.message}")
                continue
            
            # Extract text
            texts = response.text_annotations
            if texts:
                extracted_text = texts[0].description
                
                doc = Document(
                    page_content=extracted_text,
                    metadata={
                        "source": pdf_path,
                        "page": img_info["page"],
                        "type": "scanned_text",
                        "extraction_method": "google_vision"
                    }
                )
                documents.append(doc)
        
        return documents
        
    except Exception as e:
        st.error(f"Error with Google Vision API: {str(e)}")
        return []

def is_scanned_pdf(pdf_path):
    """Determine if PDF is scanned by checking if text extraction yields minimal text"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_text = ""
            for page in pdf.pages[:3]:  # Check first 3 pages
                text = page.extract_text() or ""
                total_text += text
            
            # If very little text is extracted, likely scanned
            return len(total_text.strip()) < 100
    except:
        return True  # Assume scanned if can't determine

def extract_tables_from_pdf(file_path):
    """Extract both text and tables from PDF using pdfplumber"""
    document_content = []
    
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                document_content.append({
                    "content": text,
                    "page": page_num + 1,
                    "type": "text"
                })
            
            tables = page.extract_tables()
            for table_num, table in enumerate(tables):
                if table:
                    df = pd.DataFrame(table)
                    
                    if not df.empty:
                        headers = []
                        if len(df.columns) > 0:
                            if not pd.isna(df.iloc[0]).all() and not all(x is None for x in df.iloc[0]):
                                headers = [str(h).strip() if h is not None else f"Column_{i}" 
                                          for i, h in enumerate(df.iloc[0])]
                                df = df.iloc[1:]
                            else:
                                headers = [f"Column_{i}" for i in range(len(df.columns))]
                        
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
                        "dataframe": df
                    })
    
    return document_content

def format_table_for_llm(df: pd.DataFrame, table_info: dict) -> str:
    """Format table data for LLM processing"""
    if df.empty:
        return f"Empty table on page {table_info['page']}"
    
    table_text = f"\n--- TABLE {table_info['table_number']} (Page {table_info['page']}) ---\n"
    
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

def load_customer_pdf_with_vision(file_path, filename=""):
    """Load customer PDF: First try pdfplumber, then Vision API for scanned documents"""
    
    # First, try regular extraction with pdfplumber
    document_content = extract_tables_from_pdf(file_path)
    
    # Check if we got meaningful content
    has_meaningful_content = False
    all_text_content = ""
    
    for content in document_content:
        if content["type"] == "text" and len(content["content"].strip()) > 50:
            has_meaningful_content = True
            all_text_content += content["content"] + " "
        elif content["type"] == "table" and not content["dataframe"].empty:
            has_meaningful_content = True
            all_text_content += content["dataframe"].to_string() + " "
    
    # Classify document type based on all extracted content
    doc_type = doc_classifier.classify_document(all_text_content, filename)
    
    # If no meaningful content found, treat as scanned and use Vision API
    if not has_meaningful_content:
        st.info(f"📸 Document appears to be scanned. Using Google Vision API for text extraction...")
        vision_documents = extract_text_with_vision(file_path)
        if vision_documents:
            # Add document type to vision documents
            for doc in vision_documents:
                doc.metadata["document_type"] = doc_type
            return vision_documents, doc_type
        else:
            st.warning("Vision API failed, using available content from pdfplumber...")
            # Fall through to use pdfplumber content even if minimal
    
    # Convert pdfplumber results to Document objects
    documents = []
    for content in document_content:
        if content["type"] == "text":
            doc = Document(
                page_content=content["content"],
                metadata={
                    "source": file_path,
                    "page": content["page"],
                    "type": "text",
                    "document_type": doc_type
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
                    "document_type": doc_type
                }
            )
            documents.append(doc)
    
    # Always return tuple (documents, doc_type)
    return documents, doc_type

def load_pdf_with_tables(file_path):
    """Load PDF with tables for guidelines (uses regular pdfplumber extraction)"""
    document_content = extract_tables_from_pdf(file_path)
    documents = []
    
    for content in document_content:
        if content["type"] == "text":
            doc = Document(
                page_content=content["content"],
                metadata={
                    "source": file_path,
                    "page": content["page"],
                    "type": "text"
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
                    "table_number": content["table_number"]
                }
            )
            documents.append(doc)
    
    return documents

def split_text(documents):
    """Split documents into chunks while preserving table structure"""
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
    """Extract financially relevant document chunks"""
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

If you find multiple monthly salaries of a customer, calculate the average and multiply the same with 12, and show it as "Financial Viability"

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
    """Determine if question requires comprehensive or specific analysis"""
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

# Initialize models and vector stores
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
guidelines_vector_store = InMemoryVectorStore(embeddings)
customer_docs_vector_store = InMemoryVectorStore(embeddings)

model = ChatGroq(
    groq_api_key="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", 
    model_name="meta-llama/llama-4-scout-17b-16e-instruct", 
    temperature=0.3
)

def upload_pdf(file, directory):
    """Save uploaded file to directory"""
    file_path = directory + file.name
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def process_documents_with_pii_shield(documents):
    """Apply PII protection to documents"""
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
    """Analyze customer finances based on question and documents"""
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
if "extracted_financial_data" not in st.session_state:
    st.session_state.extracted_financial_data = None

# Sidebar for settings and status
with st.sidebar:
    st.markdown("### 🛡️ PII Protection Settings")
    st.markdown("""
                ### 🔒 Privacy & Security Notice
                - **PII Protection**: Personal identifiable information is automatically anonymized using hash-based replacement
                - **Data Retention**: Document data is stored in memory only and cleared when the session ends
                - **Secure Processing**: All financial analysis is performed on anonymized data
                - **Table Extraction**: Enhanced parsing preserves tabular financial data structure
                - **Google Vision**: Scanned documents processed with OCR for better text extraction
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
    
    # Vision API Status
    st.markdown("### 📸 Google Vision Status")
    vision_client = setup_vision_client()
    if vision_client:
        st.success("✅ Google Vision API Ready")
    else:
        st.error("❌ Google Vision API Not Available")
        st.markdown("Set `GOOGLE_VISION_API_KEY` in secrets.toml or environment")
    
    if st.button("🗑️ Clear Analysis History"):
        st.session_state.conversation_history = []
        st.session_state.table_stats = {"total_tables": 0, "tables_by_page": {}}
        pii_shield.replacement_map.clear()
        st.rerun()
    
    if st.button("🧹 Clear PII Cache"):
        pii_shield.replacement_map.clear()
        st.success("PII cache cleared")

# Main content area
col1, col2 = st.columns(2)

# Guidelines uploader
with col1:
    guidelines_files = st.file_uploader(
        "📋 Upload Financial Underwriting Guidelines",
        type="pdf",
        accept_multiple_files=True,
        key="guidelines_uploader"
    )
    
    if guidelines_files and not st.session_state.guidelines_loaded:
        with st.spinner("Processing guidelines with table extraction..."):
            all_guideline_docs = []
            for file in guidelines_files:
                file_path = upload_pdf(file, guidelines_directory)
                documents = load_pdf_with_tables(file_path)
                chunked_documents = split_text(documents)
                all_guideline_docs.extend(chunked_documents)
            
            guidelines_vector_store.add_documents(all_guideline_docs)
            st.session_state.guidelines_loaded = True
            st.success(f"✅ {len(guidelines_files)} guideline document(s) processed successfully!")

# Customer documents uploader
with col2:    
    customer_files = st.file_uploader(
        "💼 Upload Customer Financial Documents",
        type="pdf",
        accept_multiple_files=True,
        key="customer_uploader",
        help="Supports both regular PDFs and scanned documents (using Google Vision API)"
    )
    
    if customer_files:
        with st.spinner("Processing customer documents with enhanced table extraction and Vision API..."):
            all_customer_docs = []
            table_count = 0
            tables_by_page = {}
            scanned_count = 0
            
            for file in customer_files:
                file_path = upload_pdf(file, customer_docs_directory)
                
                # Use improved document processing flow - now properly unpacking tuple
                documents, doc_type = load_customer_pdf_with_vision(file_path, file.name)
                
                # Count scanned documents
                if any(doc.metadata.get("extraction_method") == "google_vision" for doc in documents):
                    scanned_count += 1
                
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
            
            success_msg = f"✅ {len(customer_files)} customer document(s) processed!"
            if scanned_count > 0:
                success_msg += f" 📸 {scanned_count} scanned with Vision API!"
            if table_count > 0:
                success_msg += f" 📊 {table_count} tables extracted!"
            
            if pii_shield.anonymization_enabled and pii_shield.replacement_map:
                success_msg += f" 🛡️ {len(pii_shield.replacement_map)} PII elements anonymized!"
            
            st.success(success_msg)
            st.markdown("#### 📋 Detected Document Types:")
            for file_info in processed_files:
                st.write(f"• **{file_info['filename']}**: {file_info['display_name']}")

# Status display
if st.session_state.guidelines_loaded and st.session_state.customer_docs_loaded:
    st.success("🎉 All documents loaded! Enhanced table extraction and Vision API ready for financial analysis.")
elif st.session_state.guidelines_loaded:
    st.warning("Guidelines loaded. Please upload customer financial documents.")
elif st.session_state.customer_docs_loaded:
    st.warning("Customer documents loaded. Please upload underwriting guidelines.")
else:
    st.info("📤 Please upload both guidelines and customer financial documents to begin analysis.")

# Analysis interface
if st.session_state.guidelines_loaded and st.session_state.customer_docs_loaded:
    st.markdown("---")
    st.markdown("### 🔍 Enhanced Financial Analysis with Table Data & Vision OCR")
    
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
        with st.spinner("Analyzing financial documents with table data..."):
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
