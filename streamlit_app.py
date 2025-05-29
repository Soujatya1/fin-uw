
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

from google.cloud import vision
from google.oauth2 import service_account
import json
import fitz
from PIL import Image
import io

from datetime import datetime
import xlsxwriter
from io import BytesIO

st.set_page_config(
    page_title="Financial Underwriting Assistant",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Financial Underwriting Assistant")

def get_income_multiplier(age: int, policy_type: str) -> int:
    """Get income multiplier based on age and policy type"""
    if policy_type.lower() == "term":
        if 18 <= age <= 30:
            return 25
        elif 31 <= age <= 35:
            return 25
        elif 36 <= age <= 40:
            return 20
        elif 41 <= age <= 45:
            return 15
        elif 46 <= age <= 50:
            return 12
        elif 51 <= age <= 55:
            return 10
        elif age >= 56:
            return 5
    else:
        if 18 <= age <= 30:
            return 35
        elif 31 <= age <= 35:
            return 30
        elif 36 <= age <= 40:
            return 25
        elif 41 <= age <= 45:
            return 20
        elif 46 <= age <= 50:
            return 15
        elif 51 <= age <= 65:
            return 10
        elif age > 65:
            return 6
    
    return 10

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
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata
            
            if metadata.get("type") == "table":
                self.extract_table_data(content, metadata)
            else:
                self.extract_text_data(content, metadata)
        
        return self.extracted_data
    
    def extract_table_data(self, content, metadata):
        lines = content.split('\n')
        table_data = []
        
        for line in lines:
            if line.strip() and not line.startswith('---'):
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
        content_lower = content.lower()
        
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
        
        self.extracted_data['raw_text_data'].append({
            'source': metadata.get('source', ''),
            'page': metadata.get('page', ''),
            'content': content[:500] + '...' if len(content) > 500 else content
        })

def create_excel_export(extracted_data, filename="financial_data_export.xlsx"):
    buffer = BytesIO()
    
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        workbook = writer.book
        
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
        
        for category, items in extracted_data.items():
            if not items or category == 'raw_text_data':
                continue
                
            sheet_name = category.replace('_', ' ').title()[:31]
            
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
                
                for col_num, _ in enumerate(df.columns):
                    worksheet.write(0, col_num, df.columns[col_num], header_format)
        
        if extracted_data.get('raw_text_data'):
            raw_data = extracted_data['raw_text_data'][:100]
            df_raw = pd.DataFrame(raw_data)
            df_raw.to_excel(writer, sheet_name='Raw Text Data', index=False)
            worksheet = writer.sheets['Raw Text Data']
            worksheet.set_column('A:C', 30, cell_format)
    
    buffer.seek(0)
    return buffer

pii_shield = PIIShield()
financial_extractor = FinancialDataExtractor()

def setup_vision_client():
    try:
        api_key = "AIzaSyDz9toLotDK35LQUWat9E4sQ8DjFmXO4HE"
        
        if not api_key or api_key == "YOUR_ACTUAL_GOOGLE_VISION_API_KEY_HERE":
            st.error("⚠️ Please replace 'YOUR_ACTUAL_GOOGLE_VISION_API_KEY_HERE' with your actual Google Vision API key.")
            return None
            
        client = vision.ImageAnnotatorClient(
            client_options={"api_key": api_key}
        )
        return client
    except Exception as e:
        st.error(f"Error setting up Google Vision client: {str(e)}")
        return None

def pdf_to_images(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
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
    vision_client = setup_vision_client()
    if not vision_client:
        return []
    
    try:
        images = pdf_to_images(pdf_path)
        documents = []
        
        for img_info in images:
            image = vision.Image(content=img_info["data"])
            
            response = vision_client.text_detection(image=image)
            
            if response.error.message:
                st.error(f"Vision API error: {response.error.message}")
                continue
            
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
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_text = ""
            for page in pdf.pages[:3]:
                text = page.extract_text() or ""
                total_text += text
            
            return len(total_text.strip()) < 100
    except:
        return True

def is_page_scanned(pdf_path: str, page_num: int, page_text: str = None) -> bool:
    """
    Robust method to detect if a page is scanned using multiple indicators:
    1. Check if page is mostly covered by images
    2. Check text density and quality
    3. Check for embedded fonts
    """
    try:
        # Method 1: Check image coverage using PyMuPDF
        import fitz
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)  # Convert to 0-based index
        
        # Get page dimensions
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        
        # Get all images on the page
        image_list = page.get_images()
        total_image_area = 0
        
        for img_index, img in enumerate(image_list):
            # Get image rectangles
            img_rects = page.get_image_rects(img[0])
            for rect in img_rects:
                total_image_area += rect.width * rect.height
        
        doc.close()
        
        # If images cover more than 80% of the page, likely scanned
        image_coverage_ratio = total_image_area / page_area if page_area > 0 else 0
        
        # Method 2: Check text characteristics
        text_density = len(page_text.strip()) if page_text else 0
        has_meaningful_text = text_density > 100
        
        # Method 3: Check for searchable text vs image-based content
        words_per_area = text_density / page_area * 10000 if page_area > 0 else 0
        
        # Decision logic:
        # - High image coverage (>80%) = likely scanned
        # - Very low text density (<50 chars) = likely scanned  
        # - Very low words per area (<5) = likely scanned
        is_scanned = (
            image_coverage_ratio > 0.8 or 
            text_density < 50 or 
            words_per_area < 5
        )
        
        return is_scanned
        
    except Exception as e:
        # Fallback to simple text-based detection
        text_density = len(page_text.strip()) if page_text else 0
        return text_density < 50

def extract_tables_from_pdf(file_path):
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

def load_customer_pdf_with_vision(file_path):
    """Enhanced function to handle mixed PDFs (both scanned and digital pages)"""
    # Extract content using pdfplumber first
    document_content = extract_tables_from_pdf(file_path)
    
    # Analyze each page to determine if it's scanned or digital
    page_analysis = {}
    scanned_pages = set()
    digital_documents = []
    
    # First pass: Analyze all pages
    for content in document_content:
        page_num = content["page"]
        if page_num not in page_analysis:
            page_text = content["content"] if content["type"] == "text" else ""
            is_scanned = is_page_scanned(file_path, page_num, page_text)
            page_analysis[page_num] = is_scanned
            
            if is_scanned:
                scanned_pages.add(page_num)
    
    # Second pass: Process content based on page analysis
    for content in document_content:
        page_num = content["page"]
        
        if content["type"] == "text":
            if not page_analysis.get(page_num, True):  # Digital page
                doc = Document(
                    page_content=content["content"],
                    metadata={
                        "source": file_path,
                        "page": page_num,
                        "type": "text"
                    }
                )
                digital_documents.append(doc)
                
        elif content["type"] == "table":
            # Tables can be extracted from both digital and some scanned pages
            # Only include if from digital pages or if table extraction was successful
            if not page_analysis.get(page_num, True) or not content["dataframe"].empty:
                table_text = format_table_for_llm(content["dataframe"], content)
                doc = Document(
                    page_content=table_text,
                    metadata={
                        "source": file_path,
                        "page": page_num,
                        "type": "table",
                        "table_number": content["table_number"]
                    }
                )
                digital_documents.append(doc)
    
    # Process scanned pages with Google Vision API
    vision_documents = []
    if scanned_pages:
        st.info(f"📸 Processing {len(scanned_pages)} scanned page(s) with Google Vision API...")
        st.info(f"Scanned pages: {sorted(list(scanned_pages))}")
        vision_documents = extract_text_with_vision_selective(file_path, list(scanned_pages))
    
    # Combine all documents
    all_documents = digital_documents + vision_documents
    
    if not all_documents:
        st.warning("No content could be extracted from the document.")
        return []
    
    # Log detailed processing summary
    digital_pages = [p for p, is_scanned in page_analysis.items() if not is_scanned]
    
    st.success(f"✅ Page Analysis Complete:")
    st.info(f"📄 Digital pages: {sorted(digital_pages)} ({len(digital_pages)} pages)")
    st.info(f"📸 Scanned pages: {sorted(list(scanned_pages))} ({len(scanned_pages)} pages)")
    st.info(f"📊 Total elements extracted: {len(all_documents)}")
    
    return all_documents

def extract_text_with_vision_selective(pdf_path, target_pages):
    """Extract text using Vision API only for specified pages"""
    vision_client = setup_vision_client()
    if not vision_client:
        return []
    
    try:
        images = pdf_to_images(pdf_path)
        documents = []
        
        for img_info in images:
            # Only process pages that were identified as scanned
            if img_info["page"] in target_pages:
                image = vision.Image(content=img_info["data"])
                
                response = vision_client.text_detection(image=image)
                
                if response.error.message:
                    st.error(f"Vision API error on page {img_info['page']}: {response.error.message}")
                    continue
                
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

def load_pdf_with_tables(file_path):
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
You are an expert Financial Underwriting AI Assistant specialized in insurance policy underwriting. Your primary task is to analyze customer financial documents and calculate financial viability using the EXACT methods specified in the underwriting guidelines for each document type.

**CRITICAL WORKFLOW - FOLLOW THESE STEPS IN ORDER:**

**STEP 1: DOCUMENT TYPE IDENTIFICATION**
Analyze the customer documents and identify the PRIMARY document type from these categories:
- Salary Slips
- Bank Statement (Salaried)
- Bank Statement (Closing Balance)
- ITR & COI (Income Tax Return & Certificate of Income)
- Form 16
- Mutual Fund Statement - SIP
- Credit Card Statements
- Car Ownership Documents
- Fixed Deposits
- Home Loan
- House/Shop Ownership

**Output Format:**
```
PRIMARY DOCUMENT TYPE IDENTIFIED: [Document Type]
CONFIDENCE LEVEL: High/Medium/Low
SUPPORTING EVIDENCE: [Key identifiers found in the document]
```

**STEP 2: GUIDELINE LOOKUP AND FORMULA EXTRACTION**
Based on the identified document type, customer age ({customer_age}), and policy type ({policy_type}), extract the EXACT calculation method from the guidelines.

**From the Guidelines Document, use these SPECIFIC formulas:**

**For Salary Slips:**
- Term Cases: Annual Salary = Gross Monthly Salary × 12; Financial Viability = Annual Salary × Income Multiplier
- Non-Term Cases: Annual Salary = Gross Monthly Salary × 12; Annual Bonus = Annual Salary × 0.10; Total = Annual Salary + Annual Bonus; Financial Viability = Total × Income Multiplier

**For Bank Statement (Salaried):**
- Term Cases: Average Monthly Salary (last 3 months) × 12; Add 30% to get Gross Annual Salary; Financial Viability = Gross Annual Salary × Age-based Multiplier
- Non-Term Cases: Average Monthly Salary (last 6 months) × 12; Financial Viability = Annual Income × Age-based Multiplier

**For Bank Statement (Closing Balance):**
- Term Cases: Average Closing Balance (last 3 months) × 12; Financial Viability = Annual Average Income × Age-based Multiplier

**For ITR & COI:**
- Term Cases: Only Earned Income (exclude unearned); Financial Viability = Total Earned Income × Age-based Multiplier
- Non-Term Cases: Include ALL income types (earned + unearned); Financial Viability = Total Income × Age-based Multiplier

**For Form 16:**
- Term Cases: Gross Income from Part A; Financial Viability = Annual Income × Age-based Multiplier

**For Mutual Fund Statement - SIP:**
- Term Cases: Monthly SIP × 12 = Annual Income; Financial Viability = Annual Income × Age-based Multiplier

**For Credit Card Statements:**
- Term Cases: Monthly CC Statement Value × 6 = Annual Income; Financial Viability = Annual Income × Age-based Multiplier

**For Car Ownership:**
- Term Cases: Car IDV Value × 2 = Annual Income; Financial Viability = Annual Income × Age-based Multiplier

**For Fixed Deposits:**
- Term Cases: Investment Value × 0.05 = Estimated Annual Income; Financial Viability = Annual Income × Age-based Multiplier

**For Home Loan:**
- Term Cases: Monthly EMI × 24 = Annual Income; Financial Viability = Annual Income × Age-based Multiplier

**For House/Shop Ownership:**
- Term Cases: Financial Viability = Property Value × 0.50

**STEP 3: AGE-BASED MULTIPLIER SELECTION**
Use the correct age-based multiplier from guidelines:

**Term Cases Multipliers:**
- Age 18-30: 25x
- Age 31-35: 25x  
- Age 36-40: 20x
- Age 41-45: 15x
- Age 46-50: 12x
- Age 51-55: 10x
- Age ≥56: 5x

**Non-Term Cases Multipliers:**
- Age 18-30: 35x
- Age 31-35: 30x
- Age 36-40: 25x
- Age 41-45: 20x
- Age 46-50: 15x
- Age 51-65: 10x
- Age >65: 6x

**STEP 4: PRECISE CALCULATION**
Apply the exact formula for the identified document type using actual values from customer documents.

**Output Format:**
```
GUIDELINE-BASED CALCULATION:
Document Type: [Type]
Policy Type: {policy_type}
Customer Age: {customer_age}
Applicable Formula: [Exact formula from guidelines]
Age-based Multiplier: [X]x

INPUT VALUES:
[List all extracted values with source references]

CALCULATION STEPS:
Step 1: [First calculation with actual numbers]
Step 2: [Second calculation if applicable]
Step 3: [Final calculation]

FINANCIAL VIABILITY: ₹[Final Amount]
```

**STEP 5: VALIDATION AND COMPARISON**
Compare your calculation with the generic method to ensure accuracy.

**STEP 6: COMPREHENSIVE ANALYSIS**
Provide detailed analysis including:

**Document Analysis Summary:**
| Parameter | Value | Source Location | Document Type Method | Age-Generic Method | Variance |
|-----------|-------|----------------|---------------------|-------------------|----------|

**Risk Assessment:**
- Income Stability: [Analysis]
- Debt-to-Income Ratio: [If applicable]
- Financial Capacity: [Assessment]
- Premium Affordability: [Based on calculated viability]

**Final Recommendation:**
- Recommended Coverage: ₹[Amount based on guideline calculation]
- Policy Eligibility: Approved/Conditional/Declined
- Justification: [Why this specific calculation method was used]

**CRITICAL INSTRUCTIONS:**
1. NEVER use the generic age-based multiplier ({income_multiplier}x) alone - it's only for reference
2. ALWAYS use the document-type-specific calculation method from guidelines
3. Extract EXACT numerical values from customer documents
4. Show ALL calculation steps with actual numbers
5. Reference specific guideline sections/pages
6. If document type is unclear, state this and explain your reasoning
7. If multiple document types are present, prioritize the most reliable one (usually Salary Slip or ITR)

**Customer Information:**
- Age: {customer_age} years
- Policy Type: {policy_type}
- Reference Generic Multiplier: {income_multiplier}x (DO NOT USE - for reference only)

**Question:** {question}
**Guidelines Context:** {guidelines_context}
**Customer Financial Documents:** {customer_context}

**RESPONSE:**
"""

specific_template = """
You are a financial underwriting expert. Answer the specific question asked based on the customer's financial documents and underwriting guidelines. 
IMPORTANT: Mention the customer financial document type.

**Customer Information:**
- Age: {customer_age} years
- Policy Type: {policy_type}
- Income Multiplier: {income_multiplier}x

FINANCIAL VIABILITY CALCULATION:
Formula: Monthly Income × 12 × {income_multiplier} = Financial Viability
(This multiplier is age and policy-type specific)

IMPORTANT: The documents contain both TEXT and TABLE data.

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

guidelines_directory = '.github/guidelines/'
customer_docs_directory = '.github/customer_docs/'

os.makedirs(guidelines_directory, exist_ok=True)
os.makedirs(customer_docs_directory, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
guidelines_vector_store = InMemoryVectorStore(embeddings)
customer_docs_vector_store = InMemoryVectorStore(embeddings)

model = ChatGroq(
    groq_api_key="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", 
    model_name="meta-llama/llama-4-scout-17b-16e-instruct", 
    temperature=0.3
)

def upload_pdf(file, directory):
    file_path = directory + file.name
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

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
    
    customer_age = st.session_state.customer_age or 30
    policy_type = st.session_state.policy_type or "Term"
    income_multiplier = get_income_multiplier(customer_age, policy_type)
    
    if question_type == "comprehensive":
        template = comprehensive_template
    else:
        template = specific_template
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    response = chain.invoke({
        "question": question,
        "guidelines_context": guidelines_context,
        "customer_context": customer_context,
        "customer_age": customer_age,
        "policy_type": policy_type,
        "income_multiplier": income_multiplier
    })
    
    return response.content

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
if "customer_age" not in st.session_state:
    st.session_state.customer_age = None
if "policy_type" not in st.session_state:
    st.session_state.policy_type = "Term"

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

col1, col2 = st.columns(2)

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
                
                documents = load_customer_pdf_with_vision(file_path)
                
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

if st.session_state.guidelines_loaded and st.session_state.customer_docs_loaded:
    st.success("🎉 All documents loaded! Enhanced table extraction and Vision API ready for financial analysis.")
elif st.session_state.guidelines_loaded:
    st.warning("Guidelines loaded. Please upload customer financial documents.")
elif st.session_state.customer_docs_loaded:
    st.warning("Customer documents loaded. Please upload underwriting guidelines.")
else:
    st.info("📤 Please upload both guidelines and customer financial documents to begin analysis.")

if st.session_state.guidelines_loaded and st.session_state.customer_docs_loaded:
    st.markdown("---")
    st.markdown("### 👤 Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        customer_age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=80,
            value=st.session_state.customer_age or 30,
            help="Required for calculating income multiplier"
        )
        st.session_state.customer_age = customer_age
    
    with col2:
        policy_type = st.selectbox(
            "Policy Type",
            ["Term", "Non-Term"],
            index=0 if st.session_state.policy_type == "Term" else 1,
            help="Policy type affects income multiplier calculation"
        )
        st.session_state.policy_type = policy_type
    
    if customer_age:
        multiplier = get_income_multiplier(customer_age, policy_type)
        st.info(f"📊 Current Income Multiplier: **{multiplier}x** (Age: {customer_age}, Policy: {policy_type})")

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
        if st.session_state.customer_age is None:
            st.warning("⚠️ Please enter customer age before proceeding with analysis.")
            st.stop()
    
        with st.spinner("Analyzing financial documents with table data..."):
            latest_question = st.session_state.conversation_history[-1]["content"]
            guidelines_docs = guidelines_vector_store.similarity_search(latest_question, k=5)
            customer_docs = customer_docs_vector_store.similarity_search(latest_question, k=15)  # Increased k to capture more table data
        
            answer = analyze_customer_finances(latest_question, guidelines_docs, customer_docs)
        
            st.session_state.conversation_history.append({"role": "assistant", "content": answer})
    
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])
