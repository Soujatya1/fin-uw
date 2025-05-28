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

def extract_tables_from_pdf(file_path):
    """Enhanced PDF extraction with better error handling and metrics"""
    document_content = []
    extraction_stats = {
        'total_pages': 0,
        'pages_with_text': 0,
        'pages_with_tables': 0,
        'total_text_length': 0,
        'total_tables': 0,
        'extraction_quality': 'unknown'
    }
    
    try:
        with pdfplumber.open(file_path) as pdf:
            extraction_stats['total_pages'] = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages):
                # Extract text with better handling
                text = page.extract_text() or ""
                
                if text.strip():
                    extraction_stats['pages_with_text'] += 1
                    extraction_stats['total_text_length'] += len(text.strip())
                    
                    document_content.append({
                        "content": text,
                        "page": page_num + 1,
                        "type": "text",
                        "char_count": len(text.strip())
                    })
                
                # Extract tables with enhanced processing
                tables = page.extract_tables()
                if tables:
                    extraction_stats['pages_with_tables'] += 1
                
                for table_num, table in enumerate(tables):
                    if table and len(table) > 0:
                        extraction_stats['total_tables'] += 1
                        
                        try:
                            df = pd.DataFrame(table)
                            
                            if not df.empty:
                                # Enhanced header processing
                                headers = []
                                if len(df.columns) > 0:
                                    # Check if first row contains headers
                                    first_row = df.iloc[0] if len(df) > 0 else None
                                    if first_row is not None and not pd.isna(first_row).all() and not all(x is None for x in first_row):
                                        headers = [str(h).strip() if h is not None and str(h).strip() != '' else f"Column_{i}" 
                                                  for i, h in enumerate(first_row)]
                                        df = df.iloc[1:]  # Remove header row from data
                                    else:
                                        headers = [f"Column_{i}" for i in range(len(df.columns))]
                                
                                # Ensure unique headers
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
                                
                                # Clean the dataframe
                                df = df.dropna(how='all')  # Remove completely empty rows
                                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
                            
                            document_content.append({
                                "page": page_num + 1,
                                "type": "table",
                                "table_number": table_num + 1,
                                "dataframe": df,
                                "row_count": len(df),
                                "col_count": len(df.columns) if not df.empty else 0
                            })
                            
                        except Exception as e:
                            st.warning(f"Error processing table {table_num + 1} on page {page_num + 1}: {str(e)}")
                            continue
        
        # Determine extraction quality
        avg_text_per_page = extraction_stats['total_text_length'] / max(extraction_stats['total_pages'], 1)
        text_coverage = extraction_stats['pages_with_text'] / max(extraction_stats['total_pages'], 1)
        
        if avg_text_per_page > 200 and text_coverage > 0.5:
            extraction_stats['extraction_quality'] = 'good'
        elif avg_text_per_page > 50 and text_coverage > 0.3:
            extraction_stats['extraction_quality'] = 'moderate'
        else:
            extraction_stats['extraction_quality'] = 'poor'
            
        return document_content, extraction_stats
        
    except Exception as e:
        st.error(f"Error extracting from PDF {file_path}: {str(e)}")
        extraction_stats['extraction_quality'] = 'failed'
        return [], extraction_stats

def is_scanned_pdf_enhanced(file_path):
    """Enhanced detection of scanned PDFs with detailed analysis"""
    try:
        document_content, stats = extract_tables_from_pdf(file_path)
        
        # Multiple criteria for determining if PDF is scanned
        criteria = {
            'low_text_density': stats['total_text_length'] < (stats['total_pages'] * 50),
            'low_page_coverage': (stats['pages_with_text'] / max(stats['total_pages'], 1)) < 0.3,
            'extraction_quality_poor': stats['extraction_quality'] in ['poor', 'failed'],
            'very_short_text': stats['total_text_length'] < 100
        }
        
        # If 2 or more criteria are met, consider it scanned
        scanned_indicators = sum(criteria.values())
        is_scanned = scanned_indicators >= 2
        
        # Return detailed analysis
        return {
            'is_scanned': is_scanned,
            'confidence': scanned_indicators / len(criteria),
            'stats': stats,
            'criteria_met': criteria,
            'recommendation': 'vision_api' if is_scanned else 'regular_extraction'
        }
        
    except Exception as e:
        st.error(f"Error analyzing PDF type: {str(e)}")
        return {
            'is_scanned': True,  # Default to scanned if analysis fails
            'confidence': 1.0,
            'recommendation': 'vision_api',
            'error': str(e)
        }

def setup_vision_client():
    """Setup Google Vision client with API key"""
    try:
        # Get API key from Streamlit secrets or environment
        api_key = st.secrets.get("GOOGLE_VISION_API_KEY") or os.getenv("GOOGLE_VISION_API_KEY")
        
        if not api_key:
            st.warning("⚠️ Google Vision API key not found. Please set GOOGLE_VISION_API_KEY in secrets.toml or environment variables.")
            return None
            
        # Create credentials from API key
        client = vision.ImageAnnotatorClient(
            client_options={"api_key": api_key}
        )
        return client
    except Exception as e:
        st.error(f"Error setting up Google Vision client: {str(e)}")
        return None

def pdf_to_images(pdf_path, max_pages=20):
    """Convert PDF pages to images for Vision API with page limit"""
    try:
        doc = fitz.open(pdf_path)
        images = []
        
        total_pages = len(doc)
        pages_to_process = min(total_pages, max_pages)
        
        if total_pages > max_pages:
            st.warning(f"📄 Processing first {max_pages} pages of {total_pages} total pages to manage Vision API costs.")
        
        for page_num in range(pages_to_process):
            page = doc.load_page(page_num)
            # Higher resolution for better OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            images.append({
                "data": img_data,
                "page": page_num + 1,
                "size": len(img_data)
            })
        
        doc.close()
        return images
    except Exception as e:
        st.error(f"Error converting PDF to images: {str(e)}")
        return []

def extract_text_with_vision(pdf_path):
    """Extract text from scanned PDF using Google Vision API with progress tracking"""
    vision_client = setup_vision_client()
    if not vision_client:
        return []
    
    try:
        images = pdf_to_images(pdf_path)
        if not images:
            return []
        
        documents = []
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, img_info in enumerate(images):
            # Update progress
            progress = (i + 1) / len(images)
            progress_bar.progress(progress)
            status_text.text(f"Processing page {img_info['page']} with Google Vision API...")
            
            try:
                # Prepare image for Vision API
                image = vision.Image(content=img_info["data"])
                
                # Perform text detection
                response = vision_client.text_detection(image=image)
                
                if response.error.message:
                    st.error(f"Vision API error on page {img_info['page']}: {response.error.message}")
                    continue
                
                # Extract text
                texts = response.text_annotations
                if texts and len(texts) > 0:
                    extracted_text = texts[0].description
                    
                    if extracted_text and len(extracted_text.strip()) > 10:  # Only add if meaningful text
                        doc = Document(
                            page_content=extracted_text,
                            metadata={
                                "source": pdf_path,
                                "page": img_info["page"],
                                "type": "scanned_text",
                                "extraction_method": "google_vision",
                                "char_count": len(extracted_text),
                                "image_size": img_info["size"]
                            }
                        )
                        documents.append(doc)
                
            except Exception as page_error:
                st.warning(f"Error processing page {img_info['page']}: {str(page_error)}")
                continue
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return documents
        
    except Exception as e:
        st.error(f"Error with Google Vision API: {str(e)}")
        return []

def load_customer_pdf_with_enhanced_fallback(file_path):
    """Enhanced customer PDF processing with intelligent fallback logic"""
    
    st.info(f"📄 Analyzing document: {os.path.basename(file_path)}")
    
    # Step 1: Analyze PDF type
    analysis = is_scanned_pdf_enhanced(file_path)
    
    # Display analysis results
    with st.expander(f"📊 Document Analysis - {os.path.basename(file_path)}", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Document Type", "Scanned" if analysis['is_scanned'] else "Text-based")
            st.metric("Confidence", f"{analysis['confidence']:.0%}")
            st.metric("Recommended Method", analysis['recommendation'].replace('_', ' ').title())
        
        with col2:
            if 'stats' in analysis:
                stats = analysis['stats']
                st.metric("Total Pages", stats['total_pages'])
                st.metric("Pages with Text", stats['pages_with_text'])
                st.metric("Tables Found", stats['total_tables'])
                st.metric("Text Quality", stats['extraction_quality'].title())
    
    # Step 2: Choose extraction method
    if analysis['is_scanned'] and analysis['confidence'] > 0.5:
        st.info(f"📸 Using Google Vision API for scanned document extraction...")
        
        # Try Vision API first
        vision_documents = extract_text_with_vision(file_path)
        
        if vision_documents and len(vision_documents) > 0:
            st.success(f"✅ Vision API extracted text from {len(vision_documents)} pages")
            return vision_documents, "vision_api"
        else:
            st.warning("⚠️ Vision API extraction failed or returned no text. Falling back to regular extraction...")
    
    # Step 3: Fallback to regular extraction
    st.info("📝 Using regular PDF text extraction...")
    document_content, stats = extract_tables_from_pdf(file_path)
    
    if document_content:
        # Convert to Document format
        documents = []
        for content in document_content:
            if content["type"] == "text":
                doc = Document(
                    page_content=content["content"],
                    metadata={
                        "source": file_path,
                        "page": content["page"],
                        "type": "text",
                        "extraction_method": "pdfplumber",
                        "char_count": content.get("char_count", 0)
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
                        "extraction_method": "pdfplumber",
                        "row_count": content.get("row_count", 0),
                        "col_count": content.get("col_count", 0)
                    }
                )
                documents.append(doc)
        
        extraction_method = "pdfplumber_with_tables"
        st.success(f"✅ Regular extraction found {stats['total_text_length']} characters and {stats['total_tables']} tables")
        
    else:
        st.error("❌ Both Vision API and regular extraction failed to extract meaningful content")
        documents = []
        extraction_method = "failed"
    
    return documents, extraction_method

def format_table_for_llm(df: pd.DataFrame, table_info: dict) -> str:
    """Enhanced table formatting for LLM consumption"""
    if df.empty:
        return f"Empty table on page {table_info['page']}"
    
    table_text = f"\n--- TABLE {table_info['table_number']} (Page {table_info['page']}) ---\n"
    table_text += f"Dimensions: {len(df)} rows × {len(df.columns)} columns\n\n"
    
    # Add table content with better formatting
    try:
        table_text += df.to_string(index=False, na_rep='', max_rows=50) + "\n"
    except:
        table_text += "Table content could not be formatted\n"
    
    # Extract key financial indicators
    table_text += "\nKey Financial Data from this table:\n"
    for col in df.columns:
        try:
            non_null_values = df[col].dropna()
            if not non_null_values.empty:
                # Look for numeric values or currency amounts
                numeric_values = []
                for val in non_null_values:
                    val_str = str(val).strip()
                    if val_str and (val_str.replace(',', '').replace('.', '').isdigit() or 
                                   any(char.isdigit() for char in val_str)):
                        numeric_values.append(val_str)
                
                if numeric_values:
                    table_text += f"- {col}: {', '.join(numeric_values[:5])}"
                    if len(numeric_values) > 5:
                        table_text += f" (and {len(numeric_values) - 5} more)"
                    table_text += "\n"
        except:
            continue
    
    table_text += "--- END TABLE ---\n"
    return table_text

# Example usage in your main processing function:
def process_customer_documents_enhanced(customer_files):
    """Process customer documents with enhanced extraction and detailed reporting"""
    
    if not customer_files:
        return [], {}
    
    all_customer_docs = []
    processing_summary = {
        'total_files': len(customer_files),
        'successful_extractions': 0,
        'failed_extractions': 0,
        'vision_api_used': 0,
        'regular_extraction_used': 0,
        'total_tables': 0,
        'total_pages': 0,
        'extraction_methods': {}
    }
    
    with st.spinner("🔄 Processing customer documents with enhanced extraction..."):
        for i, file in enumerate(customer_files):
            st.info(f"Processing file {i+1}/{len(customer_files)}: {file.name}")
            
            try:
                # Save uploaded file
                file_path = os.path.join(customer_docs_directory, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Process with enhanced method
                documents, extraction_method = load_customer_pdf_with_enhanced_fallback(file_path)
                
                if documents:
                    processing_summary['successful_extractions'] += 1
                    processing_summary['extraction_methods'][file.name] = extraction_method
                    
                    if extraction_method == "vision_api":
                        processing_summary['vision_api_used'] += 1
                    else:
                        processing_summary['regular_extraction_used'] += 1
                    
                    # Count tables and pages
                    for doc in documents:
                        if doc.metadata.get("type") == "table":
                            processing_summary['total_tables'] += 1
                        processing_summary['total_pages'] += 1
                    
                    all_customer_docs.extend(documents)
                    
                else:
                    processing_summary['failed_extractions'] += 1
                    st.error(f"❌ Failed to extract content from {file.name}")
                
            except Exception as e:
                processing_summary['failed_extractions'] += 1
                st.error(f"❌ Error processing {file.name}: {str(e)}")
    
    return all_customer_docs, processing_summary
