import streamlit as st
import pandas as pd
from google.cloud import vision
import io
import base64
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import Graph, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import json
import re
from datetime import datetime
import numpy as np

# State definition for LangGraph
class UnderwritingState(TypedDict):
    documents: List[Dict[str, Any]]
    extracted_data: Dict[str, Any]
    calculations: Dict[str, Any]
    recommendations: Dict[str, Any]
    errors: List[str]
    current_step: str

# Google Vision OCR Setup
def setup_vision_client():
    try:
        api_key = st.session_state.get('google_vision_api_key')
        
        if not api_key:
            st.error("⚠️ Please enter your Google Vision API key in the sidebar.")
            return None
            
        client = vision.ImageAnnotatorClient(
            client_options={"api_key": api_key}
        )
        return client
    except Exception as e:
        st.error(f"Error setting up Google Vision client: {str(e)}")
        return None

# OCR Agent
class OCRAgent:
    def __init__(self, vision_client):
        self.vision_client = vision_client
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF using Google Vision OCR"""
        try:
            # Convert PDF to images and extract text
            content = pdf_file.read()
            image = vision.Image(content=content)
            
            response = self.vision_client.text_detection(image=image)
            texts = response.text_annotations
            
            if texts:
                return texts[0].description
            return ""
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
            return ""
    
    def process_documents(self, state: UnderwritingState) -> UnderwritingState:
        """Process uploaded documents with OCR"""
        extracted_texts = {}
        
        for doc in state["documents"]:
            if doc["type"] == "pdf":
                text = self.extract_text_from_pdf(doc["file"])
                extracted_texts[doc["name"]] = {
                    "text": text,
                    "document_type": doc.get("document_type", "unknown")
                }
        
        state["extracted_data"]["ocr_results"] = extracted_texts
        state["current_step"] = "data_extraction"
        return state

# Data extraction patterns for different document types
EXTRACTION_PATTERNS = {
    "salary_slip": {
        "gross_salary": r"gross\s*salary[:\s]*₹?(\d+(?:,\d+)*)",
        "basic_salary": r"basic\s*salary[:\s]*₹?(\d+(?:,\d+)*)",
        "month": r"month[:\s]*(\w+\s*\d{4}|\d{1,2}\/\d{4})"
    },
    "bank_statement": {
        "salary_credit": r"salary|sal\s*cr.*?₹?(\d+(?:,\d+)*)",
        "closing_balance": r"closing\s*balance[:\s]*₹?(\d+(?:,\d+)*)",
        "date": r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})"
    },
    "itr": {
        "salary_income": r"salary\s*income[:\s]*₹?(\d+(?:,\d+)*)",
        "business_income": r"business\s*income[:\s]*₹?(\d+(?:,\d+)*)",
        "rental_income": r"rental\s*income[:\s]*₹?(\d+(?:,\d+)*)",
        "total_income": r"total\s*income[:\s]*₹?(\d+(?:,\d+)*)"
    },
    "form16": {
        "gross_income": r"gross\s*total\s*income[:\s]*₹?(\d+(?:,\d+)*)",
        "tax_deducted": r"tax\s*deducted[:\s]*₹?(\d+(?:,\d+)*)"
    }
}

# Data Extraction Agent
class DataExtractionAgent:
    def __init__(self):
        self.patterns = EXTRACTION_PATTERNS
    
    def extract_financial_data(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Extract financial data using regex patterns"""
        extracted_data = {}
        
        if doc_type in self.patterns:
            for field, pattern in self.patterns[doc_type].items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Clean and convert numeric values
                    if field != "month" and field != "date":
                        value = matches[0].replace(",", "")
                        try:
                            extracted_data[field] = float(value)
                        except ValueError:
                            extracted_data[field] = value
                    else:
                        extracted_data[field] = matches[0]
        
        return extracted_data
    
    def process_extraction(self, state: UnderwritingState) -> UnderwritingState:
        """Extract structured data from OCR results"""
        structured_data = {}
        
        for doc_name, ocr_result in state["extracted_data"]["ocr_results"].items():
            doc_type = ocr_result["document_type"]
            extracted = self.extract_financial_data(ocr_result["text"], doc_type)
            structured_data[doc_name] = {
                "type": doc_type,
                "data": extracted
            }
        
        state["extracted_data"]["structured_data"] = structured_data
        state["current_step"] = "calculation"
        return state

# Calculation Agent
class CalculationAgent:
    def __init__(self):
        # Age-based multipliers from the guidelines
        self.term_multipliers = {
            (18, 30): 25, (31, 35): 25, (36, 40): 20,
            (41, 45): 15, (46, 50): 12, (51, 55): 10, (56, 120): 5
        }
        self.non_term_multipliers = {
            (18, 30): 35, (31, 35): 30, (36, 40): 25,
            (41, 45): 20, (46, 50): 15, (51, 65): 10, (66, 120): 6
        }
    
    def get_age_multiplier(self, age: int, is_term: bool) -> int:
        """Get age-based income multiplier"""
        multipliers = self.term_multipliers if is_term else self.non_term_multipliers
        
        for (min_age, max_age), multiplier in multipliers.items():
            if min_age <= age <= max_age:
                return multiplier
        return 5  # Default fallback
    
    def calculate_salary_slip(self, data: Dict, age: int, is_term: bool) -> Dict[str, float]:
        """Calculate financial viability from salary slip"""
        gross_monthly = data.get("gross_salary", 0)
        annual_salary = gross_monthly * 12
        
        if is_term:
            # Term case: no bonus addition
            financial_viability = annual_salary * self.get_age_multiplier(age, is_term)
        else:
            # Non-term case: add 10% bonus
            annual_bonus = annual_salary * 0.10
            total_annual = annual_salary + annual_bonus
            financial_viability = total_annual * self.get_age_multiplier(age, is_term)
        
        return {
            "gross_monthly_salary": gross_monthly,
            "annual_salary": annual_salary,
            "annual_bonus": annual_salary * 0.10 if not is_term else 0,
            "total_annual_salary": annual_salary + (annual_salary * 0.10 if not is_term else 0),
            "financial_viability": financial_viability,
            "age_multiplier": self.get_age_multiplier(age, is_term)
        }
    
    def calculate_bank_statement_salary(self, salaries: List[float], age: int, is_term: bool) -> Dict[str, float]:
        """Calculate from bank statement salary credits"""
        if is_term:
            # Term: last 3 months average + 30%
            avg_monthly = sum(salaries[-3:]) / len(salaries[-3:]) if len(salaries) >= 3 else sum(salaries) / len(salaries)
            annual_salary = avg_monthly * 12
            gross_annual = annual_salary * 1.30  # Add 30%
        else:
            # Non-term: last 6 months average
            avg_monthly = sum(salaries[-6:]) / len(salaries[-6:]) if len(salaries) >= 6 else sum(salaries) / len(salaries)
            gross_annual = avg_monthly * 12
        
        financial_viability = gross_annual * self.get_age_multiplier(age, is_term)
        
        return {
            "average_monthly_salary": avg_monthly,
            "annual_salary": annual_salary if is_term else gross_annual,
            "gross_annual_salary": gross_annual,
            "financial_viability": financial_viability,
            "age_multiplier": self.get_age_multiplier(age, is_term)
        }
    
    def calculate_itr(self, data: Dict, age: int, is_term: bool) -> Dict[str, float]:
        """Calculate from ITR data"""
        if is_term:
            # Term: Only earned income
            total_earned = data.get("salary_income", 0) + data.get("business_income", 0)
            financial_viability = total_earned * self.get_age_multiplier(age, is_term)
        else:
            # Non-term: All income including unearned
            total_income = (data.get("salary_income", 0) + 
                          data.get("business_income", 0) + 
                          data.get("rental_income", 0) + 
                          data.get("interest_income", 0))
            financial_viability = total_income * self.get_age_multiplier(age, is_term)
        
        return {
            "total_income": total_earned if is_term else total_income,
            "financial_viability": financial_viability,
            "age_multiplier": self.get_age_multiplier(age, is_term)
        }
    
    def process_calculations(self, state: UnderwritingState) -> UnderwritingState:
        """Perform all financial calculations"""
        calculations = {}
        
        # Get user inputs
        age = state.get("age", 30)
        is_term = state.get("is_term_case", True)
        
        for doc_name, doc_info in state["extracted_data"]["structured_data"].items():
            doc_type = doc_info["type"]
            data = doc_info["data"]
            
            if doc_type == "salary_slip" and "gross_salary" in data:
                calculations[doc_name] = self.calculate_salary_slip(data, age, is_term)
            elif doc_type == "itr":
                calculations[doc_name] = self.calculate_itr(data, age, is_term)
            # Add more calculation methods as needed
        
        state["calculations"] = calculations
        state["current_step"] = "recommendation"
        return state

# Recommendation Agent
class RecommendationAgent:
    def generate_recommendations(self, state: UnderwritingState) -> UnderwritingState:
        """Generate underwriting recommendations"""
        recommendations = {
            "approved_amount": 0,
            "risk_assessment": "Low",
            "recommendations": [],
            "summary": {}
        }
        
        # Find highest financial viability
        max_viability = 0
        best_document = ""
        
        for doc_name, calc in state["calculations"].items():
            viability = calc.get("financial_viability", 0)
            if viability > max_viability:
                max_viability = viability
                best_document = doc_name
        
        recommendations["approved_amount"] = max_viability
        recommendations["best_document"] = best_document
        
        # Risk assessment based on amount
        if max_viability > 50000000:  # 5 Crores
            recommendations["risk_assessment"] = "High"
            recommendations["recommendations"].append("Requires senior underwriter approval")
        elif max_viability > 10000000:  # 1 Crore
            recommendations["risk_assessment"] = "Medium"
            recommendations["recommendations"].append("Additional verification recommended")
        else:
            recommendations["risk_assessment"] = "Low"
            recommendations["recommendations"].append("Standard processing")
        
        state["recommendations"] = recommendations
        state["current_step"] = "complete"
        return state

# Main Streamlit Application
def main():
    st.set_page_config(
        page_title="Financial Underwriter Agent System",
        page_icon="💼",
        layout="wide"
    )
    
    st.title("🏦 Financial Underwriter Agent System")
    st.markdown("AI-powered underwriting with document processing and calculations")
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        google_vision_api_key = st.text_input(
            "Google Vision API Key",
            type="password",
            help="Enter your Google Vision API key for OCR processing"
        )
        st.session_state['google_vision_api_key'] = google_vision_api_key
        
        st.header("📋 Case Details")
        age = st.number_input("Customer Age", min_value=18, max_value=80, value=30)
        is_term_case = st.radio("Case Type", ["Term", "Non-Term"]) == "Term"
        
        st.header("📄 Document Types")
        doc_type_mapping = {
            "Salary Slip": "salary_slip",
            "Bank Statement": "bank_statement", 
            "ITR": "itr",
            "Form 16": "form16"
        }
    
    # Initialize session state
    if 'underwriting_state' not in st.session_state:
        st.session_state.underwriting_state = UnderwritingState(
            documents=[],
            extracted_data={},
            calculations={},
            recommendations={},
            errors=[],
            current_step="document_upload"
        )
    
    # Update state with user inputs
    st.session_state.underwriting_state["age"] = age
    st.session_state.underwriting_state["is_term_case"] = is_term_case
    
    # Document upload section
    st.header("📤 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload Financial Documents",
        type=['pdf', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload salary slips, bank statements, ITR, Form 16, etc."
    )
    
    if uploaded_files:
        documents = []
        for file in uploaded_files:
            doc_type = st.selectbox(
                f"Document type for {file.name}",
                options=list(doc_type_mapping.keys()),
                key=f"type_{file.name}"
            )
            
            documents.append({
                "name": file.name,
                "file": file,
                "type": "pdf" if file.name.endswith('.pdf') else "image",
                "document_type": doc_type_mapping[doc_type]
            })
        
        st.session_state.underwriting_state["documents"] = documents
    
    # Process button
    if st.button("🚀 Start Underwriting Process", type="primary"):
        if not google_vision_api_key:
            st.error("Please enter Google Vision API key in the sidebar")
            return
        
        if not uploaded_files:
            st.error("Please upload at least one document")
            return
        
        # Initialize agents
        vision_client = setup_vision_client()
        if not vision_client:
            return
        
        ocr_agent = OCRAgent(vision_client)
        extraction_agent = DataExtractionAgent()
        calculation_agent = CalculationAgent()
        recommendation_agent = RecommendationAgent()
        
        # Create LangGraph workflow
        workflow = StateGraph(UnderwritingState)
        
        # Add nodes
        workflow.add_node("ocr", ocr_agent.process_documents)
        workflow.add_node("extraction", extraction_agent.process_extraction)
        workflow.add_node("calculation", calculation_agent.process_calculations)
        workflow.add_node("recommendation", recommendation_agent.generate_recommendations)
        
        # Add edges
        workflow.add_edge("ocr", "extraction")
        workflow.add_edge("extraction", "calculation")
        workflow.add_edge("calculation", "recommendation")
        workflow.add_edge("recommendation", END)
        
        # Set entry point
        workflow.set_entry_point("ocr")
        
        # Compile and run
        app = workflow.compile()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Run the workflow
            status_text.text("🔍 Processing documents with OCR...")
            progress_bar.progress(25)
            
            result = app.invoke(st.session_state.underwriting_state)
            
            status_text.text("📊 Extracting financial data...")
            progress_bar.progress(50)
            
            status_text.text("🧮 Performing calculations...")
            progress_bar.progress(75)
            
            status_text.text("✅ Generating recommendations...")
            progress_bar.progress(100)
            
            # Display results
            st.success("Underwriting process completed successfully!")
            
            # Results section
            st.header("📊 Underwriting Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💰 Financial Summary")
                if result["recommendations"]:
                    approved_amount = result["recommendations"].get("approved_amount", 0)
                    st.metric("Approved Amount", f"₹{approved_amount:,.0f}")
                    st.metric("Risk Assessment", result["recommendations"].get("risk_assessment", "Unknown"))
                    st.info(f"Best Document: {result['recommendations'].get('best_document', 'N/A')}")
            
            with col2:
                st.subheader("📋 Recommendations")
                for rec in result["recommendations"].get("recommendations", []):
                    st.write(f"• {rec}")
            
            # Detailed calculations
            if result["calculations"]:
                st.subheader("🧮 Detailed Calculations")
                for doc_name, calc in result["calculations"].items():
                    with st.expander(f"📄 {doc_name}"):
                        calc_df = pd.DataFrame([(k, f"₹{v:,.0f}" if isinstance(v, (int, float)) and k != "age_multiplier" else v) 
                                              for k, v in calc.items()], 
                                             columns=["Parameter", "Value"])
                        st.dataframe(calc_df, use_container_width=True)
            
            # Raw extracted data (for debugging)
            with st.expander("🔍 Raw Extracted Data (Debug)"):
                st.json(result["extracted_data"])
                
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            progress_bar.progress(0)
            status_text.text("❌ Processing failed")

if __name__ == "__main__":
    main()
