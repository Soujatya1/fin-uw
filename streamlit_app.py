import streamlit as st
import pandas as pd
from google.cloud import vision
import io
import base64
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import Graph, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
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
    age: int
    is_term_case: bool
    chat_groq: Any

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

# Setup ChatGroq
def setup_chatgroq():
    try:
        api_key = st.session_state.get('groq_api_key')
        
        if not api_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar.")
            return None
            
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="mixtral-8x7b-32768",  # You can change this to other models
            temperature=0.1,
            max_tokens=4000
        )
        return llm
    except Exception as e:
        st.error(f"Error setting up ChatGroq: {str(e)}")
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

# LLM-Powered Data Extraction Agent
class LLMDataExtractionAgent:
    def __init__(self):
        self.extraction_prompts = {
            "salary_slip": """
            You are a financial data extraction expert. Extract the following information from this salary slip text:
            - Gross salary (monthly)
            - Basic salary
            - Month/Period
            - Any allowances or deductions
            - Employee name
            - Company name
            
            Return the data in JSON format with numeric values as numbers (not strings).
            
            Salary Slip Text:
            {text}
            """,
            
            "bank_statement": """
            You are a financial data extraction expert. Extract the following information from this bank statement:
            - All salary credits (look for patterns like "salary", "sal cr", etc.)
            - Closing balance
            - Statement period
            - Account holder name
            - Transaction dates
            
            Return the data in JSON format. For salary credits, provide an array of amounts.
            
            Bank Statement Text:
            {text}
            """,
            
            "itr": """
            You are a financial data extraction expert. Extract the following information from this ITR document:
            - Salary income
            - Business income
            - Rental income
            - Interest income
            - Total income
            - Tax year
            - Name of taxpayer
            
            Return the data in JSON format with numeric values as numbers.
            
            ITR Text:
            {text}
            """,
            
            "form16": """
            You are a financial data extraction expert. Extract the following information from this Form 16:
            - Gross total income
            - Tax deducted at source
            - Financial year
            - Employee name
            - Employer name
            
            Return the data in JSON format with numeric values as numbers.
            
            Form 16 Text:
            {text}
            """
        }
    
    def extract_financial_data(self, text: str, doc_type: str, llm: ChatGroq) -> Dict[str, Any]:
        """Extract financial data using LLM"""
        try:
            if doc_type in self.extraction_prompts:
                prompt = self.extraction_prompts[doc_type].format(text=text)
                
                response = llm.invoke([HumanMessage(content=prompt)])
                
                # Try to parse JSON response
                try:
                    extracted_data = json.loads(response.content)
                    return extracted_data
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract using regex as fallback
                    st.warning(f"LLM response wasn't valid JSON for {doc_type}, using fallback extraction")
                    return self._fallback_extraction(text, doc_type)
            
            return {}
        except Exception as e:
            st.error(f"LLM extraction error: {str(e)}")
            return self._fallback_extraction(text, doc_type)
    
    def _fallback_extraction(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Fallback regex extraction if LLM fails"""
        patterns = {
            "salary_slip": {
                "gross_salary": r"gross\s*salary[:\s]*₹?(\d+(?:,\d+)*)",
                "basic_salary": r"basic\s*salary[:\s]*₹?(\d+(?:,\d+)*)",
            },
            "bank_statement": {
                "salary_credits": r"salary|sal\s*cr.*?₹?(\d+(?:,\d+)*)",
            },
            "itr": {
                "total_income": r"total\s*income[:\s]*₹?(\d+(?:,\d+)*)",
            }
        }
        
        extracted_data = {}
        if doc_type in patterns:
            for field, pattern in patterns[doc_type].items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if field == "salary_credits":
                        extracted_data[field] = [float(m.replace(",", "")) for m in matches]
                    else:
                        try:
                            extracted_data[field] = float(matches[0].replace(",", ""))
                        except ValueError:
                            extracted_data[field] = matches[0]
        
        return extracted_data
    
    def process_extraction(self, state: UnderwritingState) -> UnderwritingState:
        """Extract structured data from OCR results using LLM"""
        structured_data = {}
        llm = state["chat_groq"]
        
        for doc_name, ocr_result in state["extracted_data"]["ocr_results"].items():
            doc_type = ocr_result["document_type"]
            extracted = self.extract_financial_data(ocr_result["text"], doc_type, llm)
            structured_data[doc_name] = {
                "type": doc_type,
                "data": extracted
            }
        
        state["extracted_data"]["structured_data"] = structured_data
        state["current_step"] = "calculation"
        return state

# LLM-Powered Calculation Agent
class LLMCalculationAgent:
    def __init__(self):
        self.calculation_prompt = """
        You are a financial underwriting expert. Calculate the financial viability for life insurance based on the following:

        Customer Details:
        - Age: {age}
        - Case Type: {case_type}

        Document Data:
        {document_data}

        Underwriting Guidelines:
        1. Age-based multipliers for Term Insurance:
           - 18-30: 25x, 31-35: 25x, 36-40: 20x, 41-45: 15x, 46-50: 12x, 51-55: 10x, 56+: 5x
        
        2. Age-based multipliers for Non-Term Insurance:
           - 18-30: 35x, 31-35: 30x, 36-40: 25x, 41-45: 20x, 46-50: 15x, 51-65: 10x, 66+: 6x

        3. Document-specific calculations:
           - Salary Slip: For Term - use annual salary × multiplier; For Non-Term - add 10% bonus then × multiplier
           - Bank Statement: For Term - last 3 months avg + 30%, then × 12 × multiplier; For Non-Term - last 6 months avg × 12 × multiplier
           - ITR: For Term - only earned income × multiplier; For Non-Term - all income × multiplier

        Calculate and return results in JSON format with:
        - financial_viability: final calculated amount
        - calculation_method: explanation of how it was calculated
        - annual_income: calculated annual income
        - multiplier_used: age-based multiplier applied
        - risk_factors: any identified risk factors
        """
    
    def process_calculations(self, state: UnderwritingState) -> UnderwritingState:
        """Perform calculations using LLM"""
        llm = state["chat_groq"]
        calculations = {}
        
        age = state["age"]
        case_type = "Term" if state["is_term_case"] else "Non-Term"
        
        for doc_name, doc_info in state["extracted_data"]["structured_data"].items():
            try:
                prompt = self.calculation_prompt.format(
                    age=age,
                    case_type=case_type,
                    document_data=json.dumps(doc_info, indent=2)
                )
                
                response = llm.invoke([HumanMessage(content=prompt)])
                
                # Parse LLM response
                try:
                    calc_result = json.loads(response.content)
                    calculations[doc_name] = calc_result
                except json.JSONDecodeError:
                    # Fallback calculation
                    calculations[doc_name] = self._fallback_calculation(doc_info, age, state["is_term_case"])
                    
            except Exception as e:
                st.error(f"Calculation error for {doc_name}: {str(e)}")
                calculations[doc_name] = {"error": str(e)}
        
        state["calculations"] = calculations
        state["current_step"] = "recommendation"
        return state
    
    def _fallback_calculation(self, doc_info: Dict, age: int, is_term: bool) -> Dict:
        """Simple fallback calculation"""
        data = doc_info.get("data", {})
        
        # Simple multiplier logic
        if age <= 30:
            multiplier = 25 if is_term else 35
        elif age <= 40:
            multiplier = 20 if is_term else 25
        else:
            multiplier = 15 if is_term else 20
        
        # Basic calculation
        income = 0
        if "gross_salary" in data:
            income = data["gross_salary"] * 12
        elif "total_income" in data:
            income = data["total_income"]
        
        return {
            "financial_viability": income * multiplier,
            "calculation_method": "Fallback calculation",
            "annual_income": income,
            "multiplier_used": multiplier
        }

# LLM-Powered Recommendation Agent
class LLMRecommendationAgent:
    def __init__(self):
        self.recommendation_prompt = """
        You are a senior underwriting manager. Based on the following financial analysis, provide comprehensive underwriting recommendations:

        Customer Profile:
        - Age: {age}
        - Case Type: {case_type}

        Financial Calculations:
        {calculations}

        Provide recommendations in JSON format with:
        - approved_amount: highest viable amount
        - risk_assessment: Low/Medium/High
        - confidence_score: 1-10 scale
        - recommendations: array of specific recommendations
        - required_documents: any additional documents needed
        - approval_conditions: any conditions for approval
        - summary: brief executive summary
        - red_flags: any concerns identified

        Consider factors like:
        - Consistency across documents
        - Income stability
        - Age-appropriate coverage
        - Risk indicators
        """
    
    def generate_recommendations(self, state: UnderwritingState) -> UnderwritingState:
        """Generate recommendations using LLM"""
        llm = state["chat_groq"]
        
        try:
            prompt = self.recommendation_prompt.format(
                age=state["age"],
                case_type="Term" if state["is_term_case"] else "Non-Term",
                calculations=json.dumps(state["calculations"], indent=2)
            )
            
            response = llm.invoke([HumanMessage(content=prompt)])
            
            try:
                recommendations = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback recommendations
                recommendations = self._fallback_recommendations(state)
            
        except Exception as e:
            st.error(f"Recommendation generation error: {str(e)}")
            recommendations = self._fallback_recommendations(state)
        
        state["recommendations"] = recommendations
        state["current_step"] = "complete"
        return state
    
    def _fallback_recommendations(self, state: UnderwritingState) -> Dict:
        """Simple fallback recommendations"""
        max_viability = 0
        best_doc = ""
        
        for doc_name, calc in state["calculations"].items():
            viability = calc.get("financial_viability", 0)
            if viability > max_viability:
                max_viability = viability
                best_doc = doc_name
        
        return {
            "approved_amount": max_viability,
            "risk_assessment": "Medium",
            "confidence_score": 7,
            "recommendations": ["Standard processing recommended"],
            "summary": f"Based on {best_doc}, approved amount: ₹{max_viability:,.0f}"
        }

# Main Streamlit Application
def main():
    st.set_page_config(
        page_title="AI Financial Underwriter with ChatGroq",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI Financial Underwriter with ChatGroq")
    st.markdown("Intelligent underwriting powered by ChatGroq LLM and document processing")
    
    # Sidebar for API keys and settings
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        groq_api_key = st.text_input(
            "ChatGroq API Key",
            type="password",
            help="Enter your ChatGroq API key"
        )
        st.session_state['groq_api_key'] = groq_api_key
        
        google_vision_api_key = st.text_input(
            "Google Vision API Key",
            type="password",
            help="Enter your Google Vision API key for OCR processing"
        )
        st.session_state['google_vision_api_key'] = google_vision_api_key
        
        st.header("📋 Case Details")
        age = st.number_input("Customer Age", min_value=18, max_value=80, value=30)
        is_term_case = st.radio("Case Type", ["Term", "Non-Term"]) == "Term"
        
        st.header("🤖 LLM Settings")
        model_name = st.selectbox(
            "ChatGroq Model",
            ["meta-llama/llama-4-scout-17b-16e-instruct"],
            help="Select the ChatGroq model to use"
        )
        
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
            current_step="document_upload",
            age=30,
            is_term_case=True,
            chat_groq=None
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
    if st.button("🚀 Start AI Underwriting Process", type="primary"):
        if not groq_api_key:
            st.error("Please enter ChatGroq API key in the sidebar")
            return
        
        if not google_vision_api_key:
            st.error("Please enter Google Vision API key in the sidebar")
            return
        
        if not uploaded_files:
            st.error("Please upload at least one document")
            return
        
        # Setup clients
        vision_client = setup_vision_client()
        chat_groq = setup_chatgroq()
        
        if not vision_client or not chat_groq:
            return
        
        # Add ChatGroq to state
        st.session_state.underwriting_state["chat_groq"] = chat_groq
        
        # Initialize agents
        ocr_agent = OCRAgent(vision_client)
        extraction_agent = LLMDataExtractionAgent()
        calculation_agent = LLMCalculationAgent()
        recommendation_agent = LLMRecommendationAgent()
        
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
            status_text.text("🔍 Extracting text from documents...")
            progress_bar.progress(20)
            
            status_text.text("🤖 AI analyzing document content...")
            progress_bar.progress(40)
            
            status_text.text("🧮 AI performing financial calculations...")
            progress_bar.progress(70)
            
            status_text.text("📊 Generating AI recommendations...")
            progress_bar.progress(90)
            
            result = app.invoke(st.session_state.underwriting_state)
            
            progress_bar.progress(100)
            status_text.text("✅ AI underwriting complete!")
            
            # Display results
            st.success("🎉 AI Underwriting process completed successfully!")
            
            # Results section
            st.header("📊 AI Underwriting Results")
            
            # Main metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                approved_amount = result["recommendations"].get("approved_amount", 0)
                st.metric("💰 Approved Amount", f"₹{approved_amount:,.0f}")
            
            with col2:
                risk_assessment = result["recommendations"].get("risk_assessment", "Unknown")
                st.metric("⚠️ Risk Assessment", risk_assessment)
            
            with col3:
                confidence_score = result["recommendations"].get("confidence_score", 0)
                st.metric("🎯 Confidence Score", f"{confidence_score}/10")
            
            # Detailed recommendations
            st.subheader("🤖 AI Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Key Recommendations:**")
                for rec in result["recommendations"].get("recommendations", []):
                    st.write(f"• {rec}")
                
                if "required_documents" in result["recommendations"]:
                    st.write("**Additional Documents Needed:**")
                    for doc in result["recommendations"]["required_documents"]:
                        st.write(f"• {doc}")
            
            with col2:
                if "summary" in result["recommendations"]:
                    st.write("**Executive Summary:**")
                    st.info(result["recommendations"]["summary"])
                
                if "red_flags" in result["recommendations"] and result["recommendations"]["red_flags"]:
                    st.write("**⚠️ Red Flags:**")
                    for flag in result["recommendations"]["red_flags"]:
                        st.warning(f"• {flag}")
            
            # Detailed calculations
            if result["calculations"]:
                st.subheader("🧮 AI Financial Analysis")
                for doc_name, calc in result["calculations"].items():
                    with st.expander(f"📄 Analysis: {doc_name}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if "financial_viability" in calc:
                                st.metric("Financial Viability", f"₹{calc['financial_viability']:,.0f}")
                            if "annual_income" in calc:
                                st.metric("Annual Income", f"₹{calc['annual_income']:,.0f}")
                            if "multiplier_used" in calc:
                                st.metric("Multiplier Used", f"{calc['multiplier_used']}x")
                        
                        with col2:
                            if "calculation_method" in calc:
                                st.write("**Method:**", calc["calculation_method"])
                            if "risk_factors" in calc:
                                st.write("**Risk Factors:**")
                                for factor in calc["risk_factors"]:
                                    st.write(f"• {factor}")
            
            # Raw data for debugging
            with st.expander("🔍 Raw Analysis Data (Debug)"):
                st.json(result)
                
        except Exception as e:
            st.error(f"❌ Error during AI processing: {str(e)}")
            progress_bar.progress(0)
            status_text.text("❌ Processing failed")

if __name__ == "__main__":
    main()
