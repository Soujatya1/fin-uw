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
            You are a financial data extraction expert analyzing a salary slip. Please provide a detailed analysis in the following format:

            DOCUMENT ANALYSIS - SALARY SLIP
            ================================
            
            EMPLOYEE INFORMATION:
            - Employee Name: [Extract name]
            - Company Name: [Extract company]
            - Month/Period: [Extract period]
            
            SALARY BREAKDOWN:
            - Gross Monthly Salary: ₹[amount] 
            - Basic Salary: ₹[amount]
            - House Rent Allowance: ₹[amount if found]
            - Other Allowances: ₹[amount if found]
            - Total Deductions: ₹[amount if found]
            - Net Salary: ₹[amount if found]
            
            KEY OBSERVATIONS:
            - [Any important notes about salary structure]
            - [Consistency indicators]
            - [Any anomalies or concerns]
            
            EXTRACTED NUMERICAL DATA:
            GROSS_SALARY: [numeric value only]
            BASIC_SALARY: [numeric value only]
            NET_SALARY: [numeric value only]
            
            Salary Slip Text:
            {text}
            """,
            
            "bank_statement": """
            You are a financial data extraction expert analyzing a bank statement. Please provide a detailed analysis:

            DOCUMENT ANALYSIS - BANK STATEMENT
            =================================
            
            ACCOUNT INFORMATION:
            - Account Holder: [Extract name]
            - Account Number: [Extract if visible]
            - Statement Period: [Extract period]
            
            SALARY CREDIT ANALYSIS:
            - Number of Salary Credits Found: [count]
            - Salary Credit Amounts: [list all amounts found]
            - Average Monthly Salary Credit: ₹[calculated average]
            - Salary Credit Pattern: [regular/irregular]
            
            ACCOUNT HEALTH:
            - Opening Balance: ₹[amount]
            - Closing Balance: ₹[amount]
            - Minimum Balance During Period: ₹[amount if found]
            
            KEY OBSERVATIONS:
            - [Income stability assessment]
            - [Any bounce charges or penalties]
            - [Regular vs irregular income pattern]
            
            EXTRACTED NUMERICAL DATA:
            SALARY_CREDITS: [comma-separated list of amounts]
            CLOSING_BALANCE: [numeric value]
            AVERAGE_MONTHLY_SALARY: [calculated value]
            
            Bank Statement Text:
            {text}
            """,
            
            "itr": """
            You are a financial data extraction expert analyzing an Income Tax Return. Please provide a detailed analysis:

            DOCUMENT ANALYSIS - INCOME TAX RETURN
            ====================================
            
            TAXPAYER INFORMATION:
            - Name: [Extract name]
            - PAN: [Extract if visible]
            - Assessment Year: [Extract year]
            
            INCOME BREAKDOWN:
            - Salary Income: ₹[amount]
            - Business/Professional Income: ₹[amount if found]
            - Capital Gains: ₹[amount if found]
            - Income from House Property: ₹[amount if found]
            - Income from Other Sources: ₹[amount if found]
            - Total Income: ₹[calculated total]
            
            TAX INFORMATION:
            - Total Tax Liability: ₹[amount if found]
            - Tax Paid/TDS: ₹[amount if found]
            - Refund/Balance Tax: ₹[amount if found]
            
            KEY OBSERVATIONS:
            - [Primary income source analysis]
            - [Income diversity assessment]
            - [Tax compliance indicators]
            
            EXTRACTED NUMERICAL DATA:
            SALARY_INCOME: [numeric value]
            BUSINESS_INCOME: [numeric value]
            RENTAL_INCOME: [numeric value]
            TOTAL_INCOME: [numeric value]
            
            ITR Text:
            {text}
            """,
            
            "form16": """
            You are a financial data extraction expert analyzing Form 16. Please provide a detailed analysis:

            DOCUMENT ANALYSIS - FORM 16
            ===========================
            
            EMPLOYEE & EMPLOYER DETAILS:
            - Employee Name: [Extract name]
            - Employee PAN: [Extract if visible]
            - Employer Name: [Extract company]
            - Employer TAN: [Extract if visible]
            - Financial Year: [Extract year]
            
            INCOME & TAX DETAILS:
            - Gross Total Income: ₹[amount]
            - Total Income (after deductions): ₹[amount]
            - Tax Deducted at Source: ₹[amount]
            - Income Tax: ₹[amount if separate]
            - Education Cess: ₹[amount if found]
            
            DEDUCTIONS CLAIMED:
            - Section 80C: ₹[amount if found]
            - Section 80D: ₹[amount if found]
            - Other Deductions: ₹[amount if found]
            
            KEY OBSERVATIONS:
            - [Tax planning assessment]
            - [Deduction utilization analysis]
            - [Income consistency with other documents]
            
            EXTRACTED NUMERICAL DATA:
            GROSS_INCOME: [numeric value]
            TOTAL_INCOME: [numeric value]
            TDS_AMOUNT: [numeric value]
            
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
                
                # Parse structured response to extract numerical data
                extracted_data = self._parse_llm_response(response.content, doc_type)
                
                # Store the full LLM analysis for display
                extracted_data["llm_analysis"] = response.content
                
                return extracted_data
            
            return {}
        except Exception as e:
            st.error(f"LLM extraction error: {str(e)}")
            return self._fallback_extraction(text, doc_type)
    
    def _parse_llm_response(self, response_text: str, doc_type: str) -> Dict[str, Any]:
        """Parse structured LLM response to extract numerical data"""
        extracted_data = {}
        
        # Look for the EXTRACTED NUMERICAL DATA section
        lines = response_text.split('\n')
        in_data_section = False
        
        for line in lines:
            if "EXTRACTED NUMERICAL DATA:" in line:
                in_data_section = True
                continue
            
            if in_data_section and line.strip():
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Handle different data types
                    if key == "SALARY_CREDITS":
                        # Parse comma-separated list
                        try:
                            amounts = [float(x.strip()) for x in value.split(',') if x.strip().replace('.','').isdigit()]
                            extracted_data["salary_credits"] = amounts
                        except:
                            pass
                    else:
                        # Try to extract numeric value
                        try:
                            # Remove currency symbols and commas
                            clean_value = re.sub(r'[₹,\s]', '', value)
                            if clean_value.replace('.', '').isdigit():
                                extracted_data[key.lower().replace('_', '_')] = float(clean_value)
                        except:
                            extracted_data[key.lower().replace('_', '_')] = value
        
        return extracted_data
    
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
        You are a senior financial underwriter with 15+ years of experience. Analyze the following case and provide detailed calculations:

        CASE DETAILS:
        =============
        Customer Age: {age} years
        Insurance Type: {case_type}
        
        DOCUMENT ANALYSIS:
        ==================
        {document_data}

        UNDERWRITING GUIDELINES:
        ========================
        
        AGE-BASED MULTIPLIERS:
        
        For Term Life Insurance:
        • Ages 18-30: 25x annual income
        • Ages 31-35: 25x annual income  
        • Ages 36-40: 20x annual income
        • Ages 41-45: 15x annual income
        • Ages 46-50: 12x annual income
        • Ages 51-55: 10x annual income
        • Ages 56+: 5x annual income
        
        For Non-Term Life Insurance:
        • Ages 18-30: 35x annual income
        • Ages 31-35: 30x annual income
        • Ages 36-40: 25x annual income
        • Ages 41-45: 20x annual income
        • Ages 46-50: 15x annual income
        • Ages 51-65: 10x annual income
        • Ages 66+: 6x annual income

        DOCUMENT-SPECIFIC CALCULATION RULES:
        ====================================
        
        SALARY SLIP CALCULATIONS:
        • Term Insurance: Annual salary × appropriate multiplier
        • Non-Term Insurance: (Annual salary + 10% bonus) × appropriate multiplier
        
        BANK STATEMENT CALCULATIONS:
        • Term Insurance: Average of last 3 months salary + 30% grossing up, then × 12 × multiplier
        • Non-Term Insurance: Average of last 6 months salary × 12 × multiplier
        
        ITR CALCULATIONS:
        • Term Insurance: Only earned income (salary + business) × multiplier
        • Non-Term Insurance: Total income (including rental, interest) × multiplier
        
        FORM 16 CALCULATIONS:
        • Use gross total income × appropriate multiplier

        Please provide your analysis in the following format:

        FINANCIAL VIABILITY CALCULATION
        ==============================
        
        INCOME ASSESSMENT:
        • Primary Income Source: [source and amount]
        • Monthly Income: ₹[amount]
        • Annual Income: ₹[amount]
        • Income Adjustments: [any adjustments made]
        • Final Annual Income: ₹[amount]
        
        MULTIPLIER APPLICATION:
        • Customer Age: {age} years
        • Insurance Type: {case_type}
        • Applicable Multiplier: [X]x
        • Justification: [why this multiplier applies]
        
        FINANCIAL VIABILITY CALCULATION:
        • Formula: ₹[final annual income] × [multiplier]
        • Financial Viability: ₹[final calculated amount]
        
        RISK ASSESSMENT:
        • Income Stability: [High/Medium/Low]
        • Documentation Quality: [Excellent/Good/Fair/Poor]
        • Consistency Check: [Pass/Fail with explanation]
        • Overall Risk Level: [Low/Medium/High]
        
        KEY OBSERVATIONS:
        • [Important findings about the customer's financial profile]
        • [Any red flags or positive indicators]
        • [Recommendations for underwriting decision]
        
        NUMERICAL SUMMARY:
        ANNUAL_INCOME: [numeric value]
        MULTIPLIER: [numeric value]
        FINANCIAL_VIABILITY: [numeric value]
        RISK_SCORE: [1-10 scale]
        """
    
    def process_calculations(self, state: UnderwritingState) -> UnderwritingState:
        """Perform calculations using LLM"""
        llm = state["chat_groq"]
        calculations = {}
        
        age = state["age"]
        case_type = "Term Life Insurance" if state["is_term_case"] else "Non-Term Life Insurance"
        
        for doc_name, doc_info in state["extracted_data"]["structured_data"].items():
            try:
                # Format document data for LLM
                doc_data_str = f"Document Type: {doc_info['type']}\n"
                doc_data_str += f"Document Name: {doc_name}\n"
                doc_data_str += "Extracted Data:\n"
                
                for key, value in doc_info['data'].items():
                    if key != "llm_analysis":  # Skip the analysis text
                        doc_data_str += f"  {key}: {value}\n"
                
                prompt = self.calculation_prompt.format(
                    age=age,
                    case_type=case_type,
                    document_data=doc_data_str
                )
                
                response = llm.invoke([HumanMessage(content=prompt)])
                
                # Parse numerical data from response
                calc_result = self._parse_calculation_response(response.content)
                
                # Store full LLM analysis
                calc_result["llm_calculation_analysis"] = response.content
                
                calculations[doc_name] = calc_result
                    
            except Exception as e:
                st.error(f"Calculation error for {doc_name}: {str(e)}")
                calculations[doc_name] = {"error": str(e)}
        
        state["calculations"] = calculations
        state["current_step"] = "recommendation"
        return state
    
    def _parse_calculation_response(self, response_text: str) -> Dict:
        """Parse calculation response to extract numerical data"""
        result = {}
        
        # Look for NUMERICAL SUMMARY section
        lines = response_text.split('\n')
        in_summary_section = False
        
        for line in lines:
            if "NUMERICAL SUMMARY:" in line:
                in_summary_section = True
                continue
            
            if in_summary_section and line.strip() and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                try:
                    # Clean and convert numeric values
                    clean_value = re.sub(r'[₹,\s]', '', value)
                    if clean_value.replace('.', '').isdigit():
                        result[key.lower()] = float(clean_value)
                    else:
                        result[key.lower()] = value
                except:
                    result[key.lower()] = value
        
        # Also try to extract key amounts from the text
        financial_viability_match = re.search(r'Financial Viability:\s*₹([0-9,]+)', response_text)
        if financial_viability_match:
            amount = financial_viability_match.group(1).replace(',', '')
            result['financial_viability'] = float(amount)
        
        return result
    
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
        You are the Chief Underwriting Officer of a leading life insurance company with 20+ years of experience. 
        Review the following underwriting case and provide your executive recommendation:

        CASE SUMMARY:
        =============
        Customer Age: {age} years
        Insurance Type: {case_type}
        
        FINANCIAL ANALYSIS RESULTS:
        ===========================
        {calculations}

        UNDERWRITING DECISION FRAMEWORK:
        ================================
        
        APPROVAL THRESHOLDS:
        • Ages 18-40: Up to ₹5 Crores (standard approval)
        • Ages 41-50: Up to ₹3 Crores (standard approval)  
        • Ages 51-60: Up to ₹2 Crores (standard approval)
        • Ages 60+: Up to ₹1 Crore (standard approval)
        
        RISK CATEGORIES:
        • Low Risk: Stable employment, consistent income, good documentation
        • Medium Risk: Some inconsistencies, requires additional verification
        • High Risk: Significant concerns, senior management approval required
        
        DOCUMENTATION REQUIREMENTS:
        • Excellent: All documents consistent, no additional requirements
        • Good: Minor clarifications needed
        • Fair: Additional documents required
        • Poor: Significant documentation gaps, high risk

        Please provide your comprehensive recommendation in the following format:

        EXECUTIVE UNDERWRITING RECOMMENDATION
        ====================================
        
        CASE OVERVIEW:
        • Customer Profile: [Brief customer summary]
        • Application Type: {case_type}
        • Total Documents Reviewed: [number]
        • Primary Income Source: [source]
        
        FINANCIAL ASSESSMENT:
        • Highest Calculated Amount: ₹[amount]
        • Recommended Approval Amount: ₹[amount]
        • Basis for Recommendation: [Best document/calculation used]
        • Income Verification Status: [Verified/Pending/Concerns]
        
        RISK EVALUATION:
        • Overall Risk Rating: [Low/Medium/High]
        • Risk Factors Identified:
          - [List specific risk factors]
        • Mitigating Factors:
          - [List positive factors]
        • Confidence Level: [1-10 scale]
        
        COMPLIANCE & DOCUMENTATION:
        • Documentation Completeness: [Excellent/Good/Fair/Poor]
        • Additional Documents Required:
          - [List any additional documents needed]
        • Compliance Concerns: [Any regulatory concerns]
        
        UNDERWRITING CONDITIONS:
        • Medical Requirements: [Standard/Enhanced/Specialized]
        • Financial Conditions: [Any specific conditions]
        • Policy Restrictions: [Any restrictions to be applied]
        • Review Period: [When to review the case]
        
        FINAL RECOMMENDATION:
        • Decision: [APPROVE/DECLINE/REFER TO SENIOR UNDERWRITER]
        • Approved Amount: ₹[final amount]
        • Validity Period: [How long this decision is valid]
        • Next Steps: [What needs to happen next]
        
        EXECUTIVE SUMMARY:
        [2-3 sentence summary of the key recommendation and rationale]
        
        NUMERICAL SUMMARY:
        APPROVED_AMOUNT: [numeric value]
        RISK_SCORE: [1-10 numeric value]
        CONFIDENCE_SCORE: [1-10 numeric value]
        DOCUMENTATION_SCORE: [1-10 numeric value]
        """
    
    def generate_recommendations(self, state: UnderwritingState) -> UnderwritingState:
        """Generate recommendations using LLM"""
        llm = state["chat_groq"]
        
        try:
            # Format calculations for LLM
            calc_summary = ""
            for doc_name, calc in state["calculations"].items():
                calc_summary += f"\nDocument: {doc_name}\n"
                calc_summary += f"Financial Viability: ₹{calc.get('financial_viability', 0):,.0f}\n"
                calc_summary += f"Risk Score: {calc.get('risk_score', 'N/A')}\n"
                if 'llm_calculation_analysis' in calc:
                    # Include key points from calculation analysis
                    analysis_lines = calc['llm_calculation_analysis'].split('\n')
                    for line in analysis_lines:
                        if any(keyword in line.upper() for keyword in ['RISK', 'INCOME', 'VIABILITY', 'OBSERVATION']):
                            calc_summary += f"  {line.strip()}\n"
                calc_summary += "---\n"
            
            case_type = "Term Life Insurance" if state["is_term_case"] else "Non-Term Life Insurance"
            
            prompt = self.recommendation_prompt.format(
                age=state["age"],
                case_type=case_type,
                calculations=calc_summary
            )
            
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Parse numerical data from response
            recommendations = self._parse_recommendation_response(response.content)
            
            # Store full LLM analysis
            recommendations["llm_recommendation_analysis"] = response.content
            
        except Exception as e:
            st.error(f"Recommendation generation error: {str(e)}")
            recommendations = self._fallback_recommendations(state)
        
        state["recommendations"] = recommendations
        state["current_step"] = "complete"
        return state
    
    def _parse_recommendation_response(self, response_text: str) -> Dict:
        """Parse recommendation response to extract key data"""
        result = {}
        
        # Extract decision
        if "APPROVE" in response_text.upper():
            result["decision"] = "APPROVED"
        elif "DECLINE" in response_text.upper():
            result["decision"] = "DECLINED"
        elif "REFER" in response_text.upper():
            result["decision"] = "REFER TO SENIOR"
        else:
            result["decision"] = "PENDING REVIEW"
        
        # Extract numerical summary
        lines = response_text.split('\n')
        in_summary_section = False
        
        for line in lines:
            if "NUMERICAL SUMMARY:" in line:
                in_summary_section = True
                continue
            
            if in_summary_section and line.strip() and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                try:
                    clean_value = re.sub(r'[₹,\s]', '', value)
                    if clean_value.replace('.', '').isdigit():
                        result[key.lower()] = float(clean_value)
                    else:
                        result[key.lower()] = value
                except:
                    result[key.lower()] = value
        
        # Extract approved amount from text
        approved_match = re.search(r'Approved Amount:\s*₹([0-9,]+)', response_text)
        if approved_match:
            amount = approved_match.group(1).replace(',', '')
            result['approved_amount'] = float(amount)
        
        # Extract risk assessment
        if "Low Risk" in response_text or "LOW RISK" in response_text:
            result["risk_assessment"] = "Low"
        elif "High Risk" in response_text or "HIGH RISK" in response_text:
            result["risk_assessment"] = "High"
        else:
            result["risk_assessment"] = "Medium"
        
        return result
    
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
            ["mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"],
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
