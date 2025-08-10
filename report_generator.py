import streamlit as st
import base64
from fpdf import FPDF
import datetime
from typing import Dict, Any, List
import os
import re

def sanitize_text(text):
    """Replace all non-ASCII characters with ASCII equivalents"""
    if not isinstance(text, str):
        return str(text)
    
    # Replace common Unicode characters
    replacements = {
        '₹': 'Rs.',
        '✅': '[OK]',
        '❌': '[X]',
        '→': '->',
        '←': '<-',
        '✨': '*',
        '💰': '$',
        '📊': '[CHART]',
        '🧠': '[AI]',
        '💡': '[TIP]',
        '🎯': '[TARGET]',
        '👉': '->',
        '📥': '[DOWNLOAD]',
        '✓': '[CHECK]',
        '✗': '[X]',
        '–': '-',
        '—': '-',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '…': '...',
        '•': '*',
        '·': '*',
        '▪': '*',
        '◦': '*',
        '➔': '->',
        '➢': '->',
        '❖': '+',
        '★': '*',
        '☆': '*',
        '✦': '*',
        '❏': '[  ]',
        '✓': '[x]',
        '✔': '[x]'
    }
    
    # Replace all known special characters
    for unicode_char, ascii_char in replacements.items():
        if unicode_char in text:
            text = text.replace(unicode_char, ascii_char)
    
    # Replace any remaining non-ASCII characters with spaces or approximations
    result = ""
    for char in text:
        if ord(char) < 128:  # ASCII range
            result += char
        else:
            result += ' '  # Replace with space
            
    return result

def format_currency_safe(value):
    """Format currency value without Unicode symbols"""
    if isinstance(value, (int, float)):
        return f"Rs. {value:,.2f}"
    return str(value)

def generate_tax_report(input_data: Dict[str, Any], results: Dict[str, Any]) -> str:
    """Generate a PDF tax planning report"""
    try:
        # Create PDF instance
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(24, 136, 229)  # Blue color
        pdf.cell(0, 10, 'AI Tax Planner - Tax Report', 0, 1, 'C')
        
        # Date
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(100, 100, 100)  # Gray color
        pdf.cell(0, 5, f"Generated on: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}", 0, 1, 'C')
        pdf.ln(10)
        
        # Add user information section
        pdf.set_font('Arial', 'B', 14)
        pdf.set_text_color(24, 136, 229)
        pdf.cell(0, 10, "Your Financial Information", 0, 1)
        pdf.ln(4)
        
        # User financial details
        pdf.set_font('Arial', '', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 7, f"Annual Income: {format_currency_safe(input_data.get('AnnualIncome', 0))}", 0, 1)
        pdf.cell(0, 7, f"80C Investments: {format_currency_safe(input_data.get('Investments', 0))}", 0, 1)
        pdf.cell(0, 7, f"Other Deductions: {format_currency_safe(input_data.get('Deductions', 0))}", 0, 1)
        pdf.cell(0, 7, f"HRA: {format_currency_safe(input_data.get('HRA', 0))}", 0, 1)
        pdf.cell(0, 7, f"Age: {input_data.get('Age', 0)}", 0, 1)
        pdf.cell(0, 7, f"Employment Type: {sanitize_text(input_data.get('EmploymentType', 'Not specified'))}", 0, 1)
        pdf.ln(10)
        
        # Tax metrics section
        pdf.set_font('Arial', 'B', 14)
        pdf.set_text_color(24, 136, 229)
        pdf.cell(0, 10, "Your Tax Analysis", 0, 1)
        
        # Key metrics
        pdf.set_font('Arial', 'B', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(90, 10, "Annual Tax:", 0, 0)
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 10, format_currency_safe(results.get('tax_liability', 0)), 0, 1)
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(90, 10, "Monthly Tax:", 0, 0)
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 10, format_currency_safe(results.get('monthly_tax', 0)), 0, 1)
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(90, 10, "Effective Tax Rate:", 0, 0)
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 10, f"{results.get('effective_rate', 0):.2f}%", 0, 1)
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(90, 10, "Tax Savings:", 0, 0)
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 10, format_currency_safe(results.get('tax_savings', 0)), 0, 1)
        
        pdf.ln(5)
        
        # Tax strategy section
        pdf.set_font('Arial', 'B', 14)
        pdf.set_text_color(24, 136, 229)
        pdf.cell(0, 10, "AI Strategy Recommendations", 0, 1)
        
        # Strategy details
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(24, 136, 229)
        pdf.cell(0, 10, f"Strategy: {sanitize_text(results.get('recommended_strategy', 'No strategy available'))}", 0, 1)
        
        pdf.set_font('Arial', '', 11)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 6, sanitize_text(results.get('strategy_description', '')))
        pdf.ln(5)
        
        # Recommendations
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "Specific Recommendations:", 0, 1)
        
        # Sanitize all recommendations
        sanitized_recommendations = []
        for rec in results.get('recommendations', []):
            sanitized_recommendations.append(sanitize_text(rec))
            
        for rec in sanitized_recommendations:
            pdf.set_font('Arial', '', 11)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(5, 6, "*", 0, 0)
            pdf.multi_cell(0, 6, rec)
        
        # Potential savings
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 11)
        pdf.set_fill_color(240, 248, 255)  # Light blue background
        pdf.set_text_color(24, 136, 229)
        pdf.cell(0, 10, f"Potential additional savings: {sanitize_text(results.get('potential_savings', 'N/A'))}", 1, 1, 'L', True)
        
        # Age-based tips
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.set_text_color(24, 136, 229)
        pdf.cell(0, 10, "Personalized Tax Planning Tips", 0, 1)
        
        # Determine career stage
        age = input_data.get('Age', 35)
        investments = input_data.get('Investments', 0)
        
        if age < 30:
            career_stage = "Early-Career Focus"
            tips = [
                "Start small with ELSS mutual funds",
                "Consider PPF for long-term tax benefits",
                "Term insurance premium",
            ]
            
        elif age < 45:
            career_stage = "Mid-Career Focus" 
            tips = [
                "Maximize home loan benefits (principal under 80C)",
                "ELSS + PPF combination", 
                "Family health insurance premium"
            ]
            
        else:
            career_stage = "Pre-Retirement Focus"
            tips = [
                "Max out 80C investments",
                "Consider Senior Citizen Saving Scheme",
                "NPS additional tax benefits under 80CCD(1B)"
            ]
        
        # Add career stage tips
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"{career_stage}", 0, 1)
        
        for tip in tips:
            pdf.set_font('Arial', '', 11)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(5, 6, "*", 0, 0)
            pdf.multi_cell(0, 6, sanitize_text(tip))
        
        # Additional options
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(24, 136, 229)
        pdf.cell(0, 10, "Additional Tax Saving Options:", 0, 1)
        
        additional_options = [
            "Health Insurance (80D): Up to Rs.25,000 deduction",
            "NPS Additional: Extra Rs.50,000 deduction",
            "Home Loan Interest (24b): Up to Rs.2L"
        ]
        
        # Add additional options if high income
        if input_data.get('AnnualIncome', 0) > 1000000:
            additional_options.append("Consider old vs new tax regime comparison")
            
        for option in additional_options:
            pdf.set_font('Arial', '', 11)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(5, 6, "*", 0, 0)
            pdf.multi_cell(0, 6, sanitize_text(option))
        
        # Add small disclaimer at bottom of the page instead of a footer
        pdf.ln(10)  # Add some space
        pdf.set_font("Arial", "I", 8)  # Italic, small font
        pdf.set_text_color(100, 100, 100)  # Gray color
        pdf.multi_cell(0, 5, "Note: This report is for informational purposes only and should not be considered as tax advice.", 0, 'C')
        
        # Save to a temporary file
        temp_file = "tax_planning_report.pdf"
        pdf.output(temp_file)
        
        # Convert to base64 for download
        with open(temp_file, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")
            
        # Remove temporary file
        os.remove(temp_file)
        
        return pdf_data
        
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        return None


def generate_chat_report(messages: List[Dict[str, str]]) -> str:
    """Generate a PDF report of the chat conversation"""
    try:
        # Create PDF instance
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(24, 136, 229)  # Blue color
        pdf.cell(0, 10, "Your Tax Planning Conversation", 0, 1, 'C')
        
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(100, 100, 100)  # Gray color
        pdf.cell(0, 5, f"Generated on: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}", 0, 1, 'C')
        pdf.ln(10)
        
        # Intro text
        pdf.set_font("Arial", "", 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 6, "This report contains your conversation with the TaxGPT Advisor.", 0, 1)
        pdf.ln(5)
        
        # Add each message
        for msg in messages:
            role = msg.get("role", "")
            content = sanitize_text(msg.get("content", ""))
            
            if role == "user":
                pdf.set_fill_color(230, 243, 255)  # Light blue
                pdf.set_font("Arial", "B", 10)
                pdf.cell(0, 6, "You:", 0, 1, 'L', True)
            else:
                pdf.set_fill_color(240, 242, 246)  # Light gray
                pdf.set_font("Arial", "B", 10)
                pdf.cell(0, 6, "TaxGPT Advisor:", 0, 1, 'L', True)
                
            pdf.set_font("Arial", "", 10)
            
            # Handle multi-line content
            lines = content.split('\n')
            for line in lines:
                if line.strip():  # Skip empty lines
                    # Process line in chunks to avoid potential encoding issues
                    chunk_size = 80  # Adjust as needed
                    for i in range(0, len(line), chunk_size):
                        chunk = line[i:i+chunk_size]
                        # Extra sanitization step for chunks
                        safe_chunk = ''.join(c for c in chunk if ord(c) < 128)
                        pdf.multi_cell(0, 5, safe_chunk)
            pdf.ln(3)
        
        # Add small disclaimer at bottom of the page instead of a footer
        pdf.ln(10)  # Add some space
        pdf.set_font("Arial", "I", 8)  # Italic, small font
        pdf.set_text_color(100, 100, 100)  # Gray color
        pdf.multi_cell(0, 5, "Note: This report is for informational purposes only and should not be considered as tax advice.", 0, 'C')
        
        # Save to a temporary file
        temp_file = "tax_chat_report.pdf"
        pdf.output(temp_file)
        
        # Convert to base64 for download
        with open(temp_file, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")
            
        # Remove temporary file
        os.remove(temp_file)
        
        return pdf_data
        
    except Exception as e:
        st.error(f"Error generating chat report: {str(e)}")
        return None