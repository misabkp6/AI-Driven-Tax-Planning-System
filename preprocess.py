import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
from chat_gemini import render_gemini_chat_page
from report_generator import generate_tax_report, generate_chat_report
from predict import predict_tax_strategy

# Set page config
st.set_page_config(
    page_title="AI Tax Planner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #424242;
    }
    .strategy-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-bottom: 10px;
    }
    .strategy-description {
        font-style: italic;
        color: #424242;
        margin-bottom: 15px;
    }
    .recommendation-item {
        margin-bottom: 8px;
        padding-left: 15px;
        border-left: 3px solid #1E88E5;
    }
    .category-header {
        font-weight: bold; 
        color: #1E88E5; 
        margin-top: 15px; 
        margin-bottom: 10px; 
        padding: 5px; 
        border-bottom: 1px solid #1E88E5;
    }
    .tab-content {
        padding-top: 1rem;
    }
    /* Chat styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #e6f3ff;
    }
    .chat-message.bot {
        background-color: #f0f2f6;
    }
    .chat-suggestion {
        cursor: pointer;
        border: 1px solid #1E88E5;
        border-radius: 4px;
        padding: 0.5rem;
        margin: 0.25rem;
        text-align: center;
        transition: all 0.3s;
    }
    .chat-suggestion:hover {
        background-color: #e6f3ff;
    }
    .download-btn {
        background-color: #1E88E5;
        color: white;
        padding: 0.75rem 1.5rem;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 4px;
        cursor: pointer;
    }
    .ml-badge {
        background-color: #2E7D32;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        margin-left: 5px;
    }
    .rule-badge {
        background-color: #FF9800;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        margin-left: 5px;
    }
    .hybrid-badge {
        background-color: #2196F3;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        margin-left: 5px;
    }
    .ml-section {
        background-color: #f0f8ff;
        border-left: 4px solid #1E88E5;
        padding: 15px;
        margin: 15px 0;
        border-radius: 4px;
    }
    .api-status {
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        margin-top: 5px;
    }
    .status-indicator {
        height: 8px;
        width: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    .online {
        background-color: #4CAF50;
    }
    .offline {
        background-color: #9E9E9E;
    }
    .confidence-bar {
        margin-top: 10px;
        margin-bottom: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        overflow: hidden;
        padding: 4px;
    }
    .confidence-label {
        font-size: 0.9rem;
        color: #424242;
        margin-bottom: 4px;
    }
    .confidence-value {
        height: 10px;
        background-color: #1E88E5;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# API URL - change this to the actual API URL when deployed
API_URL = "http://localhost:8000/predict"

def format_currency(value: float) -> str:
    """Format value as INR currency"""
    return f"₹{value:,.2f}"

def calculate_tax_liability(data: Dict[str, Any]) -> float:
    """Calculate tax liability based on input data"""
    annual_income = data.get("AnnualIncome", 0)
    investments = data.get("Investments", 0)
    deductions = data.get("Deductions", 0)
    hra = data.get("HRA", 0)
    
    # Calculate taxable income
    taxable_income = max(0, annual_income - investments - deductions - hra)
    
    # Simple progressive tax calculation (modify as per Indian tax rules)
    if taxable_income <= 250000:
        tax = 0
    elif taxable_income <= 500000:
        tax = (taxable_income - 250000) * 0.05
    elif taxable_income <= 750000:
        tax = 12500 + (taxable_income - 500000) * 0.10
    elif taxable_income <= 1000000:
        tax = 37500 + (taxable_income - 750000) * 0.15
    elif taxable_income <= 1250000:
        tax = 75000 + (taxable_income - 1000000) * 0.20
    elif taxable_income <= 1500000:
        tax = 125000 + (taxable_income - 1250000) * 0.25
    else:
        tax = 187500 + (taxable_income - 1500000) * 0.30
        
    # Add cess (4%)
    tax = tax * 1.04
    
    return tax

def get_tax_prediction(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call the API to get tax prediction and strategy"""
    try:
        # First calculate tax liability
        tax_liability = calculate_tax_liability(input_data)
        input_data["TaxLiability"] = tax_liability
        
        # Calculate tax savings
        standard_tax = input_data["AnnualIncome"] * 0.3 * 1.04  # 30% + 4% cess
        input_data["TaxSavings"] = max(0, standard_tax - tax_liability)
        
        # Try API call first, but silently fall back if it fails
        api_available = False
        try:
            # Attempt API connection with a short timeout to prevent long waits
            response = requests.post(API_URL, json=input_data, timeout=2)
            if response.status_code == 200:
                api_available = True
                # Store API status in session state
                st.session_state["api_available"] = True
                return response.json()
        except:
            # Silently continue to local prediction if API fails
            pass
            
        # Store API status in session state
        st.session_state["api_available"] = api_available
        
        # Local ML model prediction
        return predict_tax_strategy(input_data)
        
    except Exception as e:
        st.error(f"Error getting prediction: {str(e)}")
        return {
            "tax_liability": tax_liability,
            "monthly_tax": tax_liability / 12,
            "effective_rate": (tax_liability / input_data["AnnualIncome"]) * 100 if input_data["AnnualIncome"] > 0 else 0,
            "tax_savings": input_data.get("TaxSavings", 0),
            "recommended_strategy": "Error generating strategy",
            "strategy_description": "Could not determine optimal strategy",
            "recommendations": ["Please try again with complete information"],
            "potential_savings": "Unknown",
            "error": str(e),
            "prediction_method": "Error"
        }

def is_api_available():
    """Check if the API is available"""
    try:
        # Try to connect to API health endpoint with a very short timeout
        health_url = API_URL.replace("/predict", "/health")
        response = requests.get(health_url, timeout=1)
        return response.status_code == 200
    except:
        return False

def create_tax_breakdown_chart(annual_income: float, tax_liability: float) -> go.Figure:
    """Create a chart showing tax vs take-home breakdown"""
    take_home = annual_income - tax_liability
    
    labels = ['Tax', 'Take-home']
    values = [tax_liability, take_home]
    colors = ['#EF5350', '#66BB6A']
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker_colors=colors
        )
    ])
    
    fig.update_layout(
        title="Income Breakdown",
        height=300,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    
    return fig

def create_tax_progression_chart(income: float, investments: float) -> go.Figure:
    """Create a chart showing tax progression at different income levels"""
    base_data = {"Investments": investments, "Deductions": 0, "HRA": 0}
    
    # Generate income range around the current income
    lower_bound = max(100000, income * 0.5)
    upper_bound = income * 1.5
    incomes = np.linspace(lower_bound, upper_bound, 10)
    
    taxes = []
    effective_rates = []
    
    for inc in incomes:
        data = {**base_data, "AnnualIncome": inc}
        tax = calculate_tax_liability(data)
        taxes.append(tax)
        effective_rates.append((tax / inc) * 100 if inc > 0 else 0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=incomes,
        y=taxes,
        mode='lines+markers',
        name='Tax Liability',
        line=dict(color='#EF5350', width=3),
        hovertemplate='Income: ₹%{x:,.2f}<br>Tax: ₹%{y:,.2f}'
    ))
    
    # Add current income marker
    current_tax = calculate_tax_liability({**base_data, "AnnualIncome": income})
    fig.add_trace(go.Scatter(
        x=[income],
        y=[current_tax],
        mode='markers',
        marker=dict(color='#1E88E5', size=12, line=dict(color='white', width=2)),
        name='Current',
        hovertemplate='Your Income: ₹%{x:,.2f}<br>Your Tax: ₹%{y:,.2f}'
    ))
    
    fig.update_layout(
        title="Tax Progression",
        xaxis_title="Annual Income (₹)",
        yaxis_title="Tax Liability (₹)",
        height=300,
        margin=dict(t=30, b=30, l=0, r=0),
        legend=dict(orientation="h", y=1.1)
    )
    
    return fig
    
def create_feature_importance_chart(feature_importance=None) -> go.Figure:
    """Create a chart showing ML model feature importance"""
    if feature_importance:
        # Use the actual feature importance if available
        features = [item[0] for item in feature_importance]
        importances = [item[1] for item in feature_importance]
    else:
        # Use placeholder data if not available
        features = ['Income Level', 'Age', 'Investment Ratio', 'Tax Bracket', 'Deduction Utilization']
        importances = [0.35, 0.25, 0.20, 0.15, 0.05]
    
    feature_fig = px.bar(
        x=importances, y=features, 
        orientation='h',
        labels={'x': 'Importance', 'y': 'Feature'},
        title="Features That Influenced ML Recommendation",
        color=importances,
        color_continuous_scale='Blues'
    )
    
    feature_fig.update_layout(height=300)
    return feature_fig

def main():
    st.markdown('<h1 class="main-header">AI Tax Planner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Optimize your tax planning with AI-powered recommendations</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Your Financial Information")
        
        annual_income = st.number_input("Annual Income (₹)", min_value=0, value=820000, step=10000)
        
        st.subheader("Tax-saving Investments")
        investments = st.number_input("Section 80C Investments (₹)", min_value=0, max_value=150000, value=150000, 
                                    help="PPF, ELSS, Insurance Premium, etc. (max ₹1,50,000)")
        
        deductions = st.number_input("Other Deductions (₹)", min_value=0, value=50000,
                                   help="Medical insurance, education loan interest, donations, etc.")
        
        hra = st.number_input("HRA (₹)", min_value=0, value=120000,
                            help="House Rent Allowance")
        
        st.subheader("Personal Details")
        age = st.slider("Age", 18, 100, 35)
        
        employment_type = st.selectbox("Employment Type", 
                                     ["Private", "Government", "Self-Employed", "Business", "Retired"])
        
        marital_status = st.selectbox("Marital Status", 
                                    ["Single", "Married"])
        
        calculate_button = st.button("Calculate & Get AI Recommendation", use_container_width=True)
        
        # API status indicator at bottom of sidebar
        st.markdown("---")
        api_available = is_api_available()
        
        st.markdown(
            f"""
            <div class="api-status">
                <strong>ML Engine:</strong>&nbsp;
                <span class="status-indicator {'online' if api_available else 'offline'}"></span>
                <span>{'Cloud' if api_available else 'Local'}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["📊 Tax Analysis", "💬 TaxGPT Advisor"])
    
    # Tab 1: Tax Analysis
    with tab1:
        # Main content
        if calculate_button or "tax_results" in st.session_state:
            # Prepare input data
            input_data = {
                "AnnualIncome": annual_income,
                "Investments": investments,
                "Deductions": deductions,
                "HRA": hra,
                "Age": age,
                "EmploymentType": employment_type,
                "MaritalStatus": marital_status
            }
            
            # Get prediction
            with st.spinner("AI analyzing your tax profile..."):
                tax_results = get_tax_prediction(input_data)
            st.session_state["tax_results"] = tax_results
            st.session_state["input_data"] = input_data
        
        if "tax_results" in st.session_state:
            results = st.session_state["tax_results"]
            
            # Display header summary
            st.markdown("""
            <div style="background-color:#f0f8ff; padding:10px; border-radius:10px; margin-bottom:20px;">
                <h3 style="margin:0;">This system uses advanced ML to:</h3>
                <ul style="margin-top:5px;">
                    <li>Predict your optimal tax strategy using machine learning</li>
                    <li>Generate personalized recommendations tailored to your profile</li>
                    <li>Compare your situation with thousands of similar tax cases</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">✨ AI-Predicted Annual Tax</div>
                    <div class="metric-value">{}</div>
                </div>
                """.format(format_currency(results["tax_liability"])), unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">💸 Monthly Tax</div>
                    <div class="metric-value">{}</div>
                </div>
                """.format(format_currency(results["monthly_tax"])), unsafe_allow_html=True)
                
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">📊 Effective Rate</div>
                    <div class="metric-value">{:.2f}%</div>
                </div>
                """.format(results["effective_rate"]), unsafe_allow_html=True)
                
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">💰 Tax Savings</div>
                    <div class="metric-value">{}</div>
                </div>
                """.format(format_currency(results["tax_savings"])), unsafe_allow_html=True)
            
            # Strategy section with prediction method badge
            prediction_method = results.get("prediction_method", "Unknown")
            
            # Determine which badge to display
            if prediction_method == "ML":
                method_badge = """<span class="ml-badge">ML Model</span>"""
                badge_color = "#4CAF50"  # Green
            elif prediction_method == "Hybrid":
                method_badge = """<span class="hybrid-badge">Hybrid</span>"""
                badge_color = "#2196F3"  # Blue
            elif prediction_method == "Rule-Based":
                method_badge = """<span class="rule-badge">Rule-Based</span>"""
                badge_color = "#FF9800"  # Orange
            else:
                method_badge = ""
                badge_color = "#9E9E9E"  # Gray
                
            st.markdown(f'<h2>🧠 Strategy Recommendations {method_badge}</h2>', unsafe_allow_html=True)
            
            strategy_box = st.container()
            with strategy_box:
                if results.get("error"):
                    st.error(f"Suggested Strategy: Error generating strategy")
                    st.info("Please try again or contact support if the issue persists.")
                else:
                    st.markdown(f"""
                    <div class="strategy-header">Suggested Strategy: {results["recommended_strategy"]}</div>
                    <div class="strategy-description">{results["strategy_description"]}</div>
                    """, unsafe_allow_html=True)
                    
                    
                    # Display recommendations with category headers
                    for rec in results["recommendations"]:
                        if rec == "":  # Empty string is a separator
                            st.markdown("<br>", unsafe_allow_html=True)
                        elif rec.startswith("📋") or rec.startswith("💰") or rec.startswith("🛡️") or rec.startswith("🏠") or rec.startswith("🔮") or rec.startswith("✨"):
                            # This is a category header
                            st.markdown(f"""
                                <div class="category-header">
                                    {rec}
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            # This is a normal recommendation item
                            st.markdown(f'<div class="recommendation-item">{rec}</div>', unsafe_allow_html=True)
                    
                    st.info(f"Potential additional savings: {results['potential_savings']}")
            
            
            # Charts section
            st.markdown("## 📈 Tax Analysis")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                tax_breakdown = create_tax_breakdown_chart(
                    st.session_state["input_data"]["AnnualIncome"], 
                    results["tax_liability"]
                )
                st.plotly_chart(tax_breakdown, use_container_width=True)
                
            with chart_col2:
                tax_progression = create_tax_progression_chart(
                    st.session_state["input_data"]["AnnualIncome"],
                    st.session_state["input_data"]["Investments"]
                )
                st.plotly_chart(tax_progression, use_container_width=True)
            
            # Add report download section
            st.markdown("## 📄 Download Tax Report")

            if st.button("Generate PDF Report", key="generate_tax_pdf"):
                with st.spinner("Generating PDF report..."):
                    pdf_data = generate_tax_report(
                        st.session_state["input_data"], 
                        st.session_state["tax_results"]
                    )
                    if pdf_data:
                        st.success("Report generated successfully!")
                        st.markdown(
                            f'<a href="data:application/pdf;base64,{pdf_data}" download="tax_planning_report.pdf" class="download-btn">'
                            f'📥 Download Tax Planning Report (PDF)</a>', 
                            unsafe_allow_html=True
                        )
                    else:
                        st.error("Failed to generate report. Please try again.")
            
            # Additional insights
            st.markdown("## 🧾 Tax Planning Insights")
            
            income = st.session_state["input_data"]["AnnualIncome"]
            age = st.session_state["input_data"]["Age"]
            
            # Career stage-based insights
            if age < 30:
                career_stage = "Early-Career Focus"
                saving_amount = "₹5,000"
                tips = [
                    "Start small with ELSS mutual funds",
                    "Consider PPF for long-term tax benefits",
                    "Term insurance premium",
                ]
                remaining = "₹70,000" if st.session_state["input_data"]["Investments"] < 80000 else "₹0"
                
            elif age < 45:
                career_stage = "Mid-Career Focus"
                saving_amount = "₹7,500"
                tips = [
                    "Maximize home loan benefits (principal under 80C)",
                    "ELSS + PPF combination",
                    "Family health insurance premium"
                ]
                remaining = "₹50,000" if st.session_state["input_data"]["Investments"] < 100000 else "₹0"
                
            else:
                career_stage = "Pre-Retirement Focus"
                saving_amount = "₹10,000"
                tips = [
                    "Max out 80C investments",
                    "Consider Senior Citizen Saving Scheme",
                    "NPS additional tax benefits under 80CCD(1B)"
                ]
                remaining = "₹35,000" if st.session_state["input_data"]["Investments"] < 115000 else "₹0"
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🎯 {career_stage} (Potential tax saving: {saving_amount}):")
                for tip in tips:
                    st.markdown(f"- {tip}")
                st.markdown(f"• Can still invest {remaining} under 80C")
            
            with col2:
                st.markdown("### 💡 Additional Tax Saving Options:")
                st.markdown("• Health Insurance (80D): Up to ₹25,000 deduction")
                st.markdown("• NPS Additional: Extra ₹50,000 deduction")
                st.markdown("• Home Loan Interest (24b): Up to ₹2L")
                
                if income > 1000000:
                    st.markdown("• Consider old vs new tax regime comparison")
            
            # Add another download button at the bottom of the page for convenience
            st.markdown("---")
            st.markdown("### Want to save this tax plan for later?")
            col1, col2 = st.columns([2, 1])
            with col2:
                if st.button("Download Tax Report", key="download_tax_report_bottom", use_container_width=True):
                    with st.spinner("Generating your report..."):
                        pdf_data = generate_tax_report(
                            st.session_state["input_data"], 
                            st.session_state["tax_results"]
                        )
                        if pdf_data:
                            st.success("Report generated successfully!")
                            st.markdown(
                                f'<a href="data:application/pdf;base64,{pdf_data}" download="tax_planning_report.pdf" class="download-btn">'
                                f'📥 Download Tax Planning Report (PDF)</a>', 
                                unsafe_allow_html=True
                            )
                        else:
                            st.error("Failed to generate report. Please try again.")
            with col1:
                st.write("Your personalized tax plan includes all insights, recommendations, and strategies tailored to your financial profile.")
                
        else:
            st.info("👈 Enter your financial details in the sidebar and click 'Calculate & Get AI Recommendation' to begin.")
            st.image("https://img.freepik.com/free-vector/tax-concept-illustration_114360-1057.jpg", width=400)

    # Tab 2: Gemini Chat interface
    with tab2:
        render_gemini_chat_page()
        
        # Add chat report download section
        if "gemini_messages" in st.session_state and st.session_state.gemini_messages:
            st.markdown("---")
            st.markdown("## 📄 Download Chat Report")
            
            if st.button("Generate Chat Report", key="generate_chat_pdf"):
                with st.spinner("Generating chat report..."):
                    chat_pdf_data = generate_chat_report(st.session_state.gemini_messages)
                    if chat_pdf_data:
                        st.success("Chat report generated successfully!")
                        st.markdown(
                            f'<a href="data:application/pdf;base64,{chat_pdf_data}" download="tax_chat_report.pdf" class="download-btn">'
                            f'📥 Download Chat Report (PDF)</a>', 
                            unsafe_allow_html=True
                        )
                    else:
                        st.error("Failed to generate chat report. Please try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Choose the best way to save tax with our ML-powered recommendations*")

if __name__ == "__main__":
    main()