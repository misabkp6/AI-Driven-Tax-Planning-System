import google.generativeai as genai
import streamlit as st
import logging
from typing import Dict, Any, List
import json

class GeminiTaxChatbot:
    """Tax chatbot powered by Google's Gemini API"""
    
    def __init__(self, api_key: str):
        """Initialize the Gemini tax planning chatbot"""
        self.api_key = api_key
        self.user_data = {}
        self.tax_context = """
        You are TaxGPT, an AI tax planning specialist focused on Indian tax laws.
        You help users optimize their tax planning and provide guidance on tax-saving strategies.
        
        Key points:
        - Section 80C allows deductions up to ₹1.5 lakh for investments like PPF, ELSS, etc.
        - Section 80D allows deductions for health insurance premiums (₹25,000, ₹50,000 for seniors)
        - HRA exemption is available for salaried individuals paying rent
        - New tax regime has lower rates but fewer deductions
        - Old tax regime has higher rates but allows various deductions
        
        Indian tax slabs for FY 2023-24 (Old Regime):
        - Income up to ₹2.5 lakh: No tax
        - Income from ₹2.5 lakh to ₹5 lakh: 5% tax
        - Income from ₹5 lakh to ₹7.5 lakh: 10% tax
        - Income from ₹7.5 lakh to ₹10 lakh: 15% tax
        - Income from ₹10 lakh to ₹12.5 lakh: 20% tax
        - Income from ₹12.5 lakh to ₹15 lakh: 25% tax
        - Income above ₹15 lakh: 30% tax
        - Plus 4% health and education cess on the tax amount
        
        Indian tax slabs for FY 2023-24 (New Regime):
        - Income up to ₹3 lakh: No tax
        - Income from ₹3 lakh to ₹6 lakh: 5% tax
        - Income from ₹6 lakh to ₹9 lakh: 10% tax
        - Income from ₹9 lakh to ₹12 lakh: 15% tax
        - Income from ₹12 lakh to ₹15 lakh: 20% tax
        - Income above ₹15 lakh: 30% tax
        - Plus 4% health and education cess on the tax amount
        - No major deductions under Section 80C, 80D, etc. are available
        """
        
        # Configure API
        try:
            genai.configure(api_key=api_key)
            
            # Try to list available models first
            try:
                available_models = genai.list_models()
                model_names = [model.name.split('/')[-1] for model in available_models]
                logging.info(f"Available models: {model_names}")
                
                # Try to find a suitable Gemini model
                gemini_models = [name for name in model_names if 'gemini' in name.lower()]
                
                if 'gemini-1.5-pro' in gemini_models:
                    self.model = genai.GenerativeModel('gemini-1.5-pro')
                    logging.info("Using gemini-1.5-pro model")
                elif 'gemini-pro' in gemini_models:
                    self.model = genai.GenerativeModel('gemini-pro')
                    logging.info("Using gemini-pro model")
                elif gemini_models:
                    self.model = genai.GenerativeModel(gemini_models[0])
                    logging.info(f"Using available model: {gemini_models[0]}")
                else:
                    # Default fallback
                    self.model = genai.GenerativeModel('gemini-1.5-pro')
                    logging.info("No Gemini models found, defaulting to gemini-1.5-pro")
            
            except Exception as e:
                logging.warning(f"Could not list models: {str(e)}. Trying default model.")
                self.model = genai.GenerativeModel('gemini-1.5-pro')  # Updated to latest model
            
            # Initialize chat history
            self.chat = self.model.start_chat(history=[])
            self.add_system_message()
            
            logging.info("Gemini API initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Gemini API: {str(e)}")
            st.error(f"Failed to initialize chatbot: {str(e)}")
    
    def add_system_message(self):
        """Add system message with context to the chat"""
        user_info = ""
        if self.user_data:
            user_info = f"""
            USER'S FINANCIAL PROFILE:
            - Annual Income: ₹{self.user_data.get('AnnualIncome', 0):,.2f}
            - Age: {self.user_data.get('Age', 0)} years old
            - 80C Investments: ₹{self.user_data.get('Investments', 0):,.2f} (max allowed: ₹150,000)
            - Other Deductions: ₹{self.user_data.get('Deductions', 0):,.2f}
            - HRA: ₹{self.user_data.get('HRA', 0):,.2f}
            - Employment: {self.user_data.get('EmploymentType', 'Not specified')}
            - Tax Liability: ₹{self.user_data.get('TaxLiability', 0):,.2f}
            """
            
        system_message = f"""
        {self.tax_context}
        
        {user_info}
        
        RESPONSE GUIDELINES:
        - Always give accurate tax advice based on Indian tax laws
        - If the user has already maxed out 80C investments (₹150,000), suggest alternatives beyond 80C
        - For users under 30, focus on growth-oriented tax-saving options
        - For users over 60, mention senior citizen specific benefits
        - Keep responses concise, friendly and focused on practical advice
        - If unsure about any information, clearly state so rather than providing incorrect information
        - Format your responses in a readable way with bullet points for lists
        - Use emojis sparingly to make responses engaging
        """
        
        try:
            # Add system message to chat history
            self.chat.history.append({
                'role': 'user',
                'parts': [system_message]
            })
            self.chat.history.append({
                'role': 'model',
                'parts': ["I understand my role as TaxGPT. I'll provide accurate tax planning advice based on Indian tax laws and the user's financial profile."]
            })
        except Exception as e:
            logging.error(f"Failed to add system message: {str(e)}")
    
    def set_user_data(self, user_data: Dict[str, Any]) -> None:
        """Store user data for personalized responses"""
        self.user_data = user_data
        logging.info(f"User data set: {user_data}")
        
        # Reset chat with new user data
        try:
            self.chat = self.model.start_chat(history=[])
            self.add_system_message()
        except Exception as e:
            logging.error(f"Error resetting chat with user data: {str(e)}")
    
    def get_response(self, query: str) -> str:
        """Process user query and generate a response using Gemini"""
        if not self.api_key:
            return "API key not configured. Please set up your Gemini API key."
            
        try:
            # Check if query is about an unsupported topic
            if self._is_off_topic(query):
                return "I'm focused on tax planning and financial advice. Please ask me about tax saving strategies, deductions, or financial planning."
            
            # Enhance the user query with contextual information
            enhanced_query = self._enhance_query(query)
            
            # Get response from Gemini
            response = self.chat.send_message(enhanced_query)
            return response.text
            
        except Exception as e:
            logging.error(f"Error getting response from Gemini: {str(e)}")
            return f"I'm having trouble processing your request. Please try again or rephrase your question. Technical details: {str(e)}"
    
    def _is_off_topic(self, query: str) -> bool:
        """Check if query is unrelated to tax or financial advice"""
        off_topic_keywords = [
            'politics', 'religion', 'dating', 'illegal', 'hack', 'crack', 
            'pornography', 'betting', 'gambling', 'weapons'
        ]
        return any(keyword in query.lower() for keyword in off_topic_keywords)
    
    def _enhance_query(self, query: str) -> str:
        """Enhance user query with contextual information"""
        # If query is about 80C and user has maxed it
        if ('80c' in query.lower() or 'investment' in query.lower()) and self.user_data.get('Investments', 0) >= 150000:
            return f"{query} (Note: I've already invested ₹150,000 under 80C, which is the maximum limit)"
        
        # If query is about tax slabs, provide current information
        if 'tax slab' in query.lower() or 'tax bracket' in query.lower():
            return f"{query} (Please provide details for both old and new tax regimes in India for FY 2023-24)"
        
        # If query mentions age-specific planning
        age = self.user_data.get('Age', 0)
        if 'retirement' in query.lower() or 'senior' in query.lower():
            return f"{query} (Note: My current age is {age})"
            
        return query
    
    def get_suggestions(self) -> List[str]:
        """Return conversation suggestions based on user profile"""
        suggestions = [
            "How can I reduce my tax liability?",
            "Explain the difference between old and new tax regimes",
            "What are the current tax slabs in India?",
            "What tax benefits can I get from home loans?"
        ]
        
        # Add personalized suggestions based on user data
        if self.user_data:
            age = self.user_data.get('Age', 30)
            income = self.user_data.get('AnnualIncome', 500000)
            investments = self.user_data.get('Investments', 0)
            
            if investments >= 150000:
                suggestions.append("I've maxed out 80C. What other options do I have?")
            else:
                remaining = 150000 - investments
                suggestions.append(f"How should I invest my remaining ₹{remaining:,.2f} under 80C?")
                
            if age > 50:
                suggestions.append("What tax planning should I do for retirement?")
                
            if income > 1000000:
                suggestions.append("Tax saving strategies for high income earners")
                
        return suggestions