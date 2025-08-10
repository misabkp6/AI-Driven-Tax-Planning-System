import streamlit as st
from gemini import GeminiTaxChatbot
from streamlit_chat import message
import time

def render_gemini_chat_page(user_data=None):
    """Render the Gemini-powered tax chat interface"""
    st.title("💬 TaxGPT Advisor")
    
    # Check for API key
    if "GEMINI_API_KEY" not in st.secrets and "gemini_api_key" not in st.session_state:
        api_key = st.text_input("Enter your Gemini API Key", type="password", 
                              help="Get your key from https://ai.google.dev/")
        if api_key:
            st.session_state.gemini_api_key = api_key
            st.success("API Key saved! You can now chat with the tax advisor.")
            st.rerun()
        else:
            st.warning("Please enter your Gemini API key to continue.")
            st.info("Get your API key from https://ai.google.dev/")
            return
    
    # Get API key from secrets or session state
    api_key = st.secrets.get("GEMINI_API_KEY", st.session_state.get("gemini_api_key", ""))
    
    # Initialize chat history
    if "gemini_messages" not in st.session_state:
        st.session_state.gemini_messages = []
    
    # Initialize chatbot
    if "gemini_chatbot" not in st.session_state:
        st.session_state.gemini_chatbot = GeminiTaxChatbot(api_key)
        
        # If we have user data in session, pass it to the chatbot
        if "input_data" in st.session_state and "tax_results" in st.session_state:
            user_data = st.session_state["input_data"].copy()
            user_data["TaxLiability"] = st.session_state["tax_results"].get("tax_liability", 0)
            st.session_state.gemini_chatbot.set_user_data(user_data)
    
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        if not st.session_state.gemini_messages:
            st.info("👋 Hello! I'm your AI Tax Advisor. Ask me anything about tax planning and strategies!")
        else:
            for i, msg in enumerate(st.session_state.gemini_messages):
                if msg["role"] == "user":
                    message(msg["content"], is_user=True, key=f"user_msg_{i}")
                else:
                    message(msg["content"], key=f"bot_msg_{i}")
    
    # Display suggestions if no messages
    if not st.session_state.gemini_messages:
        st.markdown("#### Try asking me:")
        
        # Get suggestions from chatbot
        suggestions = st.session_state.gemini_chatbot.get_suggestions()
        cols = st.columns(len(suggestions))
        
        for i, suggestion in enumerate(suggestions):
            if cols[i].button(suggestion, key=f"gemini_suggestion_{i}"):
                st.session_state.gemini_messages.append({"role": "user", "content": suggestion})
                
                with st.spinner("Thinking..."):
                    response = st.session_state.gemini_chatbot.get_response(suggestion)
                
                st.session_state.gemini_messages.append({"role": "assistant", "content": response})
                st.rerun()

    # Input box for user query
    with st.container():
        user_input = st.chat_input("Ask your tax planning question...", key="gemini_chat_input")
        
        if user_input:
            # Add user message to chat
            st.session_state.gemini_messages.append({"role": "user", "content": user_input})
            
            # Get bot response with spinner
            with st.spinner("Generating response..."):
                bot_response = st.session_state.gemini_chatbot.get_response(user_input)
            
            # Add bot response to chat
            st.session_state.gemini_messages.append({"role": "assistant", "content": bot_response})
            
            st.rerun()
    
    # Reset button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Reset Chat", key="reset_gemini_chat"):
            st.session_state.gemini_messages = []
            
            # Reinitialize chatbot with existing user data
            if "input_data" in st.session_state and "tax_results" in st.session_state:
                user_data = st.session_state["input_data"].copy()
                user_data["TaxLiability"] = st.session_state["tax_results"].get("tax_liability", 0)
                st.session_state.gemini_chatbot = GeminiTaxChatbot(api_key)
                st.session_state.gemini_chatbot.set_user_data(user_data)
            
            st.rerun()