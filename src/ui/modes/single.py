import streamlit as st
from src.tokenizer.manager import TokenizerManager
from src.ui.components import render_tokenizer_result

def render_single_mode(tokenizer_manager: TokenizerManager, models_data: list[dict]):
    st.header("Single Prompt Mode")
    
    if not models_data:
        st.warning("Please select at least one tokenizer.")
        return

    text = st.text_area("Enter text to tokenize:", height=200, placeholder="Type something here...")
    
    if text:
        st.markdown("---")
        
        # Comparison Logic
        if len(models_data) > 1:
            st.subheader("Comparison")
            
            # Side-by-side for 2 models
            if len(models_data) == 2:
                cols = st.columns(2)
                for idx, model_info in enumerate(models_data):
                    with cols[idx]:
                        render_tokenizer_result(tokenizer_manager, model_info, text)
            else:
                # Stacked for 3+
                for model_info in models_data:
                    render_tokenizer_result(tokenizer_manager, model_info, text)
                    st.markdown("---")
        else:
            # Single model view
            render_tokenizer_result(tokenizer_manager, models_data[0], text, show_header=False)

