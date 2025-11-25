import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer.manager import TokenizerManager
from src.ui.modes.single import render_single_mode
from src.ui.modes.chat import render_chat_mode
from src.ui.modes.jsonl import render_jsonl_mode


st.set_page_config(
    page_title="Tokenizer Visualizer",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.title("✂️ Tokenizer Visualizer")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        
        # Initialize Repository
        # src/app.py is in src/, so project root is one level up
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cache_dir = os.path.join(project_root, ".cache")
        from src.tokenizer.repository import TokenizerRepository
        repo = TokenizerRepository(cache_dir)
        
        # Mode Selection
        mode = st.radio("Mode", ["Single Prompt", "Chat", "JSONL"], index=0)
        
        # Metric Unit Selection
        metric_unit = st.radio("Metric Unit", ["Character", "Word"], index=0)
        st.session_state.metric_unit = metric_unit
        
        st.markdown("---")
        
        # Unified Tokenizer Selection
        st.markdown("### Select Tokenizers")
        
        # 1. Tiktoken Options
        tiktoken_options = ["gpt-5", "gpt-5.1", "gpt-4.1", "gpt-4o", "o1"]
        
        # 2. Local/HF Options
        available_models = repo.get_available_models()
        
        # Build display map
        display_map = {}
        
        for name in tiktoken_options:
            display_name = f"tiktoken: {name}"
            display_map[display_name] = {"name": name, "source": "tiktoken", "display_name": display_name}
            
        for display_key, model_path in available_models.items():
            # display_key is already formatted nicely
            display_map[display_key] = {"name": model_path, "source": "huggingface", "display_name": display_key}

        all_options = list(display_map.keys())
        
        # Default selection
        default_selections = [all_options[0]] if all_options else None
        
        selected_options = st.multiselect(
            "Choose tokenizers (select multiple to compare)", 
            all_options, 
            default=default_selections
        )
        
        selected_models_data = [display_map[opt] for opt in selected_options]
        
        # Add New Tokenizer Section
        st.markdown("---")
        with st.expander("Add New Tokenizer"):
            tab1, tab2 = st.tabs(["Download (HF)", "Upload (JSON)"])
            
            with tab1:
                repo_id = st.text_input("HF Repo ID", value="openai/gpt-oss-20b", help="e.g. openai/gpt-oss-20b")
                if st.button("Download"):
                    with st.spinner(f"Downloading {repo_id}..."):
                        try:
                            path = repo.download_model(repo_id)
                            st.success(f"Downloaded {repo_id}!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {e}")
                            
            with tab2:
                st.markdown("### Upload Local Tokenizer")
                st.info("Upload all tokenizer files (e.g., `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `tokenizer.model`).")
                
                model_name_input = st.text_input("Model Name (Required)", help="Name for the directory to store these files")
                uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, key="uploader")
                
                if uploaded_files and model_name_input:
                    if st.button("Save Tokenizer"):
                        try:
                            path = repo.save_uploaded_model_batch(uploaded_files, model_name_input)
                            st.success(f"Saved tokenizer to {path}!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to save: {e}")
                elif uploaded_files and not model_name_input:
                    st.warning("Please enter a Model Name.")

        st.markdown("---")
        st.caption("Powered by `tiktoken` & `transformers`")

    # --- Main Content ---
    manager = TokenizerManager.get_instance()
    
    try:
        if mode == "Single Prompt":
            render_single_mode(manager, selected_models_data)
        elif mode == "Chat":
            render_chat_mode(manager, selected_models_data)
        elif mode == "JSONL":
            if selected_models_data:
                # JSONL mode currently supports only one tokenizer
                if len(selected_models_data) > 1:
                    st.warning("JSONL mode currently uses the first selected tokenizer.")
                render_jsonl_mode(manager, selected_models_data[0]["name"], selected_models_data[0]["source"])
            else:
                st.warning("Please select a tokenizer.")
            
    except Exception as e:
        st.error(f"Error loading tokenizer or processing: {e}")

if __name__ == "__main__":
    main()

