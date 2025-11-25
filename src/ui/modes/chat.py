import streamlit as st
from src.tokenizer.manager import TokenizerManager
from src.ui.components import render_tokenizer_result

def render_chat_mode(tokenizer_manager: TokenizerManager, models_data: list[dict]):
    st.header("Chat Mode")
    
    if not models_data:
        st.warning("Please select at least one tokenizer.")
        return
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
    
    # Add new message button
    if st.button("Add Message"):
        st.session_state.chat_messages.append({"role": "user", "content": ""})
    
    # Clear messages
    if st.button("Clear All"):
        st.session_state.chat_messages = []
        st.rerun()

    updated_messages = []
    
    for i, msg in enumerate(st.session_state.chat_messages):
        with st.container():
            st.markdown(f"**Message {i+1}**")
            col1, col2 = st.columns([1, 4])
            
            with col1:
                role = st.selectbox(f"Role ##{i}", ["system", "user", "assistant"], index=["system", "user", "assistant"].index(msg["role"]) if msg["role"] in ["system", "user", "assistant"] else 1, key=f"role_{i}")
            
            with col2:
                content = st.text_area(f"Content ##{i}", value=msg["content"], key=f"content_{i}", height=100)
            
            # Update state
            msg["role"] = role
            msg["content"] = content
            updated_messages.append(msg)
            
            # Tokenize and visualize
            if content:
                with st.expander("Tokenization", expanded=True):
                    if len(models_data) == 1:
                        render_tokenizer_result(tokenizer_manager, models_data[0], content, show_header=False)
                    elif len(models_data) == 2:
                        cols = st.columns(2)
                        for idx, model_info in enumerate(models_data):
                            with cols[idx]:
                                render_tokenizer_result(tokenizer_manager, model_info, content, show_header=True)
                    else:
                        for model_info in models_data:
                            render_tokenizer_result(tokenizer_manager, model_info, content, show_header=True)
                            st.markdown("---")
            
            st.markdown("---")
            
    st.session_state.chat_messages = updated_messages

