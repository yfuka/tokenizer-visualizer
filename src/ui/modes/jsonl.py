import streamlit as st
import pandas as pd
import json
from src.tokenizer.manager import TokenizerManager
from src.ui.components import render_token_chips, render_metrics

def render_jsonl_mode(tokenizer_manager: TokenizerManager, model_name: str, source: str):
    st.header("JSONL Mode")
    
    uploaded_file = st.file_uploader("Upload .jsonl file", type=["jsonl"])
    
    if uploaded_file:
        try:
            # Read JSONL
            df = pd.read_json(uploaded_file, lines=True)
            st.success(f"Loaded {len(df)} rows.")
            
            # Preview
            with st.expander("Data Preview", expanded=False):
                st.dataframe(df.head())
            
            # Row Selection / Filtering
            st.subheader("Row Selection")
            row_mode = st.radio("Selection Mode", ["All Rows", "First N Rows", "Range"], horizontal=True)
            
            selected_df = df
            if row_mode == "First N Rows":
                n = st.number_input("N", min_value=1, max_value=len(df), value=min(10, len(df)))
                selected_df = df.head(n)
            elif row_mode == "Range":
                start, end = st.slider("Range", 0, len(df), (0, min(10, len(df))))
                selected_df = df.iloc[start:end]
            
            st.info(f"Processing {len(selected_df)} rows.")
            
            if st.button("Analyze Tokens"):
                tokenizer = tokenizer_manager.get_tokenizer(model_name, source)
                
                results = []
                total_tokens = 0
                total_count = 0 # Char or Word count
                
                metric_unit = st.session_state.get("metric_unit", "Character")
                count_label = "Word Count" if metric_unit == "Word" else "Char Count"
                
                progress_bar = st.progress(0)
                
                for i, (index, row) in enumerate(selected_df.iterrows()):
                    # Determine text content for summary metrics
                    full_text = ""
                    if "text" in row and pd.notna(row["text"]):
                        full_text = str(row["text"])
                    elif "prompt" in row and "response" in row and pd.notna(row["prompt"]) and pd.notna(row["response"]):
                        full_text = str(row["prompt"]) + "\n" + str(row["response"])
                    elif "messages" in row and isinstance(row["messages"], list):
                        # Approximate text for messages
                        full_text = "\n".join([str(m.get("content", "")) for m in row["messages"] if isinstance(m, dict)])
                    
                    # If format is unknown, full_text remains empty.
                    
                    token_result = tokenizer.encode(full_text)
                    
                    if metric_unit == "Word":
                        current_count = len(full_text.split())
                    else:
                        current_count = len(full_text)
                    
                    results.append({
                        "Index": index,
                        "Text Preview": full_text[:50] + "..." if len(full_text) > 50 else full_text,
                        "Token Count": token_result.count,
                        count_label: current_count
                    })
                    
                    total_tokens += token_result.count
                    total_count += current_count
                    
                    progress_bar.progress((i + 1) / len(selected_df))
                
                results_df = pd.DataFrame(results)
                
                # Store in session state
                st.session_state["jsonl_results"] = results_df
                st.session_state["jsonl_total_tokens"] = total_tokens
                st.session_state["jsonl_total_count"] = total_count
                st.session_state["jsonl_count_label"] = count_label
                st.session_state["jsonl_metric_unit"] = metric_unit

            # Display Results if available
            if "jsonl_results" in st.session_state:
                results_df = st.session_state["jsonl_results"]
                total_tokens = st.session_state["jsonl_total_tokens"]
                total_count = st.session_state["jsonl_total_count"]
                count_label = st.session_state["jsonl_count_label"]
                metric_unit = st.session_state.get("jsonl_metric_unit", "Character")

                st.subheader("Analysis Results")
                
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Total Token Count", total_tokens)
                with cols[1]:
                    st.metric(f"Total {count_label}", total_count)
                with cols[2]:
                        avg_label = "Avg. Words/Token" if metric_unit == "Word" else "Avg. Chars/Token"
                        st.metric(avg_label, f"{total_count / total_tokens:.2f}" if total_tokens > 0 else "0")

                st.dataframe(results_df)
                
                # Detailed Visualization for a specific row
                st.subheader("Detailed Visualization")
                
                # Ensure the index is in the dataframe
                available_indices = results_df["Index"].tolist()
                selected_row_idx = st.selectbox("Select row to visualize", available_indices)
                
                if selected_row_idx is not None:
                    # We need to fetch the original row data. 
                    # Since 'df' might have changed if file was re-uploaded, we should be careful.
                    # But assuming standard flow, 'df' is still valid.
                    if selected_row_idx in df.index:
                        row_data = df.loc[selected_row_idx]
                        st.json(row_data.to_dict(), expanded=False)
                        
                        # Re-instantiate tokenizer for visualization (or get from manager)
                        # Note: We need to ensure we use the same tokenizer as analysis
                        tokenizer = tokenizer_manager.get_tokenizer(model_name, source)

                        # Detect format
                        from src.utils.jsonl_parser import detect_format, FORMAT_MESSAGES, FORMAT_PROMPT_RESPONSE, FORMAT_TEXT
                        
                        fmt = detect_format(row_data)
                        
                        # 1. Messages Format
                        if fmt == FORMAT_MESSAGES:
                            st.markdown("### Messages Format Detected")
                            for msg in row_data["messages"]:
                                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                                    role = msg["role"]
                                    content = msg["content"]
                                    st.markdown(f"**{role.capitalize()}**")
                                    
                                    res = tokenizer.encode(str(content))
                                    render_metrics(res.count, str(content))
                                    render_token_chips(res.grouped_tokens)
                                    st.markdown("---")
                                    
                        # 2. Prompt/Response Format
                        elif fmt == FORMAT_PROMPT_RESPONSE:
                            st.markdown("### Prompt/Response Format Detected")
                            
                            st.markdown("**Prompt**")
                            res_prompt = tokenizer.encode(str(row_data["prompt"]))
                            render_metrics(res_prompt.count, str(row_data["prompt"]))
                            render_token_chips(res_prompt.grouped_tokens)
                            
                            st.markdown("**Response**")
                            res_response = tokenizer.encode(str(row_data["response"]))
                            render_metrics(res_response.count, str(row_data["response"]))
                            render_token_chips(res_response.grouped_tokens)
                            
                        # 3. Text Format
                        elif fmt == FORMAT_TEXT:
                            st.markdown("### Text Format Detected")
                            content = str(row_data["text"])
                            res = tokenizer.encode(content)
                            render_metrics(res.count, content)
                            render_token_chips(res.grouped_tokens)

                        # 4. Unknown Format
                        else:
                            st.warning("Unknown format. Could not visualize tokens.")
                    else:
                        st.warning("Selected index not found in current dataframe.")

        except ValueError as e:
            st.error(f"Error parsing JSONL: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
