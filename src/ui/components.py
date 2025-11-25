import streamlit as st
import html
from src.tokenizer.utils import string_to_color

def render_token_chips(grouped_tokens: list, tokenizer_manager=None):
    """
    Renders tokens as colored chips using HTML/CSS.
    Accepts grouped_tokens (list of TokenGroup objects).
    """
    # CSS for the chips
    css = """
    <style>
    .token-container {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        font-family: 'Consolas', 'Monaco', monospace;
        line-height: 1.5;
        align_items: center;
    }
    .token-chip {
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid rgba(0,0,0,0.1);
        color: #333;
        font-size: 0.9em;
        white-space: pre-wrap;
        display: inline-block;
        margin-bottom: 4px;
        height: 24px;
        line-height: 18px;
        box-sizing: border-box;
    }
    .token-chip:hover {
        filter: brightness(0.95);
        cursor: default;
    }
    
    /* Accordion styles */
    details.token-group {
        display: inline-block;
        margin-bottom: 4px;
        border: 1px solid rgba(0,0,0,0.2);
        border-radius: 4px;
        background-color: #f0f2f6;
        height: 24px;
        box-sizing: border-box;
    }
    
    details.token-group[open] {
        height: auto;
        display: block; /* When open, it might need to be block to show content properly, or keep inline-block but it will expand */
        width: 100%; /* Force full width when open? No, that breaks the flow of other chips */
        /* Actually, if we want it to stay in flow, keep inline-block. But content will push things around. */
    }
    
    summary.token-summary {
        padding: 2px 6px;
        cursor: pointer;
        font-size: 0.9em;
        font-weight: bold;
        list-style: none;
        background-color: #e0e2e6;
        border-radius: 4px;
        white-space: nowrap;
        height: 22px; /* 24px - 2px border */
        line-height: 18px;
        display: flex;
        align_items: center;
        box-sizing: border-box;
    }
    
    summary.token-summary::-webkit-details-marker {
        display: none;
    }
    
    .token-group-content {
        padding: 4px;
        display: flex;
        flex-direction: column;
        gap: 2px;
        background-color: #fff;
        border-radius: 0 0 4px 4px;
        border-top: 1px solid #eee;
        position: absolute; /* Float above */
        z-index: 10;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        min-width: 150px;
    }
    
    /* We need a wrapper for relative positioning if we use absolute content */
    .group-wrapper {
        position: relative;
        display: inline-block;
        height: 24px;
        margin-bottom: 4px;
    }
    
    .sub-token {
        font-size: 0.8em;
        color: #555;
        border-bottom: 1px solid #eee;
        padding: 2px 0;
    }
    </style>
    """
    
    # Start the container
    # We strip newlines to prevent Markdown issues
    html_content = css.replace('\n', '') + '<div class="token-container">'
    
    chips_html = ""
    for group in grouped_tokens:
        if group.is_split:
            summary_text = html.escape(group.text)
            bg_color = string_to_color(group.text)
            
            content_html = '<div class="token-group-content">'
            for i, (token, tid) in enumerate(zip(group.tokens, group.ids)):
                safe_token = html.escape(repr(token))
                content_html += f'<div class="sub-token">Part {i+1}: ID {tid} <code style="background:none;padding:0;color:#d63384;">{safe_token}</code></div>'
            content_html += '</div>'
            
            # Use a wrapper for positioning
            # Note: details[open] with absolute content is a trick.
            # If we want it to push content, we don't use absolute.
            # But if it pushes content, it breaks the grid.
            # Let's try absolute positioning for the dropdown part to act like a tooltip/menu.
            
            chips_html += f'''
            <div class="group-wrapper">
                <details class="token-group">
                    <summary class="token-summary" style="background-color: {bg_color};" title="{len(group.tokens)} tokens">{summary_text} <span style="opacity:0.6;font-size:0.8em;margin-left:2px;">({len(group.tokens)})</span></summary>
                    {content_html}
                </details>
            </div>
            '''.replace('\n', '').strip()
        else:
            token = group.tokens[0]
            token_id = group.ids[0]
            
            safe_token = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;")
            display_token = safe_token.replace("\n", "↵")
            bg_color = string_to_color(safe_token)
            
            safe_token_repr = html.escape(repr(token), quote=True)
            safe_title = f"ID: {token_id}&#10;Token: {safe_token_repr}"
            
            chips_html += f'<span class="token-chip" style="background-color: {bg_color};" title="{safe_title}">{display_token}</span>'
        
    html_content += chips_html + "</div>"
    
    st.markdown(html_content, unsafe_allow_html=True)

def render_metrics(token_count: int, text: str, model_max_tokens: int = None):
    """
    Renders metric cards for token count, character/word count, etc.
    """
    metric_unit = st.session_state.get("metric_unit", "Character")
    
    if metric_unit == "Word":
        count = len(text.split())
        label = "Word Count"
        avg_label = "Avg. Words/Token"
    else:
        count = len(text)
        label = "Character Count"
        avg_label = "Avg. Chars/Token"

    cols = st.columns(3)
    
    with cols[0]:
        st.metric("Token Count", token_count)
        
    with cols[1]:
        st.metric(label, count)
        
    with cols[2]:
        if model_max_tokens:
            usage = (token_count / model_max_tokens) * 100
            st.metric("Context Usage", f"{usage:.1f}%", help=f"Max Tokens: {model_max_tokens}")
        else:
            st.metric(avg_label, f"{count / token_count:.2f}" if token_count > 0 else "0")

def render_tokenizer_result(tokenizer_manager, model_info: dict, text: str, show_header: bool = True):
    """
    Renders the tokenization result for a single tokenizer.
    Reusable across Single, Chat, and Comparison modes.
    """
    name = model_info["name"]
    source = model_info["source"]
    display_name = model_info.get("display_name", f"{name} ({source})")
    
    if show_header:
        st.subheader(display_name)
    
    try:
        tokenizer = tokenizer_manager.get_tokenizer(name, source)
        result = tokenizer.encode(text)
        
        render_metrics(result.count, text)
        render_token_chips(result.grouped_tokens)
        
    except Exception as e:
        st.error(f"Error processing with {name}: {e}")

