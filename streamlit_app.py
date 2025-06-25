import streamlit as st
import tempfile
import os
from metadata_generator import generate_metadata

def display_metadata_section(section_title, section_dict):
    st.markdown(f"#### {section_title}")
    st.markdown(
        "<div style='background-color:#f8f9fa; border-radius:8px; padding:16px; border:1px solid #dee2e6; margin-bottom:16px;'>"
        + "".join(
            f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'>"
            f"<span style='font-weight:600; color:#495057;'>{k.replace('_', ' ').capitalize()}</span>"
            f"<span style='color:#212529;'>{v}</span>"
            f"</div>"
            for k, v in section_dict.items()
        )
        + "</div>",
        unsafe_allow_html=True,
    )

# Add a subtle, professional gradient background to the app
st.markdown('''
    <style>
    body {
        background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%) !important;
    }
    .stApp {
        background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%) !important;
    }
    </style>
''', unsafe_allow_html=True)

st.set_page_config(page_title="Automated Metadata Extraction", layout="centered")
st.title("Automated Metadata Extraction")
st.write("Upload a PDF, DOCX, TXT, or image file to extract structured metadata.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    metadata = generate_metadata(tmp_path, original_filename=uploaded_file.name)
    os.unlink(tmp_path)
    if 'error' in metadata:
        st.error(metadata['error'])
    else:
        display_metadata_section("Basic Info", metadata['basic_info'])
        display_metadata_section("Content Analysis", metadata['content_analysis'])
        with st.expander("Semantic Data", expanded=True):
            st.markdown("**Summary:**")
            st.info(metadata['semantic_data']['summary'])
            st.markdown("**Key Topics:**")
            st.code(", ".join(metadata['semantic_data']['key_topics']))
            st.markdown("**Key Phrases:**")
            st.code(", ".join(metadata['semantic_data']['key_phrases']))
            st.markdown("**Entities:**")
            st.json(metadata['semantic_data']['entities'])
            st.markdown("**Extraction Log:**")
            st.code('\n'.join(metadata['semantic_data']['entraction log']))
            st.markdown("**Text Preview:**")
            st.code(metadata['semantic_data']['text preview'])