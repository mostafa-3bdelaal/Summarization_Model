import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/bart-large-cnn"

# Load model manually (بدون pipeline)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# Function to summarize a chunk
def summarize_text(text, max_length=130, min_length=30):
    inputs = tokenizer([text], return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length, min_length=min_length)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to split text into chunks (زي الكود عندك)
def generate_chunks(text, max_word=500):
    for punkt in ['.', '?', '!']:
        text = text.replace(punkt, punkt + '<eos>')
    sentences = text.split('<eos>')
    chunks = []
    current_chunk = []
    for sentence in sentences:
        words = sentence.strip().split()
        if len(current_chunk) + len(words) <= max_word:
            current_chunk.extend(words)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = words
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# Streamlit UI
st.title('Text Summarization App 📄')

if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

def clear_text():
    st.session_state.input_text = ""

st.text_area(
    "Write your article here:",
    value=st.session_state.input_text,
    height=180,
    key="input_text"
)

col1, col2 = st.columns(2)
with col1:
    if st.button('Summarize'):
        if st.session_state.input_text.strip() == "":
            st.warning('Please enter text')
        else:
            with st.spinner('Summarizing...'):
                chunks = generate_chunks(st.session_state.input_text)
                summary = " ".join([summarize_text(chunk) for chunk in chunks])
            st.subheader("Summary:")
            st.write(summary)
with col2:
    st.button("Clear", on_click=clear_text)
