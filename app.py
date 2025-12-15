# text summarization

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Load model
@st.cache_resource  # يحفظ الموديل في الذاكرة ولا يعيد تحميله كل مرة
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(r"C:\2_ projects\NLP\summarization text\summarization_model")
    tokenizer = AutoTokenizer.from_pretrained(r"C:\2_ projects\NLP\summarization text\summarization_model")
    return pipeline(
        'summarization',
        model=model,
        tokenizer=tokenizer,
        framework='pt'
    )

model = load_model()

# Function to split text into chunks
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
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = words

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Streamlit UI
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

st.title('Text Summarization App :page_with_curl:')

st.text_area(
    "Write your article here: ",
    value=st.session_state.input_text,
    height=180,
    key="input_text"
)


# Summarize button
if st.button('Summarize'):
    if st.session_state.input_text.strip() == "":
        st.warning('Please enter text')
    else:
        with st.spinner('Summarizing...'):
            chunks = generate_chunks(st.session_state.input_text)
            result = model(chunks, max_length=80, min_length=10)
            summary = " ".join([res['summary_text'] for res in result])

        st.subheader("Summary:")
        st.write(summary)


