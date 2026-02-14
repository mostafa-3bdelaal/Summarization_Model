import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

MODEL_NAME = "facebook/bart-large-cnn"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # استخدام text2text-generation بدلاً من summarization
    summarizer = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        framework="pt"
    )
    return summarizer

model = load_model()

# دالة تقسيم النصوص
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

# واجهة Streamlit
st.title("Text Summarization App 📄")

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
    if st.button("Summarize"):
        if st.session_state.input_text.strip() == "":
            st.warning("Please enter text")
        else:
            with st.spinner("Summarizing..."):
                chunks = generate_chunks(st.session_state.input_text)
                # تعديل ليتوافق مع text2text-generation
                result = model([f"summarize: {chunk}" for chunk in chunks], 
                               max_length=80, min_length=10)
                summary = " ".join([res['generated_text'] for res in result])
            st.subheader("Summary:")
            st.write(summary)

with col2:
    st.button("Clear", on_click=clear_text)
