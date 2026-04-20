import streamlit as st
from transformers import pipeline
import time

# --- Constants ---
MAX_CHARS = 4000
MAX_QUESTIONS = 10

# --- Load pipelines ---
@st.cache_resource(show_spinner=False)
def load_pipelines():
    qa = pipeline(
        task="question-answering",
        model="distilbert-base-cased-distilled-squad",
        device=-1
    )
    qg = pipeline(
        task="text2text-generation",
        model="valhalla/t5-small-qg-hl",
        device=-1
    )
    return qa, qg

qa_pipeline, qg_pipeline = load_pipelines()

# --- UI ---
st.title("📚 Smart Q&A System 💡")

context = st.text_area("📝 Enter a paragraph:")
user_question = st.text_input("❓ Enter your question (optional):")

num_qs = st.number_input(
    "🔢 Number of generated questions:",
    min_value=1,
    max_value=MAX_QUESTIONS,
    value=5
)

# --- Button ---
if st.button("🚀 Generate"):
    if not context.strip():
        st.warning("⚠️ Enter some text first")
    else:
        with st.spinner("Processing..."):
            context = context[:MAX_CHARS]

            # Answer user question
            if user_question.strip():
                try:
                    ans = qa_pipeline(question=user_question, context=context)
                    st.success(f"Answer: {ans['answer']}")
                except:
                    st.error("Could not answer your question")

            # Generate questions
            st.subheader("Generated Questions & Answers")
            progress = st.progress(0)

            for i in range(num_qs):
                try:
                    q = qg_pipeline(context, max_length=64, do_sample=True)[0]['generated_text']
                    res = qa_pipeline(question=q, context=context)

                    st.write(f"{i+1}. Q: {q}")
                    st.write(f"   A: {res['answer']}")

                except:
                    st.write(f"{i+1}. Error generating question")

                progress.progress((i+1)/num_qs)
                time.sleep(0.2)

            st.balloons()