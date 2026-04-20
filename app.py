import time
import streamlit as st
from transformers import pipeline

# --- Constants ---
MAX_CHARS = 4000       # truncate long input to avoid token overflow
MAX_QUESTIONS = 10     # maximum questions allowed

# --- Load QA and QG pipelines with caching ---
@st.cache_resource(show_spinner=False)
def load_pipelines():
    # CPU-only, small models to avoid torch issues
    qa = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        device=-1  # CPU
    )
    qg = pipeline(
        "text2text-generation",
        model="valhalla/t5-small-qg-hl",
        device=-1  # CPU
    )
    return qa, qg

qa_pipeline, qg_pipeline = load_pipelines()

# --- Title ---
st.title("📚 Smart Q&A System 💡 with Flexible Question Generation 🌱✨")

# --- User input ---
context = st.text_area("📝 Enter a paragraph (context):")
user_question = st.text_input("❓ Enter your question (optional):")
num_qs = st.number_input(
    "🔢 How many possible questions do you want?",
    min_value=1,
    max_value=MAX_QUESTIONS,
    value=5,
    step=1
)

# --- Button ---
if st.button("🚀 Get Answer + Other Questions"):
    if not context.strip():
        st.warning("⚠️ Please enter a paragraph to continue. 😔✨")
    else:
        with st.spinner("🤔 Thinking... Please wait ⏳"):
            context_truncated = context[:MAX_CHARS]

            # --- User question ---
            if user_question.strip():
                try:
                    answer = qa_pipeline(question=user_question, context=context_truncated)
                    st.success(f"💡 Answer to your question: 👉 {answer['answer']}")
                except:
                    st.error("❌ Could not find an answer for your question.")

            # --- Generate other possible questions ---
            st.markdown("### 🧠 Other Possible Questions with Answers 💭")
            try:
                progress = st.progress(0)
                for idx in range(num_qs):
                    q = qg_pipeline(context_truncated, max_length=64, do_sample=True)[0]['generated_text']
                    try:
                        qa_result = qa_pipeline(question=q, context=context_truncated)
                        st.markdown(f"**{idx+1}. ❓ Q:** {q}")
                        st.markdown(f"➡️ **A:** {qa_result['answer']} ✅")
                    except:
                        st.markdown(f"**{idx+1}. ❓ Q:** {q}")
                        st.markdown("➡️ **A:** (No clear answer found) ⚠️")
                    progress.progress((idx+1)/num_qs)
                    time.sleep(0.2)
                st.balloons()
            except:
                st.error("❌ Could not generate questions. Try shorter text or fewer questions.")