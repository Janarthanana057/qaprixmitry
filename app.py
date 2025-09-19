"""import streamlit as st
from transformers import pipeline


qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
qg_pipeline = pipeline("text2text-generation", model="iarfmoose/t5-base-question-generator")


st.title("📚 Smart Q&A System with Flexible 💡 Question Generation🌱")

# User input
context = st.text_area("📈Enter a paragraph (context):")
user_question = st.text_input("Enter your question (optional):")

# Ask user how many questions they want
num_qs = st.number_input(
    "How many possible questions do you want?",
    min_value=1, max_value=20, value=5, step=1
)

if st.button("🎯Get Answer + Other Questions"):
    if context.strip():
        # --- Case 1: User entered a question ---
        if user_question.strip():
            answer = qa_pipeline(question=user_question, context=context)
            st.success(f"Answer to your question: 👉 {answer['answer']}")

        # --- Case 2: Generate other possible questions ---
        st.markdown("🧠 Other Possible Questions with Answers: 💭")
        generated = qg_pipeline(
            context,
            max_length=64,
            do_sample=True,
            top_k=50,
            num_return_sequences=num_qs  # user controls this
        )

        for idx, q in enumerate(generated, 1):
            question_text = q['generated_text']
            try:
                qa_result = qa_pipeline(question=question_text, context=context)
                st.markdown(f"**{idx}. Q:** {question_text}")
                st.markdown(f"➡️ **A:** {qa_result['answer']}")
            except:
                st.markdown(f"**{idx}. Q:** {question_text}")
                st.markdown("➡️ **A:** (No clear answer found)")

    else:
        st.warning("⚠️ Please enter a paragraph to continue.😔જ⁀➴")
"""
import time
import streamlit as st
from transformers import pipeline

# Load QA and QG pipelines
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
qg_pipeline = pipeline("text2text-generation", model="iarfmoose/t5-base-question-generator")

# Title with emojis
st.title("📚 Smart Q&A System 💡 with Flexible Question Generation 🌱✨")

# User input
context = st.text_area("📝 Enter a paragraph (context):")
user_question = st.text_input("❓ Enter your question (optional):")

# Ask user how many questions they want
num_qs = st.number_input(
    "🔢 How many possible questions do you want?",
    min_value=1, max_value=20, value=5, step=1
)

# Button
if st.button("🚀 Get Answer + Other Questions"):
    if context.strip():
        with st.spinner("🤔 Thinking... Please wait ⏳"):
            # --- Case 1: User entered a question ---
            if user_question.strip():
                answer = qa_pipeline(question=user_question, context=context)
                st.success(f"💡 Answer to your question: 👉 {answer['answer']}")

            # --- Case 2: Generate other possible questions ---
            st.markdown("### 🧠 Other Possible Questions with Answers 💭")

            generated = qg_pipeline(
                context,
                max_length=64,
                do_sample=True,
                top_k=50,
                num_return_sequences=num_qs
            )

            progress = st.progress(0)  # progress bar
            for idx, q in enumerate(generated, 1):
                question_text = q['generated_text']
                try:
                    qa_result = qa_pipeline(question=question_text, context=context)
                    st.markdown(f"**{idx}. ❓ Q:** {question_text}")
                    st.markdown(f"➡️ **A:** {qa_result['answer']} ✅")
                except:
                    st.markdown(f"**{idx}. ❓ Q:** {question_text}")
                    st.markdown("➡️ **A:** (No clear answer found) ⚠️")

                progress.progress(idx / num_qs)
                time.sleep(0.2)  # small delay for animation effect

        st.balloons()  # 🎈 celebration when done

    else:
        st.warning("⚠️ Please enter a paragraph to continue. 😔✨")
