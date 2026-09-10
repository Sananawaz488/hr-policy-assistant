import streamlit as st
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="👩‍💼",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .success-box {
        padding: 14px;
        border-radius: 10px;
        background-color: #e8f8f0;
        border: 1px solid #b7e4cd;
        color: #176b45;
        font-weight: 600;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f7fb;
        border: 1px solid #dfe4ee;
        margin-top: 10px;
    }

    .source-box {
        padding: 12px;
        border-radius: 10px;
        background-color: #fafafa;
        border: 1px solid #e3e3e3;
    }

    .workflow-item {
        margin-bottom: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">👩‍💼 HR Policy Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your company HR policy using '
    'Retrieval-Augmented Generation (RAG).'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD SENTENCE TRANSFORMER
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(pdf_file):

    pdf_bytes = pdf_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text("text")

        if text.strip():

            pages.append(
                {
                    "page": page_number,
                    "text": text.strip()
                }
            )

    document.close()

    return pages


# =========================================================
# TEXT CHUNKING
# =========================================================

def create_chunks(
    pages,
    chunk_size=800,
    overlap=150
):

    chunks = []

    for page_data in pages:

        text = page_data["text"]
        page_number = page_data["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number
                    }
                )

            start += chunk_size - overlap

    return chunks


# =========================================================
# CREATE FAISS INDEX
# =========================================================

def create_faiss_index(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# =========================================================
# SEARCH POLICY
# =========================================================

def search_policy(
    question,
    index,
    chunks,
    top_k
):

    question_embedding = (
        embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    )

    question_embedding = (
        question_embedding.astype(
            "float32"
        )
    )

    number_of_results = min(
        top_k,
        len(chunks)
    )

    scores, indices = index.search(
        question_embedding,
        number_of_results
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number == -1:
            continue

        result = chunks[
            index_number
        ].copy()

        result["score"] = float(
            score
        )

        results.append(result)

    return results


# =========================================================
# GROQ ANSWER
# =========================================================

def generate_answer(
    question,
    search_results
):

    # API key is NOT shown in the UI.
    # It is read securely from Streamlit Secrets.

    api_key = st.secrets.get(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please add it in Streamlit Cloud Secrets."
        )

    client = Groq(
        api_key=api_key
    )

    context_parts = []

    for result in search_results:

        context_parts.append(
            f"[Page {result['page']}]\n"
            f"{result['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    system_prompt = """
You are an HR Policy Assistant.

Your task is to answer questions using ONLY
the HR policy context provided to you.

IMPORTANT RULES:

1. Never invent an HR policy.
2. Never guess information that is not in the document.
3. If the answer cannot be found in the uploaded
   HR policy, say:
   "I could not find this information in the uploaded HR policy."
4. Give clear and professional answers.
5. Mention the relevant page number when possible.
6. Do not use general HR knowledge as if it came
   from the uploaded policy.
7. Keep the answer concise but helpful.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}


USER QUESTION:

{question}


Answer the user's question using ONLY the
HR policy context above.

Mention the relevant page number when possible.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens=700
    )

    return response.choices[
        0
    ].message.content


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📚 How it works")

    st.markdown(
        """
        <div class="workflow-item">
        <b>1.</b> Upload an HR Policy PDF
        </div>

        <div class="workflow-item">
        <b>2.</b> Extract text from the PDF
        </div>

        <div class="workflow-item">
        <b>3.</b> Split text into chunks
        </div>

        <div class="workflow-item">
        <b>4.</b> Generate embeddings
        </div>

        <div class="workflow-item">
        <b>5.</b> Store embeddings in FAISS
        </div>

        <div class="workflow-item">
        <b>6.</b> Retrieve relevant policy sections
        </div>

        <div class="workflow-item">
        <b>7.</b> Generate an answer using Groq
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.header("🔎 Search Settings")

    top_k = st.slider(
        "Relevant policy sections",
        min_value=1,
        max_value=8,
        value=5
    )

    st.markdown("---")

    st.info(
        "The assistant answers questions based "
        "on the uploaded HR policy."
    )


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    '<div class="section-title">📄 Upload HR Policy</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload your HR Policy PDF",
    type=["pdf"]
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file:

    if (
        "processed_file_name"
        not in st.session_state
        or
        st.session_state.processed_file_name
        != uploaded_file.name
    ):

        with st.spinner(
            "Processing HR Policy..."
        ):

            try:

                pages = extract_text_from_pdf(
                    uploaded_file
                )

                if not pages:

                    st.error(
                        "No readable text was found "
                        "in this PDF. Please upload "
                        "a text-based PDF."
                    )

                    st.stop()

                chunks = create_chunks(
                    pages
                )

                if not chunks:

                    st.error(
                        "Could not create searchable "
                        "chunks from the PDF."
                    )

                    st.stop()

                index = create_faiss_index(
                    chunks
                )

                st.session_state.pages = pages

                st.session_state.chunks = chunks

                st.session_state.index = index

                st.session_state.processed_file_name = (
                    uploaded_file.name
                )

            except Exception as error:

                st.error(
                    f"Error processing PDF: {error}"
                )

                st.stop()

    else:

        pages = st.session_state.pages

        chunks = st.session_state.chunks

        index = st.session_state.index


    # =====================================================
    # SUCCESS MESSAGE
    # =====================================================

    st.markdown(
        f"""
        <div class="success-box">
        ✅ HR Policy processed successfully!
        Created {len(chunks)} searchable chunks.
        </div>
        """,
        unsafe_allow_html=True
    )


    st.write("")


    # =====================================================
    # METRICS
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📄 Pages",
            len(pages)
        )

    with col2:

        st.metric(
            "🧩 Searchable Chunks",
            len(chunks)
        )

    with col3:

        st.metric(
            "🔎 Retrieved Sections",
            top_k
        )


    st.markdown("---")


    # =====================================================
    # EXAMPLE QUESTIONS
    # =====================================================

    st.markdown(
        '<div class="section-title">💡 Example Questions</div>',
        unsafe_allow_html=True
    )

    question_col1, question_col2, question_col3 = (
        st.columns(3)
    )

    example_questions = [
        "How many annual leave days are employees entitled to?",
        "What is the notice period?",
        "Can employees work remotely?"
    ]

    with question_col1:

        example_1 = st.button(
            "📅 Annual Leave",
            use_container_width=True
        )

    with question_col2:

        example_2 = st.button(
            "📋 Notice Period",
            use_container_width=True
        )

    with question_col3:

        example_3 = st.button(
            "🏠 Remote Work",
            use_container_width=True
        )


    # =====================================================
    # QUESTION INPUT
    # =====================================================

    if example_1:

        st.session_state.question = (
            example_questions[0]
        )

    if example_2:

        st.session_state.question = (
            example_questions[1]
        )

    if example_3:

        st.session_state.question = (
            example_questions[2]
        )


    if "question" not in st.session_state:

        st.session_state.question = ""


    question = st.text_input(
        "💬 Ask a question about the policy",
        value=st.session_state.question,
        placeholder=(
            "Example: How many annual leaves "
            "are employees entitled to?"
        )
    )


    ask_button = st.button(
        "🔎 Ask HR Policy",
        type="primary",
        use_container_width=True
    )


    # =====================================================
    # ANSWER
    # =====================================================

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            with st.spinner(
                "🔎 Searching relevant policy sections..."
            ):

                try:

                    search_results = search_policy(
                        question,
                        index,
                        chunks,
                        top_k
                    )

                except Exception as error:

                    st.error(
                        f"Search error: {error}"
                    )

                    st.stop()


            with st.spinner(
                "🤖 Generating HR policy answer..."
            ):

                try:

                    answer = generate_answer(
                        question,
                        search_results
                    )

                except Exception as error:

                    st.error(
                        f"AI error: {error}"
                    )

                    st.stop()


            # =================================================
            # ANSWER DISPLAY
            # =================================================

            st.markdown(
                '<div class="section-title">🤖 Answer</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="answer-box">
                {answer}
                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # SOURCES
            # =================================================

            st.markdown("---")

            st.markdown(
                '<div class="section-title">📚 Retrieved Policy Sections</div>',
                unsafe_allow_html=True
            )

            for number, result in enumerate(
                search_results,
                start=1
            ):

                with st.expander(
                    f"📄 Source {number} — Page {result['page']}"
                ):

                    st.write(
                        result["text"]
                    )

                    st.caption(
                        f"Similarity score: "
                        f"{result['score']:.3f}"
                    )


else:

    st.info(
        "👆 Upload an HR Policy PDF to get started."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "HR Policy Assistant • RAG • FAISS • "
    "Sentence Transformers • PyMuPDF • Groq"
)
