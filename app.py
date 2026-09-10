import streamlit as st
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📋",
    layout="wide"
)


st.title("📋 HR Policy Assistant")
st.write(
    "Upload an HR policy PDF and ask questions about its contents."
)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


def extract_text_from_pdf(pdf_file):

    pdf_bytes = pdf_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text")

        if text.strip():

            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    document.close()

    return pages


def create_chunks(pages, chunk_size=800, overlap=150):

    chunks = []

    for page_data in pages:

        text = page_data["text"]
        page_number = page_data["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            start += chunk_size - overlap

    return chunks


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

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search_policy(
    question,
    index,
    chunks,
    top_k=5
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number == -1:
            continue

        result = chunks[index_number].copy()

        result["score"] = float(score)

        results.append(result)

    return results


def generate_answer(
    question,
    search_results
):

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Add it in Streamlit Cloud Secrets."
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

Answer questions ONLY using the provided HR policy context.

Rules:

1. Do not invent HR policies.
2. If the answer is not in the context, say:
"I could not find this information in the uploaded HR policy."
3. Give a clear and professional answer.
4. Mention the relevant page number when possible.
5. Do not use general HR knowledge as if it came from the document.
6. Keep answers concise but useful.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

USER QUESTION:

{question}

Answer using only the HR policy context.
Mention relevant page numbers when possible.
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

    return response.choices[0].message.content


with st.sidebar:

    st.header("⚙️ Settings")

    top_k = st.slider(
        "Relevant sections",
        min_value=1,
        max_value=8,
        value=5
    )

    st.markdown("---")

    st.info(
        "Upload an HR policy PDF and ask questions "
        "about leave, attendance, working hours, "
        "benefits, conduct, or company rules."
    )


uploaded_file = st.file_uploader(
    "📄 Upload HR Policy PDF",
    type=["pdf"]
)


if uploaded_file:

    if (
        "processed_file_name" not in st.session_state
        or st.session_state.processed_file_name
        != uploaded_file.name
    ):

        with st.spinner(
            "Reading and indexing the HR policy..."
        ):

            try:

                pages = extract_text_from_pdf(
                    uploaded_file
                )

                if not pages:

                    st.error(
                        "No readable text was found "
                        "in this PDF."
                    )

                    st.stop()

                chunks = create_chunks(
                    pages
                )

                if not chunks:

                    st.error(
                        "Could not create text chunks."
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

            except Exception as e:

                st.error(
                    f"Error while processing PDF: {e}"
                )

                st.stop()

    else:

        pages = st.session_state.pages
        chunks = st.session_state.chunks
        index = st.session_state.index


    st.success(
        f"✅ Policy loaded: {uploaded_file.name}"
    )


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Pages",
            len(pages)
        )

    with col2:

        st.metric(
            "Text chunks",
            len(chunks)
        )


    st.markdown("---")


    st.subheader(
        "💬 Ask about the HR policy"
    )


    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: How many annual leaves "
            "are employees allowed?"
        )
    )


    ask_button = st.button(
        "🔎 Ask Question",
        type="primary"
    )


    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            with st.spinner(
                "Searching the HR policy..."
            ):

                try:

                    search_results = search_policy(
                        question,
                        index,
                        chunks,
                        top_k
                    )

                except Exception as e:

                    st.error(
                        f"Search error: {e}"
                    )

                    st.stop()


            with st.spinner(
                "Generating answer..."
            ):

                try:

                    answer = generate_answer(
                        question,
                        search_results
                    )

                except Exception as e:

                    st.error(
                        f"AI error: {e}"
                    )

                    st.stop()


            st.subheader(
                "🤖 Answer"
            )

            st.markdown(answer)


            st.markdown("---")

            st.subheader(
                "📚 Retrieved Policy Sections"
            )


            for number, result in enumerate(
                search_results,
                start=1
            ):

                with st.expander(
                    f"Source {number} — "
                    f"Page {result['page']}"
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
        "👆 Upload an HR policy PDF to get started."
    )


st.markdown("---")

st.caption(
    "HR Policy Assistant • RAG • FAISS • "
    "Sentence Transformers • PyMuPDF • Groq"
)
