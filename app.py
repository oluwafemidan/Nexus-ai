
import os
import streamlit as st

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Nexus AI | Workplace Assistant",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------
# NEXUS BRANDING
# ---------------------------------

st.markdown("""
<style>
.stApp {
    background: #F4F7FB;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 950px;
    padding-top: 2rem;
}

.nexus-header {
    background: linear-gradient(120deg, #10243B, #1767A0);
    padding: 28px;
    border-radius: 16px;
    color: white;
    margin-bottom: 22px;
}

.nexus-header h1 {
    color: white;
    margin-bottom: 4px;
}

.nexus-header p {
    color: #D7E6F5;
    margin: 0;
}

.nexus-case p{
    color: black;
}

.nexus-caption {
    color: #536579;
    font-size: 14px;
}

div[data-testid="stChatMessage"] {
    background: black;
    border-radius: 12px;
    border: 1px solid #E3EAF2;
    padding: 12px;
}

div[data-testid="stChatInput"] {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# LOAD AI COMPONENTS
# ---------------------------------


@st.cache_resource
def load_nexus_system():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Connect to existing ChromaDB
    database = Chroma(
        collection_name="nexus_company_knowledge",
        embedding_function=embeddings,
        persist_directory="./vector_store"
    )

    # Retrieve Gemini API key
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. "
            "Check your .env file."
        )

    # Initialize Google Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0,
        max_output_tokens=512
    )

    return database, llm


# ---------------------------------
# RETRIEVAL AND GENERATION
# ---------------------------------


def ask_nexus(question, database, llm):

    # Retrieve the five most relevant document chunks
    results = database.similarity_search_with_relevance_scores(
        question,
        k=5
    )

    # Keep chunks that meet the initial relevance threshold
    relevant_results = [
        (doc, score)
        for doc, score in results
        if score >= 0.35
    ]

    # Debugging: inspect the retrieval scores
    print("\nRETRIEVAL RESULTS")
    print("-" * 50)

    for doc, score in results:
        print(
            f"Source: {doc.metadata.get('source')} | "
            f"Relevance: {score:.4f}"
        )

    # Refuse when no sufficiently relevant information is found
    if not relevant_results:
        answer = (
            "I couldn't find sufficiently relevant information "
            "to answer this question in the supplied Nexus documents."
        )
        return answer, []

    # Construct context with identifiable source references
    context_parts = []

    for i, (doc, score) in enumerate(relevant_results, 1):

        source_name = doc.metadata.get(
            "source",
            "Unknown document"
        )

        context_parts.append(
            f"""
            [Source {i}]
            Document: {source_name}

            Content:
            {doc.page_content}
            """
        )

    context = "\n".join(context_parts)

    # Improved RAG prompt

    prompt = f"""
You are Nexus AI, the workplace knowledge assistant
for Nexus Workplace Services.

Answer the employee's question using ONLY the supplied context.

IMPORTANT ANSWER CLASSIFICATION:

You must classify every response into exactly one of these
three categories.

1. [[ANSWERABLE]]
Use this when the supplied documents contain sufficient
information to answer the question.

2. [[PARTIALLY_ANSWERABLE]]
Use this when the documents contain some information
that helps answer the question, but do not contain
all the requested details.

For example, if an employee asks how to submit an expense
claim and the documents provide deadlines, receipts and
approval requirements but not the actual submission
platform, provide the available information and explain
what is missing.

3. [[NOT_ANSWERABLE]]
Use this when the supplied documents contain no useful
information to answer the question.

For example:
- Who is the current chairman?
- How many workers does Nexus employ?
- What is the company's annual revenue?

STRICT RULES:

- Never invent facts or use external knowledge.
- Do not treat a semantically similar document as proof
  that the answer exists in that document.
- Combine relevant information from multiple chunks.
- Give complete, clear and well-structured answers.
- For partial answers, explain what is known and what
  is missing.
- Cite the supporting context using references such as
  [Source 1] or [Source 2].
- Every factual claim must be supported by the supplied context.
- Do not cite a source that does not support the claim.
- Include source references where appropriate.
- The classification marker MUST be the first line
  of your response.

SUPPLIED CONTEXT:

{context}

EMPLOYEE QUESTION:

{question}

YOUR RESPONSE:
"""

    response = llm.invoke(prompt)

    answer = response.content

    if isinstance(answer, list):
        answer = "\n".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    answer = answer.strip()

    # Determine whether the answer is supported by the documents
    if answer.startswith("[[NOT_ANSWERABLE]]"):

        answer = answer.replace(
            "[[NOT_ANSWERABLE]]", "", 1
        ).strip()

        return answer, []

    elif answer.startswith("[[PARTIALLY_ANSWERABLE]]"):

        answer = answer.replace(
            "[[PARTIALLY_ANSWERABLE]]", "", 1
        ).strip()

        return answer, relevant_results

    elif answer.startswith("[[ANSWERABLE]]"):

        answer = answer.replace(
            "[[ANSWERABLE]]", "", 1
        ).strip()

        return answer, relevant_results

    else:
        # If Gemini fails to provide a classification,
        # do not display potentially misleading sources.
        return answer, []


# ---------------------------------
# APPLICATION HEADER
# ---------------------------------

st.markdown("""
<div class="nexus-header">

<h1>🧠 NEXUS AI</h1>

<p>WORKPLACE INTELLIGENCE ASSISTANT</p>

<br>

<h3>Your workplace knowledge, intelligently connected.</h3>

<p>
Ask questions about company policies, IT security,
travel procedures and employee expenses.
</p>

</div>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="nexus-caption">Welcome to your Nexus workplace assistant.</p>',
    unsafe_allow_html=True
)

# ---------------------------------
# INITIALIZE CHAT
# ---------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------
# LOAD KNOWLEDGE BASE
# ---------------------------------

try:
    database, llm = load_nexus_system()
    knowledge_ready = True

except Exception:
    knowledge_ready = False

    st.error(
        "The Nexus AI service is temporarily unavailable. "
        "Please contact IT Support."
    )

# ---------------------------------
# SUGGESTED QUESTIONS
# ---------------------------------

st.markdown(
    '<h3 style="color: black;">What would you like to know?</h3>',
    unsafe_allow_html=True
)

suggestions = [
    "How do I submit travel expense claims?",
    "What approval is required for international travel?",
    "What are the IT security requirements?"
]

cols = st.columns(3)

for index, suggestion in enumerate(suggestions):

    with cols[index]:

        if st.button(
            suggestion,
            use_container_width=True
        ):

            st.session_state.pending_question = suggestion


# ---------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("View supporting documents"):

                for source in message["sources"]:

                    st.markdown(
                        f"**{source.metadata.get('source')}**"
                    )

                    st.write(source.page_content)


# ---------------------------------
# EMPLOYEE QUESTION INPUT
# ---------------------------------

question = st.chat_input(
    "Ask Nexus AI about company policies..."
)

if "pending_question" in st.session_state:

    question = st.session_state.pop("pending_question")


# ---------------------------------
# PROCESS QUESTION
# ---------------------------------

if question and knowledge_ready:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching Nexus knowledge base..."
        ):

            answer, results = ask_nexus(
                question,
                database,
                llm
            )

        st.markdown(answer)

        source_docs = [
            doc for doc, distance in results
        ]

    # Display sources ONLY when supporting evidence exists
    if source_docs:

        with st.expander("View Supporting Documents"):

            for doc in source_docs:

                st.markdown(
                    f"**{doc.metadata.get('source')}**"
                )

                st.write(doc.page_content)

    else:

        st.caption(
            "No supporting documents were found for this response."
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": source_docs
    })


# ---------------------------------
# FOOTER
# ---------------------------------

st.markdown("---")

st.markdown(
    "<center>NEXUS AI · Workplace Intelligence "
    "Assistant · Powered by Artificial Intelligence"
    "</center>",
    unsafe_allow_html=True
)
