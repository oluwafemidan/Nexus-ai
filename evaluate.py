
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ---------------------------------
# CONFIGURATION
# Must match your existing application
# ---------------------------------

COLLECTION_NAME = "nexus_company_knowledge"

VECTOR_FOLDER = "./vector_store"

EMBEDDING_MODEL = "gemini-embedding-001"

TOP_K = 5


# ---------------------------------
# INITIALIZE EMBEDDINGS
# ---------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
   
)


# ---------------------------------
# CONNECT TO EXISTING DATABASE
# ---------------------------------

database = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=VECTOR_FOLDER
)


# ---------------------------------
# EVALUATION QUESTIONS
# ---------------------------------

test_cases = [

    {
        "question": "What security measures must employees follow?",
        "expected_source": "02_IT_Acceptable_Use_Policy.txt"
    },

    {
        "question": "What approval is required for international travel?",
        "expected_source": "01_Staff_Travel_Policy.txt"
    },

    {
        "question": "What are the requirements for expense receipts?",
        "expected_source": "03_Expense_Reimbursement_Guide.txt"
    },

    {
        "question": "What is the company's annual leave entitlement?",
        "expected_source": None
    }

]


# ---------------------------------
# EVALUATE RETRIEVAL
# ---------------------------------

def evaluate_retrieval():

    if database._collection.count() == 0:

        raise RuntimeError(
            "Knowledge base is empty. "
            "Run index_knowlegde.py first."
        )

    passed = 0

    print("\nNEXUS AI RETRIEVAL EVALUATION")

    print("=" * 50)

    for test in test_cases:

        question = test["question"]

        expected = test["expected_source"]

        results = database.similarity_search_with_score(
            question,
            k=TOP_K
        )

        retrieved_sources = {
            doc.metadata.get("source")
            for doc, distance in results
        }

        if expected is None:

            # This is an exploratory test.
            # Retrieval results alone cannot prove
            # that a question is unanswerable.

            status = "MANUAL REVIEW"

        else:

            success = expected in retrieved_sources

            status = "PASS" if success else "FAIL"

            if success:
                passed += 1

        print("\nQuestion:", question)

        print("Expected source:", expected)

        print("Retrieved sources:", retrieved_sources)

        print("Result:", status)

    print("\n" + "=" * 50)

    print(
        f"Answerable retrieval tests passed: "
        f"{passed}/3"
    )


# ---------------------------------
# EXECUTION
# ---------------------------------

if __name__ == "__main__":

    evaluate_retrieval()
