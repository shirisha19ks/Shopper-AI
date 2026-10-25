
import json
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.dataset_analysis import analyze_dataset
from src.prediction_tool import predict_purchase


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    ROOT / "documents"
    / "Online_Shoppers_Dataset_Knowledge_Base_5_Pages.pdf"
)


# --------------------------------------------------
# 2. Initialize Groq LLM
# --------------------------------------------------

api_key = input("Enter your Groq API key: ").strip()

if not api_key:
    raise ValueError("Groq API key is required.")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=api_key
)

print("Groq LLM initialized successfully!")


# --------------------------------------------------
# 3. Build RAG pipeline
# --------------------------------------------------

print("Loading dataset knowledge PDF...")

if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"Knowledge PDF not found: {PDF_PATH}"
    )

pages = PyPDFLoader(str(PDF_PATH)).load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

rag_prompt = ChatPromptTemplate.from_template("""
Answer the user's question using only the supplied
Online Shoppers dataset knowledge context.

Do not invent dataset facts or statistics.
If the context does not contain the answer, say so.

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("RAG pipeline initialized successfully!")


# --------------------------------------------------
# 4. Tool 1: Dataset knowledge tool
# --------------------------------------------------

def dataset_rag_tool(question):
    """Answer dataset definition questions using the PDF."""
    return rag_chain.invoke(question)


# --------------------------------------------------
# 5. Tool 2: Dataset analysis tool
# --------------------------------------------------

def dataset_analysis_tool(question):
    """Calculate actual statistics from the CSV dataset."""
    result = analyze_dataset(question)
    return json.dumps(result, default=str, indent=2)


# --------------------------------------------------
# 6. Tool 3: Purchase prediction tool
# --------------------------------------------------

def purchase_prediction_tool(feature_json):
    """
    Predict using a JSON dictionary containing encoded
    feature names and values expected by the trained model.
    """

    try:
        session_data = json.loads(feature_json)

        if not isinstance(session_data, dict):
            return "Input must be a JSON object of encoded features."

        result = predict_purchase(session_data)

        return json.dumps(result, default=str, indent=2)

    except json.JSONDecodeError:
        return (
            "Invalid JSON. Provide a JSON object with encoded "
            "feature names and numeric values."
        )
    except Exception as error:
        return f"Prediction error: {error}"


# --------------------------------------------------
# 7. AI Agent routing
# --------------------------------------------------

router_prompt = ChatPromptTemplate.from_template("""
Choose the best tool for the user's request.

Return exactly one label:
RAG
ANALYSIS
PREDICTION

Rules:
- RAG: asks what a dataset column means, definitions,
  dataset description, or knowledge from the PDF.
- ANALYSIS: asks for actual CSV counts, percentages,
  missing values, duplicates, or summary statistics.
- PREDICTION: asks to predict purchase for a session
  and provides encoded feature values as JSON.
- If the user asks for a prediction without encoded
  feature values, return PREDICTION.

User request:
{question}

Tool label:
""")

router = router_prompt | llm | StrOutputParser()


def run_agent(question):
    """Select and execute the appropriate tool."""

    route = router.invoke(
        {"question": question}
    ).strip().upper()

    print(f"\nSelected tool: {route}")

    if "PREDICTION" in route:
        marker = "FEATURES_JSON:"

        if marker not in question:
            return (
                "Prediction tool selected. To predict a session, "
                "provide its encoded features as JSON after "
                "'FEATURES_JSON:'. Use the 29 feature names and "
                "values expected by the trained model. Do not "
                "include the Revenue target."
            )

        feature_json = question.split(marker, 1)[1].strip()

        return purchase_prediction_tool(feature_json)

    elif "ANALYSIS" in route:
        return dataset_analysis_tool(question)

    else:
        return dataset_rag_tool(question)


# --------------------------------------------------
# 8. Interactive agent
# --------------------------------------------------

print("\n" + "=" * 55)
print("ONLINE SHOPPERS AI AGENT READY")
print("=" * 55)

print("Tools available:")
print("1. Dataset RAG")
print("2. Dataset Analysis")
print("3. Purchase Prediction")

print("\nType 'exit' to stop.")

while True:
    try:
        question = input(
            "\nEnter your question: "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("AI Agent stopped.")
            break

        if not question:
            continue

        answer = run_agent(question)

        print("\nAgent Answer:")
        print("-" * 50)
        print(answer)

    except KeyboardInterrupt:
        print("\nAI Agent stopped.")
        break

    except Exception as error:
        print(f"\nError: {error}")

