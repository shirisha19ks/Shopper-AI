
import os
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    ROOT
    / "documents"
    / "Online_Shoppers_Dataset_Knowledge_Base_5_Pages.pdf"
)


# --------------------------------------------------
# 2. Groq API key
# --------------------------------------------------

# Replace this with your NEW Groq API key.
# Never share or upload your API key.

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    api_key = input("Enter your Groq API key: ").strip()

if not api_key:
    raise ValueError("Groq API key is required.")


# --------------------------------------------------
# 3. Initialize Groq LLM
# --------------------------------------------------

llm_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    api_key=api_key
)

print("Groq LLM initialized successfully!")


# --------------------------------------------------
# 4. Load dataset knowledge PDF
# --------------------------------------------------

print("Loading dataset knowledge PDF...")

if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"Knowledge PDF not found: {PDF_PATH}"
    )

loader = PyPDFLoader(str(PDF_PATH))
documents = loader.load()

print(f"PDF pages loaded: {len(documents)}")


# --------------------------------------------------
# 5. Split documents into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

text_chunks = text_splitter.split_documents(documents)

print(f"Total text chunks: {len(text_chunks)}")


# --------------------------------------------------
# 6. Create embeddings
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


# --------------------------------------------------
# 7. Create FAISS vector database
# --------------------------------------------------

vectorstore = FAISS.from_documents(
    documents=text_chunks,
    embedding=embedding_model
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("FAISS vector database created successfully!")


# --------------------------------------------------
# 8. Create RAG prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant for the
Online Shoppers Purchasing Intention project.

Answer the question using only the retrieved
context from the dataset knowledge PDF.

Instructions:
1. Explain the answer in simple language.
2. Use dataset-specific information.
3. Do not invent statistics or column meanings.
4. If the context does not contain the answer,
   clearly say that the knowledge base does not
   provide enough information.
5. Explain relevant dataset columns when useful.

Retrieved context:
{context}

User question:
{question}

Answer:
""")


# --------------------------------------------------
# 9. Format retrieved documents
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


# --------------------------------------------------
# 10. Build RAG chain
# --------------------------------------------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm_model
    | StrOutputParser()
)

print("\nDataset RAG + Groq LLM is ready!")
print("Type 'exit' to stop.")


# --------------------------------------------------
# 11. Ask questions interactively
# --------------------------------------------------

while True:
    try:
        question = input(
            "\nEnter your dataset question: "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("RAG application stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        answer = rag_chain.invoke(question)

        print("\nRAG Answer:")
        print("-" * 50)
        print(answer)

        # Display retrieved source information
        source_docs = retriever.invoke(question)

        print("\nRetrieved Sources:")

        for i, doc in enumerate(source_docs, start=1):
            page_number = doc.metadata.get("page", "N/A")

            print(f"\nSource {i} | PDF page: {page_number + 1 if isinstance(page_number, int) else page_number}")
            print(doc.page_content[:300])

    except KeyboardInterrupt:
        print("\nRAG application stopped.")
        break

    except Exception as error:
        print(f"\nError: {error}")
        print(
            "Check your Groq API key, model availability, "
            "internet connection, and installed packages."
        )


