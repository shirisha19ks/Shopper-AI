
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent

CSV_PATH = ROOT / "data" / "online_shoppers_intention.csv"
CLEAN_PATH = ROOT / "data" / "iqr_capped_online_shoppers.csv"
ENCODED_PATH = ROOT / "data" / "encoded_online_shoppers.csv"
MODEL_PATH = ROOT / "models" / "best_tuned_model.joblib"
PDF_PATH = ROOT / "documents" / "Online_Shoppers_Dataset_Knowledge_Base_5_Pages.pdf"

st.set_page_config(
    page_title="Shopper Intelligence AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. COLORFUL CSS
# =====================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f4f4ff 0%, #eef8ff 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #231557 0%, #441b72 55%, #7436a6 100%);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.hero {
    padding: 30px;
    border-radius: 22px;
    color: white;
    background: linear-gradient(115deg, #512da8, #7c4dff, #00bcd4);
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    font-size: 34px;
}
.hero p {
    color: white;
    font-size: 17px;
}
.kpi {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border-left: 5px solid #7c4dff;
    box-shadow: 0 4px 14px rgba(60, 40, 120, 0.10);
    min-height: 110px;
}
.kpi-label {
    color: #686783;
    font-size: 14px;
}
.kpi-value {
    color: #30205f;
    font-size: 27px;
    font-weight: 700;
}
.section-title {
    color: #39216d;
    font-size: 23px;
    font-weight: 700;
}
div.stButton > button {
    border-radius: 10px;
    background: linear-gradient(90deg, #673ab7, #2979ff);
    color: white;
    border: none;
    font-weight: 600;
}
div.stButton > button:hover {
    border: 1px solid #7c4dff;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD DATA
# =====================================================

@st.cache_data
def load_dataset():
    path = CSV_PATH if CSV_PATH.exists() else CLEAN_PATH

    if not path.exists():
        return None

    return pd.read_csv(path)


@st.cache_data
def load_encoded_dataset():
    if ENCODED_PATH.exists():
        return pd.read_csv(ENCODED_PATH)
    return None


df = load_dataset()
encoded_df = load_encoded_dataset()


# =====================================================
# 4. SHARED FUNCTIONS
# =====================================================

def show_hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def kpi(label, value):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_resource
def load_rag(api_key):
    if not PDF_PATH.exists():
        raise FileNotFoundError("Dataset knowledge PDF not found.")

    pages = PyPDFLoader(str(PDF_PATH)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        api_key=api_key
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an assistant for the Online Shoppers Purchasing
    Intention dataset.

    Answer using the retrieved knowledge context only.
    Do not invent statistics or dataset facts.
    If the context is insufficient, say so clearly.
    Use simple language.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": lambda question: question
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def analyze_question(question):
    if df is None:
        return "Dataset CSV file was not found."

    q = question.lower()

    if q == "summary" or "dataset summary" in q:
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())
        }

    if "missing" in q or "null" in q:
        return df.isnull().sum().to_dict()

    if "duplicate" in q:
        return {"duplicate_rows": int(df.duplicated().sum())}

    if "purchase" in q or "revenue" in q:
        if "Revenue" not in df.columns:
            return "Revenue column not found."

        revenue = df["Revenue"].astype(str).str.lower()
        true_mask = revenue.isin(["true", "1"])
        false_mask = revenue.isin(["false", "0"])

        purchased = int(true_mask.sum())
        not_purchased = int(false_mask.sum())
        total = purchased + not_purchased

        return {
            "purchase_sessions": purchased,
            "non_purchase_sessions": not_purchased,
            "purchase_percentage": round(purchased / total * 100, 2)
            if total else 0,
            "non_purchase_percentage": round(not_purchased / total * 100, 2)
            if total else 0
        }

    return "Try asking about purchase percentage, missing values, duplicates, or dataset summary."


# =====================================================
# 5. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:
    st.markdown("# 🛍️ Shopper AI")
    st.caption("Intelligent E-commerce Analytics")
    st.markdown("---")

    page = st.radio(
    "NAVIGATION",
    [
        "🏠 Home Dashboard",
        "📊 Dataset Analysis",
        "💬 RAG Chatbot",
        "🎯 Purchase Prediction",
        "🤖 AI Agent & Tools",
        "📈 ML Model Comparison",
        "💡 AI Insights & Recommendations",
        "🧪 Evaluation & Reliability",
        "ℹ️ About Project"
    ]
)

    st.markdown("---")
    st.caption("Powered by Python • ML • RAG • Groq")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Your key is used for this app session."
        )


# =====================================================
# PAGE 1: HOME DASHBOARD
# =====================================================

if page == "🏠 Home Dashboard":
    show_hero(
        "🛍️ Shopper Intelligence AI",
        "Understand browsing behaviour and online purchase intention."
    )

    if df is None:
        st.error("CSV not found. Check the project's data folder.")
    else:
        total_rows = len(df)
        total_columns = len(df.columns)
        missing = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())

        revenue = df["Revenue"].astype(str).str.lower()
        purchase_count = int(revenue.isin(["true", "1"]).sum())
        valid_count = int(revenue.isin(
            ["true", "false", "1", "0"]
        ).sum())

        purchase_rate = (
            purchase_count / valid_count * 100
            if valid_count else 0
        )

        a, b, c, d = st.columns(4)
        with a:
            kpi("Total Sessions", f"{total_rows:,}")
        with b:
            kpi("Dataset Columns", str(total_columns))
        with c:
            kpi("Purchase Rate", f"{purchase_rate:.2f}%")
        with d:
            kpi("Missing Values", f"{missing:,}")

        st.markdown("")
        left, right = st.columns(2)

        with left:
            st.markdown(
                '<div class="section-title">Purchase Distribution</div>',
                unsafe_allow_html=True
            )
            dist = pd.DataFrame({
                "Outcome": ["Purchased", "No Purchase"],
                "Sessions": [
                    purchase_count,
                    valid_count - purchase_count
                ]
            })
            fig = px.pie(
                dist,
                names="Outcome",
                values="Sessions",
                hole=0.55,
                color="Outcome",
                color_discrete_map={
                    "Purchased": "#00b894",
                    "No Purchase": "#7c4dff"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown(
                '<div class="section-title">Project Highlights</div>',
                unsafe_allow_html=True
            )
            st.markdown("""
            - 📚 PDF-powered RAG chatbot
            - 📊 Real CSV data analysis
            - 🤖 Trained machine learning model
            - 🧠 Groq LLM integration
            - 🌈 Interactive Streamlit dashboard
            """)

        st.metric("Duplicate Rows", f"{duplicates:,}")

        st.markdown("### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)


# =====================================================
# PAGE 2: DATASET ANALYSIS
# =====================================================

elif page == "📊 Dataset Analysis":
    show_hero(
        "📊 Dataset Analytics",
        "Explore dataset quality, purchase distribution and features."
    )

    if df is None:
        st.error("Dataset not found.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi("Rows", f"{len(df):,}")
        with c2:
            kpi("Columns", f"{len(df.columns)}")
        with c3:
            kpi("Duplicates", f"{df.duplicated().sum():,}")

        st.markdown("### Ask for statistics")
        question = st.selectbox(
            "Choose analysis",
            [
                "Dataset summary",
                "Purchase percentage",
                "Missing values",
                "Duplicate rows"
            ]
        )

        if st.button("Run Analysis"):
            result = analyze_question(question)
            st.json(result)

        st.markdown("### Revenue Distribution")

        if "Revenue" in df.columns:
            revenue_counts = (
                df["Revenue"].astype(str).value_counts()
                .rename_axis("Revenue")
                .reset_index(name="Sessions")
            )

            fig = px.bar(
                revenue_counts,
                x="Revenue",
                y="Sessions",
                color="Revenue",
                color_discrete_sequence=[
                    "#7c4dff", "#00bcd4", "#ff9800"
                ],
                text="Sessions"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Column information")
        st.dataframe(
            pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing Values": df.isnull().sum().values
            }),
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Dataset CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="online_shoppers_dataset.csv",
            mime="text/csv"
        )


# =====================================================
# PAGE 3: RAG CHATBOT
# =====================================================

elif page == "💬 RAG Chatbot":
    show_hero(
        "💬 Dataset Knowledge Assistant",
        "Ask questions about columns, browsing sessions and dataset meaning."
    )

    if not api_key:
        st.warning("Enter your Groq API key in the sidebar to use the chatbot.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])

        question = st.chat_input(
            "Example: What does ProductRelated_Duration mean?"
        )

        if question:
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching the knowledge base..."):
                    try:
                        chain, retriever = load_rag(api_key)
                        answer = chain.invoke(question)
                        st.markdown(answer)

                        docs = retriever.invoke(question)
                        with st.expander("Retrieved PDF sources"):
                            for doc in docs:
                                st.write(doc.page_content)
                                st.caption(
                                    f"PDF page: {doc.metadata.get('page', 0) + 1}"
                                )

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer
                        })

                    except Exception as e:
                        st.error(f"RAG error: {e}")

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# =====================================================
# PAGE 4: PURCHASE PREDICTION
# =====================================================

elif page == "🎯 Purchase Prediction":
    show_hero(
        "🎯 Purchase Intention Predictor",
        "Enter browsing session details to estimate purchase intention."
    )

    st.info(
        "Prediction uses the existing trained model. The inputs are "
        "encoded using the feature columns in your saved dataset."
    )

    if not MODEL_PATH.exists() or encoded_df is None:
        st.error(
            "Model or encoded dataset not found. Check models/ and data/."
        )
    elif "Revenue" not in encoded_df.columns:
        st.error("Revenue target column is missing from encoded dataset.")
    else:
        model = joblib.load(MODEL_PATH)
        feature_columns = [
            col for col in encoded_df.columns if col != "Revenue"
        ]

        raw_df = pd.read_csv(CLEAN_PATH) if CLEAN_PATH.exists() else df

        numeric_fields = [
            "Administrative",
            "Administrative_Duration",
            "Informational",
            "Informational_Duration",
            "ProductRelated",
            "ProductRelated_Duration",
            "BounceRates",
            "ExitRates",
            "PageValues",
            "SpecialDay"
        ]

        categorical_fields = [
            "Month",
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
            "VisitorType",
            "Weekend"
        ]

        with st.form("prediction_form"):
            st.markdown("### Browsing Behaviour")
            values = {}

            left, right = st.columns(2)

            for index, field in enumerate(numeric_fields):
                default_value = (
                    float(raw_df[field].median())
                    if raw_df is not None and field in raw_df.columns
                    else 0.0
                )

                with (left if index % 2 == 0 else right):
                    values[field] = st.number_input(
                        field,
                        value=default_value,
                        min_value=0.0
                    )

            st.markdown("### Session Details")

            for index, field in enumerate(categorical_fields):
                if raw_df is not None and field in raw_df.columns:
                    options = sorted(
                        raw_df[field].dropna().unique().tolist(),
                        key=str
                    )
                else:
                    options = []

                if not options:
                    st.warning(f"No category options found for {field}.")
                    continue

                values[field] = st.selectbox(field, options)

            submitted = st.form_submit_button("🔮 Predict Purchase Intention")

        
        if submitted:
            try:
                input_df = pd.DataFrame([values])

                # Encode only the categorical columns used in training
                dummy_columns = ["Month", "VisitorType", "Weekend"]

                for field in dummy_columns:
                    if field in input_df.columns:
                        input_df[field] = input_df[field].astype(str)

                encoded_input = pd.get_dummies(
                    input_df,
                    columns=dummy_columns,
                    dtype=int
                )

                # Match the saved model's feature columns
                encoded_input = encoded_input.reindex(
                    columns=feature_columns,
                    fill_value=0
                )

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(encoded_input)[0]
                    positive_index = list(model.classes_).index(1)
                    probability = float(probabilities[positive_index])
                    prediction = int(probability >= 0.40)
                else:
                    prediction = int(model.predict(encoded_input)[0])
                    probability = None

                st.markdown("### Prediction Result")

                if prediction == 1:
                    st.success("🟢 Purchase likely")
                else:
                    st.warning("🟠 Purchase not likely")

                if probability is not None:
                    st.metric(
                        "Estimated Purchase Probability",
                        f"{probability * 100:.2f}%"
                    )
                    st.progress(max(0.0, min(probability, 1.0)))

                st.caption(
                    "This is a model estimate, not a guarantee of a purchase."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")



# =====================================================
# AI AGENT & TOOLS PAGE
# =====================================================

elif page == "🤖 AI Agent & Tools":

    show_hero(
        "🤖 Autonomous AI Agent",
        "Ask a question and let the agent choose the right tool."
    )

    st.markdown("### Available Tools")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("📚 RAG Tool\n\nRetrieves answers from the dataset PDF.")

    with c2:
        st.success("📊 Analysis Tool\n\nCalculates real CSV statistics.")

    with c3:
        st.warning("🎯 Prediction Tool\n\nUses the trained ML model.")

    st.markdown("### Ask the AI Agent")

    question = st.text_input(
        "Your question",
        placeholder="Example: What is the purchase percentage?"
    )

    if st.button("🚀 Run AI Agent", use_container_width=True):

        if not question.strip():
            st.warning("Please enter a question.")

        elif not api_key:
            st.warning("Enter your Groq API key in the sidebar.")

        else:
            with st.spinner("AI Agent is selecting a tool..."):

                try:
                    # Initialize the LLM router
                    from langchain_groq import ChatGroq
                    from langchain_core.prompts import ChatPromptTemplate
                    from langchain_core.output_parsers import StrOutputParser

                    router_llm = ChatGroq(
                        model="openai/gpt-oss-120b",
                        temperature=0,
                        api_key=api_key
                    )

                    router_prompt = ChatPromptTemplate.from_template("""
                    Select exactly one tool for the user request.

                    Return only one label:
                    RAG, ANALYSIS, or PREDICTION.

                    RAG:
                    Dataset definitions and PDF knowledge.

                    ANALYSIS:
                    Actual dataset counts, purchase percentages,
                    missing values, duplicates, summary statistics.

                    PREDICTION:
                    Predict purchase using provided encoded feature values.

                    User question: {question}
                    Tool:
                    """)

                    route_chain = (
                        router_prompt
                        | router_llm
                        | StrOutputParser()
                    )

                    selected_tool = route_chain.invoke(
                        {"question": question}
                    ).strip().upper()

                    st.markdown("### Tool Selection")

                    if "ANALYSIS" in selected_tool:
                        st.success("📊 Dataset Analysis Tool selected")

                        result = analyze_question(question)
                        st.json(result)

                    elif "PREDICTION" in selected_tool:
                        st.warning("🎯 Purchase Prediction Tool selected")

                        st.info(
                            "Prediction requires the encoded feature values "
                            "for one session. Use the Purchase Prediction "
                            "page to enter session details."
                        )

                    else:
                        st.success("📚 RAG Tool selected")

                        if not api_key:
                            st.warning("Groq API key required.")
                        else:
                            chain, retriever = load_rag(api_key)
                            result = chain.invoke(question)
                            st.markdown("### Agent Answer")
                            st.write(result)

                            with st.expander("Retrieved knowledge sources"):
                                docs = retriever.invoke(question)
                                for doc in docs:
                                    st.write(doc.page_content)

                except Exception as error:
                    st.error(f"AI Agent error: {error}")

# =====================================================
# PAGE 6: ML MODEL COMPARISON
# =====================================================

elif page == "📈 ML Model Comparison":

    show_hero(
        "📈 ML Model Comparison",
        "Compare model performance using your actual evaluation results."
    )

    import plotly.express as px

    evaluation_dir = ROOT / "evaluation"

    result_files = {
        "Model Evaluation": "evaluation_results.csv",
        "Cross Validation": "cross_validation_results.csv",
        "Hyperparameter Tuning": "tuning_results.csv",
        "Threshold Tuning": "threshold_results.csv",
    }

    selected_report = st.selectbox(
        "Choose evaluation report",
        list(result_files.keys())
    )

    report_path = evaluation_dir / result_files[selected_report]

    if not report_path.exists():
        st.error(f"File not found: {report_path.name}")

    else:
        try:
            results = pd.read_csv(report_path)

            st.markdown("### 📋 Actual Results")
            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )

            if results.empty:
                st.warning("This evaluation file has no rows.")

            else:
                numeric_columns = results.select_dtypes(
                    include="number"
                ).columns.tolist()

                if numeric_columns:
                    st.markdown("### 📊 Compare Metrics")

                    metric = st.selectbox(
                        "Select metric",
                        numeric_columns
                    )

                    category_columns = [
                        column for column in results.columns
                        if column not in numeric_columns
                    ]

                    if category_columns:
                        model_column = st.selectbox(
                            "Select model/name column",
                            category_columns
                        )

                        chart_data = results.dropna(
                            subset=[metric, model_column]
                        )

                        fig = px.bar(
                            chart_data,
                            x=model_column,
                            y=metric,
                            color=model_column,
                            text=metric,
                            title=f"{metric} Comparison",
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )

                        fig.update_layout(
                            template="plotly_white",
                            xaxis_title="Model",
                            yaxis_title=metric,
                            showlegend=False
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    else:
                        st.info(
                            "Model names column not detected. "
                            "Table is displayed above."
                        )

                else:
                    st.info(
                        "No numeric metrics detected in this report."
                    )

            st.download_button(
                "⬇️ Download Selected Results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name=report_path.name,
                mime="text/csv"
            )

        except Exception as error:
            st.error(f"Could not read evaluation report: {error}")

    st.markdown("### 🖼️ Saved Evaluation Plots")

    plot_files = [
        "confusion_matrix.png",
        "boxplot_before_iqr.png",
        "boxplot_after_iqr.png",
    ]

    available_plots = [
        evaluation_dir / name
        for name in plot_files
        if (evaluation_dir / name).exists()
    ]

    if available_plots:
        columns = st.columns(min(2, len(available_plots)))

        for index, plot_path in enumerate(available_plots):
            with columns[index % len(columns)]:
                st.image(
                    str(plot_path),
                    caption=plot_path.name,
                    use_container_width=True
                )
    else:
        st.info("No saved evaluation plots were found.")

# =====================================================
# AI INSIGHTS & RECOMMENDATIONS
# =====================================================

elif page == "💡 AI Insights & Recommendations":

    show_hero(
        "💡 AI Insights & Recommendations",
        "Understand purchase behaviour and generate evidence-based insights."
    )

    if df is None or "Revenue" not in df.columns:
        st.error("Dataset or Revenue column not found.")

    else:
        revenue = df["Revenue"].astype(str).str.lower()
        purchased = int(revenue.isin(["true", "1"]).sum())
        not_purchased = int(revenue.isin(["false", "0"]).sum())
        total = purchased + not_purchased

        purchase_rate = purchased / total * 100 if total else 0
        non_purchase_rate = not_purchased / total * 100 if total else 0

        c1, c2, c3 = st.columns(3)

        with c1:
            kpi("Total Sessions", f"{total:,}")

        with c2:
            kpi("Purchase Rate", f"{purchase_rate:.2f}%")

        with c3:
            kpi("No Purchase Rate", f"{non_purchase_rate:.2f}%")

        st.markdown("### 📊 Purchase Behaviour")

        chart_df = pd.DataFrame({
            "Outcome": ["Purchased", "No Purchase"],
            "Percentage": [purchase_rate, non_purchase_rate]
        })

        fig = px.bar(
            chart_df,
            x="Outcome",
            y="Percentage",
            color="Outcome",
            text="Percentage",
            title="Purchase vs Non-Purchase Sessions",
            color_discrete_sequence=["#00B894", "#7C4DFF"]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🔍 Data-Based Insights")

        st.write(
            f"• {purchase_rate:.2f}% of the valid sessions "
            "in this dataset ended with a purchase."
        )

        st.write(
            f"• {non_purchase_rate:.2f}% of the valid sessions "
            "did not end with a purchase."
        )

        st.write(
            "• These figures describe the dataset; they do not "
            "prove why a visitor did or did not purchase."
        )

        st.markdown("### 🤖 AI Recommendations")

        if st.button("Generate AI Insights", use_container_width=True):

            if not api_key:
                st.warning(
                    "Enter your Groq API key in the sidebar first."
                )

            else:
                try:
                    from langchain_groq import ChatGroq

                    llm = ChatGroq(
                        model="openai/gpt-oss-120b",
                        temperature=0.2,
                        api_key=api_key
                    )

                    prompt = f"""
                    You are an e-commerce data analyst.

                    Based only on these dataset statistics:
                    Total valid sessions: {total}
                    Purchase sessions: {purchased}
                    Non-purchase sessions: {not_purchased}
                    Purchase rate: {purchase_rate:.2f}%
                    Non-purchase rate: {non_purchase_rate:.2f}%

                    Provide:
                    1. Two observations supported by the data.
                    2. Three practical ideas an online shop could test.
                    3. One limitation of these statistics.

                    Do not claim these ideas are proven to increase sales.
                    Clearly label recommendations as suggestions to test.
                    """

                    with st.spinner("Generating insights..."):
                        response = llm.invoke(prompt)

                    st.markdown(response.content)

                    report = (
                        "ONLINE SHOPPERS AI INSIGHTS\n\n"
                        f"Total sessions: {total}\n"
                        f"Purchase sessions: {purchased}\n"
                        f"Non-purchase sessions: {not_purchased}\n"
                        f"Purchase rate: {purchase_rate:.2f}%\n\n"
                        f"AI Recommendations:\n{response.content}\n"
                    )

                    st.download_button(
                        "⬇️ Download Insights Report",
                        data=report,
                        file_name="shopper_ai_insights.txt",
                        mime="text/plain"
                    )

                except Exception as error:
                    st.error(f"Could not generate insights: {error}")


# =====================================================
# EVALUATION & RELIABILITY
# =====================================================

elif page == "🧪 Evaluation & Reliability":

    show_hero(
        "🧪 Evaluation & Reliability",
        "Review saved model evaluation, validation and reliability results."
    )

    evaluation_dir = ROOT / "evaluation"

    reports = {
        "Model Evaluation": "evaluation_results.csv",
        "Cross Validation": "cross_validation_results.csv",
        "Hyperparameter Tuning": "tuning_results.csv",
        "Threshold Tuning": "threshold_results.csv",
        "Prediction Results": "prediction_results.csv",
        "Outlier Report": "outlier_report.csv",
    }

    st.markdown("### 📋 Evaluation Reports")

    report_name = st.selectbox(
        "Select report",
        list(reports.keys())
    )

    report_path = evaluation_dir / reports[report_name]

    if report_path.exists():
        try:
            report_df = pd.read_csv(report_path)

            st.dataframe(
                report_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Report",
                data=report_df.to_csv(index=False).encode("utf-8"),
                file_name=report_path.name,
                mime="text/csv"
            )

            numeric_cols = report_df.select_dtypes(
                include="number"
            ).columns.tolist()

            text_cols = [
                col for col in report_df.columns
                if col not in numeric_cols
            ]

            if numeric_cols and text_cols and not report_df.empty:
                st.markdown("### 📊 Metric Visualization")

                metric = st.selectbox(
                    "Choose numeric metric",
                    numeric_cols
                )

                label_col = st.selectbox(
                    "Choose model or category column",
                    text_cols
                )

                chart_df = report_df.dropna(
                    subset=[metric, label_col]
                )

                if not chart_df.empty:
                    fig = px.bar(
                        chart_df,
                        x=label_col,
                        y=metric,
                        color=label_col,
                        text=metric,
                        title=f"{metric} by {label_col}"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

        except Exception as error:
            st.error(f"Could not read report: {error}")
    else:
        st.warning(f"Report not found: {report_path.name}")

    st.markdown("### 🖼️ Saved Evaluation Plots")

    plot_names = [
        "confusion_matrix.png",
        "boxplot_before_iqr.png",
        "boxplot_after_iqr.png",
    ]

    existing_plots = [
        evaluation_dir / name
        for name in plot_names
        if (evaluation_dir / name).exists()
    ]

    if existing_plots:
        cols = st.columns(2)

        for index, plot_path in enumerate(existing_plots):
            with cols[index % 2]:
                st.image(
                    str(plot_path),
                    caption=plot_path.name,
                    use_container_width=True
                )
    else:
        st.info("No saved evaluation plots were found.")


# =====================================================
# PAGE 5: ABOUT PROJECT
# =====================================================

elif page == "ℹ️ About Project":
    show_hero(
        "ℹ️ About the Project",
        "Autonomous Enterprise Intelligence for online shopping behaviour."
    )

    st.markdown("### 🎯 Project Objective")
    st.write(
        "Analyze online shopping sessions and understand purchase "
        "intention using dataset documentation, statistical analysis, "
        "retrieval-augmented generation and machine learning."
    )

    st.markdown("### 🧰 Technology Stack")

    cols = st.columns(3)
    technologies = [
        ("🐍 Python", "Core programming"),
        ("📊 Pandas", "Data processing"),
        ("🤖 Scikit-learn", "Machine learning"),
        ("🧠 Groq LLM", "Language model"),
        ("📚 FAISS", "Vector retrieval"),
        ("🌐 Streamlit", "Web interface")
    ]

    for index, (name, description) in enumerate(technologies):
        with cols[index % 3]:
            st.markdown(
                f"""
                <div class="kpi">
                    <div class="kpi-value" style="font-size:20px">{name}</div>
                    <div class="kpi-label">{description}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write("")

    st.markdown("### 🔄 System Workflow")

    st.markdown("""
    1. Load and clean the online shoppers dataset.
    2. Encode categorical features.
    3. Train and evaluate machine learning models.
    4. Retrieve dataset explanations from the knowledge PDF.
    5. Calculate statistics from the CSV file.
    6. Predict purchase intention using the trained model.
    7. Present results through this Streamlit application.
    """)

    st.markdown("### 📁 Project Resources")
    st.write(f"Knowledge PDF available: {'Yes' if PDF_PATH.exists() else 'No'}")
    st.write(f"Trained model available: {'Yes' if MODEL_PATH.exists() else 'No'}")
    st.write(f"Encoded data available: {'Yes' if encoded_df is not None else 'No'}")

    st.caption(
        "Educational project. Predictions may be incorrect and should not "
        "be interpreted as guaranteed customer behaviour."
    )
