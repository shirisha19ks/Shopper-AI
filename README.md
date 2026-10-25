# Autonomous Enterprise Intelligence Platform
### Online Shoppers Purchasing Intention using Machine Learning, RAG, Groq LLM, and Streamlit

## 1. Project Overview

The **Autonomous Enterprise Intelligence Platform** analyzes online shopping session data and provides dataset insights, purchase-intention predictions, and document-grounded question answering through a Streamlit web application.

The project combines:
- Data preprocessing and feature encoding
- Machine learning model training and evaluation
- Retrieval-Augmented Generation (RAG) over a dataset knowledge PDF
- Groq-hosted large language model (LLM) responses
- Dataset analysis tools
- A tool-routing AI Agent
- An interactive Streamlit dashboard

> **Note:** Describe features such as DOCX upload, full autonomous prediction-tool execution, and comprehensive RAG evaluation as completed only after they have been implemented and tested.

## 2. Problem Statement

Online shopping websites receive sessions with different browsing behaviours. Understanding which sessions result in a purchase can help analysts explore customer behaviour and evaluate purchase-intention models.

This project uses the Online Shoppers Purchasing Intention dataset to:
1. Explore browsing-session patterns.
2. Prepare and encode the dataset for machine learning.
3. Train and evaluate classification models.
4. Answer questions about dataset fields using a knowledge PDF.
5. Calculate statistics from the CSV dataset.
6. Present results in a colorful Streamlit application.

## 3. Dataset

**Dataset:** Online Shoppers Purchasing Intention

Each record represents a browsing session. The `Revenue` column is the target:
- `True`: a purchase occurred during the session.
- `False`: no purchase occurred during the session.

The dataset includes features such as:
- `Administrative` and `Administrative_Duration`
- `Informational` and `Informational_Duration`
- `ProductRelated` and `ProductRelated_Duration`
- `BounceRates`, `ExitRates`, and `PageValues`
- `SpecialDay`, `Month`, and `Weekend`
- `OperatingSystems`, `Browser`, `Region`, and `TrafficType`
- `VisitorType` and `Revenue`

The original public dataset contains 12,330 sessions and 18 columns. The project has also used a cleaned local version with 12,205 rows after cleaning. The exact row count shown by the application depends on which CSV file is loaded.

## 4. Main Features

### 4.1 Home Dashboard
- Displays dataset summary metrics.
- Shows purchase versus non-purchase session distribution.
- Displays a preview of the loaded dataset.

### 4.2 Dataset Analysis
- Shows row and column counts.
- Checks missing values and duplicate rows.
- Calculates purchase and non-purchase percentages.
- Displays data and charts and supports downloading the loaded CSV.

### 4.3 RAG Chatbot
- Loads the dataset knowledge PDF.
- Splits the PDF into text chunks.
- Creates embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Stores vectors in FAISS and retrieves relevant chunks.
- Uses a Groq LLM to generate answers grounded in retrieved context.
- Displays retrieved source text and PDF page metadata.

### 4.4 Purchase Prediction
- Loads the saved trained model from the `models` folder.
- Collects browsing-session feature inputs.
- Returns a model prediction and, where supported, a probability estimate.

**Important:** Input feature order and encoding must match the model's training pipeline. Validate this before relying on predictions. A probability is an estimate, not a guarantee that a visitor will purchase.

### 4.5 AI Agent & Tools
The agent page routes user questions to a relevant capability, such as:
- RAG for dataset definitions and knowledge-base questions
- Dataset analysis for statistics
- Purchase-prediction requests, subject to the current implemented input workflow

Verify each tool's actual execution and output before describing it as fully autonomous.

### 4.6 ML Model Comparison
- Reads saved CSV evaluation reports from the `evaluation` folder.
- Displays report tables and numeric metric charts when compatible columns are available.
- Displays saved evaluation plots.

### 4.7 AI Insights & Recommendations
- Shows observed purchase and non-purchase rates from the loaded CSV.
- Uses Groq to explain observations and suggest business experiments.
- Supports downloading the generated text report.

Recommendations are suggestions to test, not proof of causation or guaranteed improvements.

### 4.8 Evaluation & Reliability
- Displays saved evaluation, cross-validation, tuning, threshold, prediction, and outlier reports.
- Shows available confusion-matrix and outlier plots.
- Supports downloading report CSVs.

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and analysis |
| Scikit-learn | Machine learning and evaluation |
| Joblib | Loading saved models |
| Streamlit | Web application UI |
| Plotly | Interactive charts |
| LangChain | RAG chain and LLM integration |
| Groq | Hosted language model |
| Hugging Face Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| PyPDF | Reading the dataset knowledge PDF |
| python-dotenv | Loading environment variables from `.env` |

## 6. Project Structure

The structure below reflects files used or shown in the project. Adjust names if your local files differ.

```text
Autonomous_Enterprise_Intelligence/
├── app.py
├── .env
├── .gitignore
├── data/
│   ├── online_shoppers_intention.csv
│   ├── iqr_capped_online_shoppers.csv
│   └── encoded_online_shoppers.csv
├── documents/
│   └── Online_Shoppers_Dataset_Knowledge_Base_5_Pages.pdf
├── evaluation/
│   ├── evaluation_results.csv
│   ├── cross_validation_results.csv
│   ├── tuning_results.csv
│   ├── threshold_results.csv
│   ├── prediction_results.csv
│   ├── outlier_report.csv
│   ├── confusion_matrix.png
│   ├── boxplot_before_iqr.png
│   └── boxplot_after_iqr.png
├── models/
│   ├── best_model.joblib
│   └── best_tuned_model.joblib
└── src/
    ├── preprocessing.py
    ├── encoding.py
    ├── data_separation.py
    ├── imbalance.py
    ├── model_training.py
    ├── model_tuning.py
    ├── cross_validation.py
    ├── threshold_tuning.py
    ├── error_detection.py
    ├── predict.py
    ├── rag_pipeline.py
    ├── dataset_analysis.py
    └── ai_agent.py
```

Some optional files shown above may not exist in every local copy. Keep the README structure aligned with the actual files in your project.

## 7. Setup and Installation

### Prerequisites
- Python installed on your computer
- VS Code or another Python editor
- Internet access for package downloads and Groq API use
- A Groq API key for LLM features

### Step 1: Open the project folder

Open `Autonomous_Enterprise_Intelligence` in VS Code.

### Step 2: Install dependencies

Run this in the VS Code terminal:

```powershell
python -m pip install streamlit pandas plotly scikit-learn joblib langchain langchain-groq langchain-community langchain-core langchain-text-splitters langchain-huggingface sentence-transformers faiss-cpu pypdf python-dotenv
```

### Step 3: Configure the Groq API key

Set `GROQ_API_KEY` as an environment variable or put it in a local `.env` file in the project root:

```text
GROQ_API_KEY=your_actual_groq_api_key
```

Do not commit `.env` to Git, include the key in screenshots, or share it publicly. If a key has been exposed, revoke it and create a replacement.

If the app uses `os.getenv()` directly, make sure it loads `.env` with `load_dotenv()` or that the variable is set in the environment. The sidebar key field can be used if implemented in the app.

### Step 4: Check required files

Verify that the dataset CSV, knowledge PDF, and saved model files exist at the paths expected by `app.py`.

### Step 5: Start the Streamlit application

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## 8. Running Individual Scripts

Run commands from the project root. Use only commands matching files that exist in your project.

```powershell
python -m src.preprocessing
python -m src.encoding
python -m src.data_separation
python -m src.imbalance
python -m src.model_training
python -m src.model_tuning
python -m src.cross_validation
python -m src.threshold_tuning
python -m src.error_detection
python -m src.rag_pipeline
python -m src.dataset_analysis
python -m src.prediction_tool
python -m src.ai_agent
```

Some scripts may require a specific execution order or input/output files. Check each script before rerunning training or preprocessing, and avoid overwriting important outputs without a backup.

## 9. How RAG Works

1. Load the dataset knowledge PDF.
2. Split its content into smaller chunks.
3. Convert chunks into vector embeddings.
4. Store embeddings in a FAISS vector database.
5. Retrieve the most relevant chunks for a user question.
6. Send the retrieved context and question to the Groq LLM.
7. Display the answer and retrieved source information.

RAG answers depend on the content available in the knowledge PDF. Dataset statistics should be calculated from the CSV analysis tool rather than inferred from the PDF.

## 10. AI Agent Workflow

1. The user submits a question.
2. The agent/router determines which capability is relevant.
3. The selected tool performs retrieval, analysis, or the supported prediction workflow.
4. The result is returned to the user.
5. Errors and unsupported requests should be handled clearly.

For reliable results, test tool routing with questions for every tool. A tool label alone is not proof that the tool executed successfully.

## 11. Testing Checklist

- [ ] Streamlit starts without an exception.
- [ ] All navigation pages open.
- [ ] The correct CSV is loaded.
- [ ] Purchase percentage matches the CSV's `Revenue` counts.
- [ ] Missing-value and duplicate counts match the selected dataset.
- [ ] RAG questions return relevant PDF-grounded answers.
- [ ] Retrieved source pages are displayed correctly.
- [ ] The model loads and prediction input features match training.
- [ ] Evaluation CSV reports and plots load.
- [ ] AI insights handle missing API keys and API errors.
- [ ] CSV and report download buttons work.
- [ ] No API key is exposed in source control or screenshots.

## 12. Limitations

- Model performance depends on data quality, train/test separation, preprocessing, and evaluation method.
- Purchase-intention predictions are estimates, not guaranteed outcomes.
- RAG can return incomplete answers if relevant information is absent from the knowledge PDF.
- AI-generated recommendation quality depends on the evidence provided to the LLM.
- Do not claim DOCX ingestion, fully autonomous prediction, hybrid retrieval, reranking, or production deployment unless those features are implemented and tested.

## 13. Future Enhancements

- Add DOCX document ingestion and upload support.
- Improve source citations and retrieval evaluation.
- Integrate the prediction tool directly into the AI Agent workflow.
- Add a unified model-metrics dashboard with accuracy, precision, recall, F1-score, and confusion matrix.
- Add automated RAG and tool-routing evaluation.
- Add PDF/CSV report export and a project architecture diagram.
- Add automated tests and dependency version pinning.
- Consider FastAPI, authentication, and container deployment if required by the assignment.

## 14. Conclusion

The project brings together machine learning, dataset analytics, retrieval-augmented generation, a Groq LLM, and a Streamlit interface to explore online shopping purchase intention. Before submission, verify all paths, metrics, tool executions, and claims against the actual local implementation.

---
**Project:** Autonomous Enterprise Intelligence Platform  
**Dataset:** Online Shoppers Purchasing Intention  
**Interface:** Streamlit
