# Datheon Python + Generative AI & LangChain Internship

My completed project submissions for the **Datheon Python + Generative AI & LangChain Internship** (`DT-2026-18X3Y`).

---

## Projects

### 1. CV / Resume Project (`1st project/main.py`)
* **What it is:** A Python script that loads a sample CV/resume (`sample_resume_for_testing.txt`), creates vector embeddings with Chroma, and supports querying/searching the CV content.

### 2. TechMart Chatbot Project (`2nd project/main.py`)
* **What it is:** A Python RAG chatbot built around `TechMart Customer Support Guidebook.pdf` using LangChain and Chroma to answer customer support questions from the store guidebook.

### 3. TechMart Streamlit App (`4th project/app.py`)
* **What it is:** A Streamlit web interface wrapping the TechMart RAG assistant ("Alex") with chat history, follow-up query handling, and real-time streaming responses.

---

## Folder Structure

```text
gen ai internship/
├── 1st project/
│   ├── data/
│   │   ├── sample_resume_for_testing.txt
│   │   └── store/
│   └── main.py
├── 2nd project/
│   ├── data/
│   │   └── TechMart Customer Support Guidebook.pdf
│   └── main.py
├── 4th project/
│   ├── data/
│   │   └── TechMart Customer Support Guidebook.pdf
│   ├── store_db/
│   └── app.py
├── requirements.txt
├── .gitignore
└── README.md
How to Run Locally
Install dependencies:

Bash
pip install -r requirements.txt
Add your .env file in the root folder:

Code snippet
GOOGLE_API_KEY=your_api_key_here
Run a project:

CV project: python "1st project/main.py"

TechMart chatbot: python "2nd project/main.py"

Streamlit app: streamlit run "4th project/app.py"


Save this to `README.md`, stage and commit it with Git, and push to `origin main`.
