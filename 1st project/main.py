from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, ValidationError


class ResumeAnalysis(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate from the resume.")
    key_skills: list[str] = Field(description="Technical and soft skills listed in the resume.")
    strengths: list[str] = Field(description="Notable projects, coursework, or experience strengths.")
    gaps_or_suggestions: list[str] = Field(description="Constructive improvements or missing details.")


# 1. Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# 2. Load & split text
loader_text = TextLoader(
    file_path="c:/gen ai internship/1st project/data/sample_resume_for_testing.txt",
    encoding="utf-8",
)
doc = loader_text.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
splitted_text = splitter.split_documents(doc)

# 3. Create embeddings & vector store
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
store = Chroma.from_documents(
    documents=splitted_text,
    persist_directory="c:/gen ai internship/1st project/data/store",
    embedding=embedding,
)

# 4. Retrieve relevant chunks
query = "candidate name, education, technical skills, projects, and experience"
retrieved_docs = store.similarity_search(query, k=4)
context = "\n\n".join(chunk.page_content for chunk in retrieved_docs)

# 5. Build prompt & bind Pydantic schema
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert technical recruiter and resume reviewer. "
        "Analyze the resume context below and extract the requested structured fields. "
        "Base your analysis strictly on the provided context.\n\n"
        "Resume Context:\n{context}",
    ),
    ("human", "{question}"),
])

structured_llm = llm.with_structured_output(ResumeAnalysis)
chain = prompt | structured_llm

# 6. Run the chain & print validated output
try:
    analysis: ResumeAnalysis = chain.invoke({
        "context": context,
        "question": "Analyze this candidate's resume and highlight strengths and areas to improve.",
    })
    print(analysis.model_dump_json(indent=2))
except ValidationError as e:
    print("Validation failed:", e)