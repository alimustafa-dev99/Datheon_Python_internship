from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate 
from langchain_chroma import Chroma 
from dotenv import load_dotenv

load_dotenv()
llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

#loader
loader=PyPDFLoader(file_path="c:/gen ai internship/2nd project/data/TechMart Customer Support Guidebook.pdf")
docs=loader.load()

# splitted data
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splitted_doc=splitter.split_documents(docs)


#embeddings
embeddings=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

store=Chroma.from_documents(
    documents=splitted_doc,
    embedding=embeddings,
    persist_directory="c:/gen ai internship/2nd project/data/store_db"
)


prompt=ChatPromptTemplate.from_messages([
        (
            "system",
             """
You are Alex, the official AI customer support assistant for TechMart, an online electronics store.

Ground your answers strictly in the TechMart context provided below:
<context>
{context}
</context>

Guidelines:
- Answer concisely, clearly, and politely to help the customer.
- Only state product names, prices, discounts, categories, order details, or store policies that appear in the context above.
- If the answer is not found in the context, say clearly: "I don't know based on the available TechMart information." Do not guess or invent details.
- When listing multiple products, use short bullet points with regular price, discounted price/savings, and category if available in the context.
- Use prior conversation history only to understand follow-up questions; still verify all factual claims against the context.

                      """
        ),
        (
            "user",
            "{query}"
        )
        
    ]

    )
 





while True:
    query=input("ask anything")
    if query.lower() in ["quit","exit","bye"]:
        break
    search=store.similarity_search(query,k=3)
    context="\n\n".join(doc.page_content for doc in search)

    res=llm.stream(prompt.format_messages(query=query,context=context))
    for chunk in res:
        print(chunk.text,end="",flush=True)


