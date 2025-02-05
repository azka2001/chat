from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os


def get_all_links(url: str) -> list:
    """
    Fetches all unique links from a webpage and returns the first 10 links.

    Args:
        url (str): The URL of the webpage to fetch links from.

    Returns:
        list: A list of the first 10 unique links found on the webpage.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    domain = urlparse(url).netloc
    links = set()

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(url, href)
        parsed_url = urlparse(full_url)
        if parsed_url.netloc == domain:
            links.add(full_url)

    return list(links)  # Return the first 10 links from the list


def create_vector_store(url: str):
    """
    Creates a vector store from the content of the first 10 links found on a webpage.

    Args:
        url (str): The URL of the webpage to fetch links and content from.

    Returns:
        Chroma: A Chroma vector store containing the text from the documents.
    """
    links = get_all_links(url)
    documents = []

    for link in links:
        loader = WebBaseLoader(link)
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vectorstore = None
    vectorstore = Chroma.from_documents(texts, embeddings)

    return vectorstore


def generate_response(question: str, vectorstore):
    """
    Generates a response to a question using the provided vector store.

    Args:
        question (str): The question to be answered.
        vectorstore: The vector store to be used for retrieving relevant documents.

    Returns:
        str: The generated response to the question.
    """
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # Set up the API key for the ChatGroq model
    key = "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"
    os.environ["GROQ_API_KEY"] = key.strip()

    # Initialize the ChatGroq model
    llm = ChatGroq(
        #model="llama3-groq-8b-8192-tool-use-preview",
        model="deepseek-r1-distill-llama-70b",
        #model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    retriever = vectorstore.as_retriever()

    # Define the prompt template
    prompt = PromptTemplate(
        template="""You are an expert assistant with deep knowledge in various domains. Your task is to accurately answer the question based on the provided context extracted from relevant documents. Follow these instructions:

            1. **Understand the Question**: Carefully decode the user's question to grasp its intent.

            2. **Use Context**: Refer to the provided context from the retrieved documents to form your answer. Ensure that your response is directly supported by the context.

            3. **Concise Answer**: Provide a clear, concise answer in three sentences or less. If the context does not contain enough information to answer the question, state that you don't know the answer.

            4. **Avoid Assumptions**: Do not infer or assume information that is not present in the context. Only rely on what is explicitly provided.

            Question: {question}
            Context: {context}
            Answer:""",
        input_variables=["question", "context"],
    )

    # Create the RAG chain
    rag_chain = prompt | llm | StrOutputParser()

    # Retrieve relevant documents and generate response
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    response = rag_chain.invoke({"question": question, "context": context})

    return response
