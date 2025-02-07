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
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser

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
    Generates a response to a question using both a vector store and Tavily web search for retrieval augmentation.

    Args:
        question (str): The question to be answered.
        vectorstore: The vector store to be used for retrieving relevant documents.

    Returns:
        str: The generated response to the question.
    """

    #Set API keys
    groq = "ENTER API KEY"
    tavily = "ENTER API KEY"
    os.environ["GROQ_API_KEY"] = groq.strip()
    os.environ["TAVILY_API_KEY"] = tavily.strip()

        # Initialize the ChatGroq model
    llm = ChatGroq(
        #model="deepseek-r1-distill-llama-70b",
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    retriever = vectorstore.as_retriever()
    tavily_search = TavilySearchResults(api_key=os.environ["TAVILY_API_KEY"])

    # Define the improved prompt template
    prompt = PromptTemplate(
        template="""
        You are an expert assistant with deep knowledge in various domains. Your task is to answer the question using the most relevant information from the provided **vectorstore context**. The **web search context** will only be used to supplement the vectorstore context when necessary. Ensure that the answer stays **relevant to the main vectorstore topic**.
        
        **Instructions:**
        1. **Understand the Question**: Decode the user's query clearly and identify its intent.
        2. **Use Context**: 
           - Prioritize and base your answer on the **vectorstore context**.
           - Use the **web search context** only to supplement the vectorstore context when it provides additional useful information.
        3. **Concise & Accurate Answer**: Provide a clear, concise, and accurate response. Ensure factual correctness.
        4. **If Conflicting Information Exists**: Prioritize information from the **vectorstore context**. If both sources provide conflicting info, give preference to the most authoritative or reliable source.
        5. **Avoid Redundancy**: There's no need to mention where the information was sourced from (whether it was from the web search or vector store).
        6. **Focus on the Main Topic**: ONLY respond to topics that are **directly related to the vectorstore context**. If the question does not align with the vectorstore context, JUST SAY: This topic is irrelevant. 
        
        **Question:** {question}
        **Vector Store Context:** {vectorstore context}
        **Web Search Context:** {web search context}
        **Answer:**""",
        input_variables=["question", "context"],
    )

    # Create the RAG chain
    rag_chain = prompt | llm | StrOutputParser()

    # Retrieve relevant documents from vectorstore
    docs = retriever.get_relevant_documents(question)
    vector_context = "\n\n".join([doc.page_content for doc in docs])

    # Perform Tavily Web Search
    web_results = tavily_search.run(question)
    web_context = "\n\n".join([result["content"] for result in web_results[:3]]) if web_results else ""

    # Combine both sources of information
    combined_context = f"**Vectorstore Context:**\n{vector_context}\n\n**Web Search Context:**\n{web_context}"

    # Generate the response
    response = rag_chain.invoke({"question": question, "vectorstore context": combined_context, "web search context":web_context})

    return str(response)