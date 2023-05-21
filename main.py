import os
import requests
import xml.etree.ElementTree as ET
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI

os.environ["OPENAI_API_KEY"] = "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"

# Define constants for file paths
DOCS_FILE_PATH = "docs.pkl"
VECTOR_STORE_FILE_PATH = "faiss_store_openai.pkl"

def get_sitemap(url):
    """Get sitemap from the given URL."""
    response = requests.get(url)
    root = ET.fromstring(response.content)
    urls = [element.text for element in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    return urls

def store_data(data, file_path):
    """Store data as a pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_data(file_path):
    """Load data from a pickle file."""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def split_and_process_docs(url):
    """Split and process documents from a given URL."""
    loader = UnstructuredURLLoader(urls=[url])
    data = loader.load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(data)

def process_all_docs(urls):
    """Process all documents from a list of URLs."""
    docs = []
    for url in urls:
        try:
            split_docs = split_and_process_docs(url)
            docs.extend(split_docs)
            print(f"URL Loaded and Splitted: {url}")
        except Exception as e:
            print(f"Error loading or splitting URL: {url}, error: {e}")
    return docs

def main():
    """Main function to run the document processing pipeline."""
    url = "https://apim.docs.wso2.com/en/latest/sitemap.xml"
    urls = get_sitemap(url)
    print(f"Number of URLs: {len(urls)}")

    docs = load_data(DOCS_FILE_PATH)
    if not docs:
        docs = process_all_docs(urls)
        store_data(docs, DOCS_FILE_PATH)

    print(f"Number of Documents: {len(docs)}")

    embeddings = OpenAIEmbeddings()

    vector_store = load_data(VECTOR_STORE_FILE_PATH)
    if not vector_store:
        vector_store = FAISS.from_documents(docs, embeddings)
        store_data(vector_store, VECTOR_STORE_FILE_PATH)
    print(vector_store)

    llm = OpenAI(temperature=0)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    output = chain({"question": "in wso2 micro Integrator, how to change the payload of the incoming request "}, return_only_outputs=True)
    print(output)

if __name__ == "__main__":
    main()
