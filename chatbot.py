import os
import os
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
directory = "/home/priyanka/Documents/Chatbot_Project/chatbot.py"
os.environ["OPENAI_API_KEY"] = "Your_Key"

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
len(documents)
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))
print(docs[0].page_content)
embeddings = OpenAIEmbeddings(model_name="ada")

query_result = embeddings.embed_query("Hello world")
len(query_result)
pinecone.init(
    api_key="Your_Key",
    environment="gcp-starter"
)

index_name = "chatbot-demo"

index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
def get_similiar_docs(query, k=2, score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query, k=k)
  else:
    similar_docs = index.similarity_search(query, k=k)
  return similar_docs

query  = input("Enter the question!")
similar_docs = get_similiar_docs(query, score = True)
similar_docs
# model_name = "text-davinci-003"
model_name = "gpt-3.5-turbo"
#model_name = "gpt-4"
llm = OpenAI(model_name=model_name)

chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
  similar_docs = get_similiar_docs(query)
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer
answer = get_answer(query)
print(answer)


