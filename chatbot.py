import click
import os
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

openai_api_key = os.environ["OPENAI_API_KEY"]

@click.command()
@click.option(
    "-s",
    "--source",
    "source",
    required=True,
    help="The path where docs are sourced from for ingestion",
)
@click.option(
    "-d",
    "--destination",
    "destination",
    required=True,
    help="The path where chunks are created",
)
@click.option(
    "-c",
    "--chunk-size",
    "chunk_size",
    required=False,
    default=3000,
    type=click.INT,
    help="Chunk size",
)
@click.option(
    "-ov",
    "--overlap",
    "overlap",
    required=False,
    default=100,
    type=click.INT,
    help="Overlap size to be used for the text splitter",
)
def main(source, destination, chunk_size, overlap):
    documents = load_docs(source)
    print(len(documents))

    docs = split_docs(documents, chunk_size, overlap)
    print(len(docs))
    print(docs[0].page_content)

    embeddings = OpenAIEmbeddings(model_name="ada")

    query_result = embeddings.embed_query("Hello world")
    print(len(query_result))

    pinecone.init(
        api_key=os.environ["Your_Key"],
        environment="gcp-starter"
    )

    index_name = "chatbot-demo"
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

    query = input("Enter the question!")
    similar_docs = get_similiar_docs(query, score=True)
    print(similar_docs)

    model_name = "gpt-3.5-turbo"
    llm = OpenAI(model_name=model_name)
    chain = load_qa_chain(llm, chain_type="stuff")

    answer = get_answer(query)
    print(answer)

if __name__ == "__main__":
    main()
