import click
import os
import openai
import pinecone
import fitz #PyMuPDF
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

@click.group()
def main():
    pass

@main.command()
@click.option(
    "-s",
    "--source",
    "source",
    required=True,
    help="Source dir for docs",
)
@click.option(
    "-d",
    "--destination",
    "destination",
    required=True,
    help="destination dir for keeping chunks",
)
@click.option(
    "-c",
    "--chunk-size",
    "chunk_size",
    required=False,
    default=100,
    type=click.INT,
    help="chunks size",
)
@click.option(
    "-ov",
    "--overlap",
    "overlap",
    required=False,
    default=20,
    type=click.INT,
    help="overlap size between the chunks",
)
def create_chunks(source, destination, chunk_size, overlap):
    loader = DirectoryLoader(source)
    documents = loader.load()
    print(len(documents))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = text_splitter.split_documents(documents)
    print(len(docs))
    print(docs[0].page_content)

    for i, doc in enumerate(docs):
        chunk_filename = os.path.join(destination, f"chunk_{i}.txt")
        with open(chunk_filename, 'w') as f:
            f.write(doc.page_content)

    print("Chunks saved to destination directory:", destination)


@main.command()
@click.option(
    "-s",
    "--source",
    "source",
    required=True,
    help="Source dir for docs",
)
def create_embeddings(source):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    pinecone.init(
        api_key=pinecone_api_key,
        environment="gcp-starter"
    )

    index_name = "chatbot-demo"
    index = Pinecone(index_name)

    for filename in os.listdir(source):
        if filename.endswith(".txt"):
            with open(os.path.join(source, filename), 'r') as f:
                content = f.read()
                # Assuming content is a string of text, you need to embed it.
                embedding = embeddings.embed_query(content)
                # Add the embedding to the index
                index.upsert(str(filename), embedding)
        elif filename.endswith(".pdf"):
            with fitz.open(os.path.join(source, filename)) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                # Assuming text is a string of text, you need to embed it.
                embedding = embeddings.embed_query(text)
                # Add the embedding to the index
                index.upsert(str(filename), embedding)

    print("Embeddings created and indexed in Pinecone.")


def get_similar_docs(query, k=2, score=False):
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment="gcp-starter"
    )
    index_name = "chatbot-demo"
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    index = Pinecone(index_name)
    if score:
        similar_docs = index.similarity_search(query, k=k)
    else:
        similar_docs = index.similarity_search_with_score(query, k=k)
    return similar_docs

@main.command()
def query():
    query = input("Enter the question!")
    similar_docs = get_similar_docs(query, score=True)
    print(similar_docs)

    model_name = "gpt-3.5-turbo"
    llm = OpenAI(model_name=model_name)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=similar_docs, question=query)
    print(answer)

main.add_command(create_chunks)
main.add_command(create_embeddings)
main.add_command(query)

if __name__ == "__main__":
    main()
