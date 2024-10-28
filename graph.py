from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
import json
from langchain_community.graphs.graph_document import Node, Relationship

import os


DOCX_PATH = "/docs"

loader = DirectoryLoader(DOCX_PATH, glob="**/*.docx", loader_cls=Docx2txtLoader)

# Load both DOCX and PDF files
docs_docx = loader.load()

# Combine both document sets
docs = docs_docx

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
)

chunks = text_splitter.split_documents(docs)

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
    )

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'), 
    model_name="gpt-4o"
)

doc_transformer = LLMGraphTransformer(
    llm=llm,
    )

for chunk in chunks:
    # Extract the filename
    filename = os.path.basename(chunk.metadata["source"])

    # Create a unique identifier for the chunk    
    # Handle both PDF and DOCX cases for page metadata
    page_num = chunk.metadata.get("page", 1)  # Default to 1 if page not found
    chunk_id = f"{filename}.{page_num}"

    # Embed the chunk
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    # Add the Document and Chunk nodes to the graph
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "textEmbedding": chunk_embedding,
        "embedding": chunk_embedding  # Added this line to fix the missing parameter error
    }

    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """, 
        properties
    )

# Create the vector index
graph.query("""
    CREATE VECTOR INDEX `vector`
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
    }};""")


for chunk in chunks:
    # Generate the entities and relationships from the chunk
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])
    for graph_doc in graph_docs:
        # Get the page number, defaulting to 1 if not found
        page_num = chunk.metadata.get("page", 1)
        chunk_id = f"{os.path.basename(chunk.metadata['source'])}.{page_num}"
        
        chunk_node = Node(
            id=chunk_id,
            type="Chunk"
        )

        for node in graph_doc.nodes:
            graph_doc.relationships.append(
                Relationship(
                    source=chunk_node,
                    target=node, 
                    type="HAS_ENTITY"
                    )
                )

    # add the graph documents to the graph
    graph.add_graph_documents(graph_docs)    