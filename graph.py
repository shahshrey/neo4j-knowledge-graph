from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
import os

def load_documents(docs_path):
    """Load documents from the specified path"""
    loader = DirectoryLoader(docs_path, glob="**/*.docx", loader_cls=Docx2txtLoader)
    return loader.load()

def create_chunks(docs, chunk_size=300, chunk_overlap=100):
    """Split documents into chunks"""
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_documents(docs)

def initialize_services():
    """Initialize OpenAI and Neo4j services"""
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
    
    return embedding_provider, graph, llm

def process_chunk(chunk, embedding_provider, graph):
    """Process a single document chunk"""
    filename = os.path.basename(chunk.metadata["source"])
    page_num = chunk.metadata.get("page", 1)
    chunk_id = f"{filename}.{page_num}"
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)
    
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "textEmbedding": chunk_embedding,
        "embedding": chunk_embedding
    }
    
    # Add Document and Chunk nodes to graph
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

def create_vector_index(graph):
    """Create vector index for similarity search"""
    graph.query("""
        CREATE VECTOR INDEX `vector`
        FOR (c: Chunk) ON (c.textEmbedding)
        OPTIONS {indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
        }};""")

def process_entities(chunk, graph, doc_transformer):
    """Process entities in a chunk and create relationships"""
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])
    
    for graph_doc in graph_docs:
        page_num = chunk.metadata.get("page", 1)
        chunk_id = f"{os.path.basename(chunk.metadata['source'])}.{page_num}"
        chunk_node = Node(id=chunk_id, type="Chunk")
        
        # Create relationships between chunk and entities
        for node in graph_doc.nodes:
            graph_doc.relationships.append(
                Relationship(source=chunk_node, target=node, type="HAS_ENTITY")
            )
        
        graph.add_graph_documents(graph_docs)

def main():
    DOCX_PATH = "/docs"
    docs = load_documents(DOCX_PATH)
    chunks = create_chunks(docs)
    
    embedding_provider, graph, llm = initialize_services()
    doc_transformer = LLMGraphTransformer(llm=llm)
    
    # Process chunks and create graph structure
    for chunk in chunks:
        process_chunk(chunk, embedding_provider, graph)
    
    create_vector_index(graph)
    
    # Process entities and relationships
    for chunk in chunks:
        process_entities(chunk, graph, doc_transformer)

if __name__ == "__main__":
    main()