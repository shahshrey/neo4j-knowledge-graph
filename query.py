import os
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Initialize environment and services
load_dotenv()

llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), temperature=0)
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

# Configure Cypher query generation
prompt = PromptTemplate(
    template="""Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Only include the generated Cypher statement in your response.

Always use case insensitive search when matching strings.

Schema:
{schema}

The question is:
{question}""",
    input_variables=["schema", "question"]
)

# Set up query chain
cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph, 
    cypher_prompt=prompt,
    verbose=True
)

def query_graph(question: str) -> dict:
    """Execute a natural language query against the graph database"""
    return cypher_chain.invoke({
        "query": question,
        "schema": graph.schema
    })

# Interactive query loop
while True:
    question = input("Input your query: ")
    print(query_graph(question))
