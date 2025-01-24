import os
import json
import argparse

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from dotenv import load_dotenv
import logging
from langchain_core.documents import Document
logging.basicConfig(level=logging.DEBUG)
load_dotenv()
# Print to verify
print("API Key:", bool(os.getenv("UNSTRUCTURED_API_KEY")))  # Should print True if set
print("API URL:", os.getenv("UNSTRUCTURED_API_URL"))  # Should show the URL
print("Input Dir:", os.getenv("LOCAL_FILE_INPUT_DIR"))  # Should show your PDF directory
print("Output Dir:", os.getenv("LOCAL_FILE_OUTPUT_DIR"))  # Should show output directory

# Unstructured API
from unstructured_ingest.v2.pipeline.pipeline import Pipeline
from unstructured_ingest.v2.interfaces import ProcessorConfig
from unstructured_ingest.v2.processes.connectors.local import (
    LocalIndexerConfig,
    LocalDownloaderConfig,
    LocalConnectionConfig,
    LocalUploaderConfig
)
from unstructured_ingest.v2.processes.partitioner import PartitionerConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reprocess', action='store_true', help='Force reprocessing of PDFs')
    args = parser.parse_args()

    output_dir = os.getenv("LOCAL_FILE_OUTPUT_DIR")
    
    # Check if output directory has processed files
    has_processed_files = False
    if os.path.exists(output_dir):
        json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        has_processed_files = len(json_files) > 0
    
    # Run pipeline if no files exist or reprocess flag is set
    if not has_processed_files or args.reprocess:
        if args.reprocess:
            print("Reprocessing PDFs...")
        else:
            print("No processed files found. Processing PDFs...")
        Pipeline.from_configs(
            context=ProcessorConfig(),
            indexer_config=LocalIndexerConfig(input_path=os.getenv("LOCAL_FILE_INPUT_DIR")),
            downloader_config=LocalDownloaderConfig(),
            source_connection_config=LocalConnectionConfig(),
            partitioner_config=PartitionerConfig(
                partition_by_api=True,
                api_key=os.getenv("UNSTRUCTURED_API_KEY"),
                partition_endpoint=os.getenv("UNSTRUCTURED_API_URL"),
                strategy="hi_res",
                additional_partition_args={
                    "split_pdf_page": True,
                    "split_pdf_allow_failed": True,
                    "split_pdf_concurrency_level": 15
                }
            ),
            uploader_config=LocalUploaderConfig(output_dir=output_dir)
        ).run()
        print("PDF processing complete.")
    else:
        print("Found existing processed files, skipping PDF processing...")

    # Continue with Neo4j processing...
    print("Loading processed chunks into Neo4j...")

llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'), 
    model_name="gpt-3.5-turbo"
)

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
    )

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

# Gather the documents
DOCS_PATH = os.path.join(os.path.dirname(__file__), "../data/course/sethspdfs")

# Everything below this is how I define and seed the knowledge graph using the chunks extracted from the documents. in this case it's a pdf file type
# i want to do agentic chunking based on 'propostion based retrieval' and then do the vectorization and then the graph transformation.
# to do so I will us langhub to import a prompt template for the chunking and then use the vectorization and then the graph transformation.
# idk yet what this entails for tools directory or cypher queries
# if you want to delete the graph and start fresh:  MATCH (n) DETACH DELETE n


# schema for the graph
doc_transformer = LLMGraphTransformer(
    llm=llm,
    #node_properties=["name", "description"],
    relationship_properties=["description"]
)

# After the Pipeline.run(), modify this section:
processed_docs_path = os.getenv("LOCAL_FILE_OUTPUT_DIR")
docs = []
for root, _, files in os.walk(processed_docs_path):
    for file in files:
        if file.endswith('.json'):
            with open(os.path.join(root, file), 'r') as f:
                content = json.load(f)
                # Each element in content is a chunk
                for idx, element in enumerate(content):
                    if 'text' in element:
                        chunk_id = f"{file}.{idx}"
                        chunk_embedding = embedding_provider.embed_query(element['text'])
                        
                        properties = {
                            "filename": file,
                            "chunk_id": chunk_id,
                            "text": element['text'],
                            "embedding": chunk_embedding
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

                        # Create a proper Document object
                        doc = Document(
                            page_content=element['text'],
                            metadata={'source': file, 'page': idx}
                        )

                        # Generate the entities and relationships from the chunk
                        graph_docs = doc_transformer.convert_to_graph_documents([doc])

                        # Map the entities in the graph documents to the chunk node
                        for graph_doc in graph_docs:
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

# Vectorize the data... Create the vector index
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
    }};""")
