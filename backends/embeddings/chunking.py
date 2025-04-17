from nanoid import generate as generate_uuid
from typing import List, Tuple
from llama_index.core import Document
from llama_index.core.schema import IndexNode, TextNode
from core import common


# Chunks are created from each document and will inherit their metadata
def chunks_from_documents(
    documents: List[Document], parsed_nodes: List[TextNode], source_record: dict
) -> Tuple[List[str], List[IndexNode]]:
    chunk_nodes = []
    chunks_ids = []

    for doc in documents:
        source_id = source_record.get("id")
        # Set metadata on each chunk node
        for chunk_ind, parsed_node in enumerate(parsed_nodes):
            # Create metadata for chunk
            chunk_metadata = dict(
                sourceId=source_id,
                order=chunk_ind,
                # description="", # @TODO Ai generate based on chunk's text content
                # tags="", # @TODO Ai generate based on chunk's description
                # name="", # @TODO Ai generate based on chunk's text description above
            )
            # Set metadatas
            excluded_llm_metadata_keys = doc.excluded_llm_metadata_keys
            excluded_llm_metadata_keys.append("order")
            excluded_embed_metadata_keys = doc.excluded_embed_metadata_keys
            excluded_embed_metadata_keys.append("order")
            # Create chunk
            chunk_node = IndexNode(
                id_=generate_uuid(),
                text=parsed_node.text or "None",
                index_id=str(source_id),
                metadata=chunk_metadata,
            )
            # Tell query engine to ignore these metadata keys
            chunk_node.excluded_llm_metadata_keys = excluded_llm_metadata_keys
            chunk_node.excluded_embed_metadata_keys = excluded_embed_metadata_keys
            # Once your metadata is converted into a string using metadata_seperator
            # and metadata_template, the metadata_templates controls what that metadata
            # looks like when joined with the text content
            chunk_node.metadata_seperator = "::"
            chunk_node.metadata_template = "{key}=>{value}"
            chunk_node.text_template = (
                "Metadata: {metadata_str}\n-----\nContent: {content}"
            )
            # Return chunk `IndexNode`
            chunk_nodes.append(chunk_node)
            chunks_ids.append(chunk_node.node_id)  # or id_
    print(f"{common.PRNT_EMBED} Added {len(chunk_nodes)} chunks to collection")
    return [chunks_ids, chunk_nodes]


# Create a document record for a Collection to track
# @TODO Perhaps we can do the same with a `docstore` ?
def create_source_record(document: Document) -> dict:
    doc_metadata = document.metadata
    total_pages = doc_metadata.get("total_pages")
    modified_last = (
        doc_metadata.get("last_modified_date")
        or doc_metadata.get("modified_last")
        or ""
    )
    # Create an object to store metadata
    source_record = dict(
        **doc_metadata,
        id=document.id_,  # @TODO do we need this?
        modified_last=modified_last,
        chunkIds=[],  # filled in after chunks created
    )
    if total_pages:
        source_record.update(totalPages=total_pages)
    # Return result
    print(f"{common.PRNT_EMBED} Created document record:\n{source_record}", flush=True)
    return source_record
