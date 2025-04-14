import os
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core.readers import SimpleDirectoryReader, Document
from llama_index.readers.file import (
    PyMuPDFReader,
    DocxReader,
    CSVReader,
    RTFReader,
    UnstructuredReader,
    XMLReader,
    PptxReader,
    ImageReader,
    VideoAudioReader,
)
from llama_parse import LlamaParse
from core import classes, common

###########
# METHODS #
###########


def create_source_document(text: str, source_id: str, metadata: dict):
    # Create metadata
    document_metadata = {
        **metadata,
        "sourceId": source_id,  # track this document
    }
    # Create source document
    source_doc = Document(
        id_=source_id,
        text=text,
        metadata=document_metadata,
    )
    return source_doc


def set_ignored_metadata(source_document: Document, ignore_metadata: dict):
    # Tell query engine to ignore these metadata keys
    source_document.excluded_llm_metadata_keys.extend(["sourceId", "embedder"])
    source_document.excluded_embed_metadata_keys.extend(["sourceId", "embedder"])
    # Insert user submitted metadata into document
    source_document.metadata.update(ignore_metadata)
    source_document.excluded_llm_metadata_keys.extend(ignore_metadata)
    source_document.excluded_embed_metadata_keys.extend(ignore_metadata)
    return source_document


###########
# LOADERS #
###########


def simple_file_loader(
    sources: List[str],
    source_id,
    source_metadata: dict,
) -> List[Document]:
    documents = []
    reader = SimpleDirectoryReader(input_files=sources)
    # Process each files nodes as they load
    for loaded_nodes in reader.iter_data():
        # Build a document from loaded file
        # Reader splits into text nodes which we use to rebuild text content
        document_node = loaded_nodes[0]
        doc_text = [d.get_content() for d in loaded_nodes]
        # Create document node
        source_doc = create_source_document(
            text="".join(doc_text),
            source_id=source_id,
            metadata=document_node.metadata,
        )
        # Set ignored metadata
        source_doc = set_ignored_metadata(
            source_document=source_doc, ignore_metadata=source_metadata
        )
        # Return source `Document`
        documents.append(source_doc)
    return documents


# https://github.com/run-llama/llama_index/tree/4f967b839f7e986f178f24cae2038224eb33147f/llama-index-integrations/readers/llama-index-readers-smart-pdf-loader
def smart_pdf_loader():
    return []


# https://github.com/run-llama/llama_index/tree/4f967b839f7e986f178f24cae2038224eb33147f/llama-index-integrations/readers/llama-index-readers-nougat-ocr
def scientific_pdf_loader():
    return []


# https://github.com/run-llama/llama_index/tree/40913847ba47d435b40b7fac3ae83eba89b56bb9/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/pymu_pdf
def simple_pdf_loader(
    sources: str,
    source_id: str,
    source_metadata: dict,
    make_metadata=True,
) -> List[Document]:
    document_results: List[Document] = []
    reader = PyMuPDFReader()
    for path in sources:
        documents = reader.load_data(file_path=path, metadata=make_metadata)
        # Create document node from file source
        doc_text = [d.get_content() for d in documents]
        # Combine all nodes into one document
        source_doc = create_source_document(
            text="".join(doc_text),
            source_id=source_id,
            metadata={
                "total_pages": documents[0].metadata.get("total_pages"),
                # "page": documents[0].metadata.get("source"), # not rly useful
                **source_metadata,
            },
        )
        # Set ignored metadata
        source_doc = set_ignored_metadata(
            source_document=source_doc,
            ignore_metadata=source_metadata,
        )
        document_results.append(source_doc)
    return document_results


# https://github.com/run-llama/llama_index/tree/40913847ba47d435b40b7fac3ae83eba89b56bb9/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs
def ms_doc_loader(
    sources: str,
    source_id: str,
    source_metadata: dict,
):
    document_results: List[Document] = []
    reader = DocxReader()
    for path in sources:
        documents = reader.load_data(file=Path(path))
        # Create document node from file source
        doc_text = [d.get_content() for d in documents]
        # Combine all nodes into one document
        source_doc = create_source_document(
            text="".join(doc_text),
            source_id=source_id,
            metadata=source_metadata,
        )
        # Set ignored metadata
        source_doc = set_ignored_metadata(
            source_document=source_doc,
            ignore_metadata=source_metadata,
        )
        document_results.append(source_doc)
    return document_results


# https://github.com/run-llama/llama_index/tree/40913847ba47d435b40b7fac3ae83eba89b56bb9/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/rtf
def rtf_loader(
    sources: str,
    source_id: str,
    source_metadata: dict,
):
    document_results: List[Document] = []
    reader = RTFReader()
    for path in sources:
        documents = reader.load_data(input_file=path)
        # Create document node from file source
        doc_text = [d.get_content() for d in documents]
        # Combine all nodes into one document
        source_doc = create_source_document(
            text="".join(doc_text),
            source_id=source_id,
            metadata=source_metadata,
        )
        # Set ignored metadata
        source_doc = set_ignored_metadata(
            source_document=source_doc,
            ignore_metadata=source_metadata,
        )
        document_results.append(source_doc)
    return document_results


# https://github.com/run-llama/llama_index/tree/40913847ba47d435b40b7fac3ae83eba89b56bb9/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/tabular
def csv_loader(
    sources: str,
    source_id: str,
    source_metadata: dict,
):
    document_results: List[Document] = []
    reader = CSVReader()
    for path in sources:
        documents = reader.load_data(file=Path(path))
        # Create document node from file source
        doc_text = [d.get_content() for d in documents]
        # Combine all nodes into one document
        source_doc = create_source_document(
            text="".join(doc_text),
            source_id=source_id,
            metadata=source_metadata,
        )
        # Set ignored metadata
        source_doc = set_ignored_metadata(
            source_document=source_doc,
            ignore_metadata=source_metadata,
        )
        document_results.append(source_doc)
    return document_results


# Can handle .txt, .docx, .pptx, .jpg, .png, .eml, .html, and .pdf
# https://github.com/run-llama/llama_index/tree/40913847ba47d435b40b7fac3ae83eba89b56bb9/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/unstructured
def unstructured_loader(
    sources: str,
    source_id: str,
    source_metadata: dict,
):
    document_results: List[Document] = []
    reader = UnstructuredReader()
    for path in sources:
        documents = reader.load_data(file=Path(path))
        # Create document node from file source
        doc_text = [d.get_content() for d in documents]
        # Combine all nodes into one document
        source_doc = create_source_document(
            text="".join(doc_text),
            source_id=source_id,
            metadata=source_metadata,
        )
        # Set ignored metadata
        source_doc = set_ignored_metadata(
            source_document=source_doc,
            ignore_metadata=source_metadata,
        )
        document_results.append(source_doc)
    return document_results


# https://github.com/run-llama/llama_index/tree/40913847ba47d435b40b7fac3ae83eba89b56bb9/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/xml
def xml_loader(
    sources: str,
    source_id: str,
    source_metadata: dict,
):
    document_results: List[Document] = []
    reader = XMLReader()
    for path in sources:
        documents = reader.load_data(file=Path(path))
        # Create document node from file source
        doc_text = [d.get_content() for d in documents]
        # Combine all nodes into one document
        source_doc = create_source_document(
            text="".join(doc_text),
            source_id=source_id,
            metadata=source_metadata,
        )
        # Set ignored metadata
        source_doc = set_ignored_metadata(
            source_document=source_doc,
            ignore_metadata=source_metadata,
        )
        document_results.append(source_doc)
    return document_results


# https://github.com/run-llama/llama_index/tree/40913847ba47d435b40b7fac3ae83eba89b56bb9/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/slides
def pptx_slides_loader(
    sources: str,
    source_id: str,
    source_metadata: dict,
):
    document_results: List[Document] = []
    reader = PptxReader()
    for path in sources:
        documents = reader.load_data(file=Path(path))
        # Create document node from file source
        doc_text = [d.get_content() for d in documents]
        # Combine all nodes into one document
        source_doc = create_source_document(
            text="".join(doc_text),
            source_id=source_id,
            metadata=source_metadata,
        )
        # Set ignored metadata
        source_doc = set_ignored_metadata(
            source_document=source_doc,
            ignore_metadata=source_metadata,
        )
        document_results.append(source_doc)
    return document_results


# https://github.com/run-llama/llama_index/tree/40913847ba47d435b40b7fac3ae83eba89b56bb9/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/image
def simple_image_loader(
    sources: str,
    source_id: str,
    source_metadata: dict,
):
    document_results: List[Document] = []
    reader = ImageReader()
    for path in sources:
        documents = reader.load_data(file=Path(path))
        # Create document node from file source
        doc_text = [d.get_content() for d in documents]
        # Combine all nodes into one document
        source_doc = create_source_document(
            text="".join(doc_text),
            source_id=source_id,
            metadata=source_metadata,
        )
        # Set ignored metadata
        source_doc = set_ignored_metadata(
            source_document=source_doc,
            ignore_metadata=source_metadata,
        )
        document_results.append(source_doc)
    return document_results


# https://github.com/run-llama/llama_index/blob/40913847ba47d435b40b7fac3ae83eba89b56bb9/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/image_vision_llm/base.py
def image_vision_loader():
    return []


# Requires OpenAI models
# https://github.com/run-llama/llama_index/tree/40913847ba47d435b40b7fac3ae83eba89b56bb9/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/video_audio
def simple_audio_video_loader(
    sources: str,
    source_id: str,
    source_metadata: dict,
):
    document_results: List[Document] = []
    reader = VideoAudioReader()
    for path in sources:
        documents = reader.load_data(file=Path(path))
        # Create document node from file source
        doc_text = [d.get_content() for d in documents]
        # Combine all nodes into one document
        source_doc = create_source_document(
            text="".join(doc_text),
            source_id=source_id,
            metadata=source_metadata,
        )
        # Set ignored metadata
        source_doc = set_ignored_metadata(
            source_document=source_doc,
            ignore_metadata=source_metadata,
        )
        document_results.append(source_doc)
    return document_results


async def llama_parse_loader(
    sources: List[str],
    source_id: str,
    source_metadata: dict,
) -> List[Document]:
    document_results: List[Document] = []

    # Get an api key at https://cloud.llamaindex.ai/login
    load_dotenv()
    llama_parse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

    # Parse .pdf files using LlamaParse service.
    # Any input file format is converted to text/markdown.
    # https://github.com/run-llama/llama_parse/tree/main
    parser = LlamaParse(
        api_key=llama_parse_api_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=8,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="en",  # Optionally you can define a language, default=en
    )

    # ----------------
    # Chunking methods
    # ----------------

    # sync
    # documents = parser.load_data(source_file_path)

    # sync batch
    # documents = parser.load_data(["./my_file1.pdf", "./my_file2.pdf"])

    # async
    # documents = await parser.aload_data(source_file_path)

    # async batch
    # documents = await parser.aload_data(["./my_file1.pdf", "./my_file2.pdf"])

    # FYI, results can take some time
    for path in sources:
        results = await parser.aload_data(path)
        document_results.extend(results)

    # Finalize documents
    for source_doc in document_results:
        # Set id
        source_doc.id_ = source_id
        # Set metadata
        source_doc.metadata.update(source_metadata)
        # Set ignored metadata
        source_doc = set_ignored_metadata(
            source_document=source_doc,
            ignore_metadata=source_metadata,
        )
    return document_results


# Free, no api key required
# https://jina.ai/reader/#apiform
def jina_reader_loader(
    app: classes.FastAPIApp,
    sources: List[str],
    source_id: str,
    source_metadata: dict,
) -> List[Document]:
    document_results: List[Document] = []
    # Create documents
    for path in sources:
        # Make an http request with the target address appended to `https://r.jina.ai/`
        req_url = f"https://r.jina.ai/{path}"
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/octet-stream",
        }
        client = app.state.requests_client
        text = ""
        with client.stream(method="GET", url=req_url, headers=headers) as res:
            res.raise_for_status()
            if res.status_code == 200:
                # Write data
                res.read()
                text = res.text
            else:
                raise Exception("Something went wrong reading data.")
        # Create document node
        source_doc = create_source_document(
            text=text,
            source_id=source_id,
            metadata=source_metadata,
        )
        document_results.append(source_doc)
    # Finalize documents
    for source_doc in document_results:
        # Set id
        source_doc.id_ = source_id
        # Set metadata
        source_doc.metadata.update(source_metadata)
        # Set ignored metadata
        source_doc = set_ignored_metadata(
            source_document=source_doc,
            ignore_metadata=source_metadata,
        )
    return document_results


# Read in source file(s) and build a document node with metadata.
# We will use this to base our chunks on.
async def documents_from_sources(
    app: dict,
    sources: List[str],
    source_id,
    source_metadata: dict,
    parsing_method: Optional[classes.FILE_LOADER_SOLUTIONS] = None,
) -> List[Document]:
    print(f"{common.PRNT_EMBED} Reading files...", flush=True)
    documents = []
    for source in sources:
        filename = os.path.basename(source)
        file_extension = common.get_file_extension_from_path(filename).lower()
        payload = dict(
            sources=[source],
            source_id=source_id,
            source_metadata=source_metadata,
        )
        # Use loader solution based on file type
        match file_extension:
            case "mdx" | "md" | "json" | "txt":
                # Regular text file
                documents = simple_file_loader(**payload)
            case "doc" | "docx":
                documents = ms_doc_loader(**payload)
            case "rtf":
                documents = rtf_loader(**payload)
            case "csv":
                documents = csv_loader(**payload)
            case "xml":
                documents = xml_loader(**payload)
            case "pptx":
                documents = pptx_slides_loader(**payload)
            case "png" | "jpg" | "jpeg" | "gif":
                documents = simple_image_loader(**payload)
            case "mp4" | "mp3":
                documents = simple_audio_video_loader(**payload)
            case "pdf":
                # PDF file
                match (parsing_method):
                    case classes.FILE_LOADER_SOLUTIONS.LLAMA_PARSE.value:
                        documents = await llama_parse_loader(**payload)
                    case _:
                        # default
                        documents = simple_pdf_loader(**payload)
            case _:
                is_url = source[:4] == "http"
                # Read from website using service
                if (
                    is_url
                    and parsing_method == classes.FILE_LOADER_SOLUTIONS.READER.value
                ):
                    documents = jina_reader_loader(
                        app=app,
                        **payload,
                    )
                # Unsupported
                else:
                    raise Exception(
                        f"The supplied file/url is not currently supported: {source}"
                    )
    # Return list of Documents
    return documents
