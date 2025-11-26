import re
from typing import List
from core.document import Document, TextNode

# Optional pysbd for better sentence boundary detection
try:
    import pysbd

    PYSBD_AVAILABLE = True
except ImportError:
    PYSBD_AVAILABLE = False

# Alternative:
# wtpsplit (State-of-the-art, heavier)
# Install: pip install wtpsplit
# Uses ML models but they're bundled with the package
# 85 languages, best accuracy
# Heavier dependency (requires torch)

OUTPUT_PDF_IMAGES_PATH = "memories/parsed/pdfImages/"

# Helpers


def combine_sentences(sentences, buffer_size=1):
    for i in range(len(sentences)):
        # Create string to hold joined sentences
        combined_sentence = ""
        # Add sentences before current one, based on buffer
        for j in range(i - buffer_size, i):
            # Check if index j is not negative
            if j >= 0:
                # Add sentence to combined str
                combined_sentence += sentences[j]["sentence"] + " "

        # Add the current sentence
        combined_sentence += sentences[i]["sentence"]

        # Add sentences after current one, based on buffer
        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                # Add sentence to combined str
                combined_sentence += " " + sentences[j]["sentence"]

        # Then add everything to dict
        sentences[i]["combined_sentence"] = combined_sentence
    return sentences


# Uncomment when needed
# def image_to_base64(image_path: str):
#     with Image.open(image_path) as image:
#         buffered = io.BytesIO()
#         image.save(buffered, format=image.format)
#         img_str = base64.b64encode(buffered.getvalue())
#         return img_str.decode("utf-8")


# Document Splitters


def _split_into_sentences(text: str, language: str = "en") -> List[str]:
    """
    Split text into sentences using pysbd if available, otherwise fall back to regex.

    pysbd handles edge cases like:
    - Abbreviations: "Dr. Smith went home."
    - Decimal numbers: "The price is $3.50."
    - URLs: "Visit https://example.com. It's great."
    - Ellipsis: "Wait... what happened?"
    """
    if PYSBD_AVAILABLE:
        segmenter = pysbd.Segmenter(language=language, clean=False)
        return segmenter.segment(text)
    else:
        # Fallback regex - handles basic sentence boundaries
        # but may incorrectly split on abbreviations like "Dr." or "Mr."
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in sentences if s.strip()]


class SimpleSentenceSplitter:
    """
    Text splitter that respects sentence boundaries and chunk size.

    Uses pysbd for robust sentence boundary detection if available,
    otherwise falls back to regex-based splitting.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 0,
        paragraph_separator: str = "\n\n",
        language: str = "en",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.paragraph_separator = paragraph_separator
        self.language = language

    def get_nodes_from_documents(
        self, documents: List[Document], show_progress: bool = False
    ) -> List[TextNode]:
        """Split documents into text nodes."""
        nodes = []
        node_count = 0

        if not documents:
            return nodes

        for doc in documents:
            text = doc.text if doc.text else ""

            # Handle empty documents
            if not text.strip():
                continue

            # Split by paragraph separator first, then by sentences within paragraphs
            paragraphs = text.split(self.paragraph_separator)

            current_chunk = ""
            previous_chunk = ""  # For overlap handling

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # If paragraph fits in current chunk, add it
                if (
                    len(current_chunk) + len(para) + len(self.paragraph_separator)
                    <= self.chunk_size
                ):
                    if current_chunk:
                        current_chunk += self.paragraph_separator + para
                    else:
                        current_chunk = para
                else:
                    # Save current chunk if not empty
                    if current_chunk.strip():
                        nodes.append(
                            TextNode(
                                id_=str(node_count),
                                text=current_chunk.strip(),
                                metadata=doc.metadata.copy() if doc.metadata else {},
                            )
                        )
                        node_count += 1
                        previous_chunk = current_chunk

                    # If paragraph itself is too long, split by sentences
                    if len(para) > self.chunk_size:
                        sentences = _split_into_sentences(para, self.language)

                        # Handle overlap from previous chunk
                        if self.chunk_overlap > 0 and previous_chunk:
                            current_chunk = previous_chunk[-self.chunk_overlap :]
                        else:
                            current_chunk = ""

                        for sentence in sentences:
                            sentence = sentence.strip()
                            if not sentence:
                                continue

                            # If single sentence exceeds chunk_size, add it as its own node
                            if len(sentence) > self.chunk_size:
                                # Save current chunk first
                                if current_chunk.strip():
                                    nodes.append(
                                        TextNode(
                                            id_=str(node_count),
                                            text=current_chunk.strip(),
                                            metadata=(
                                                doc.metadata.copy()
                                                if doc.metadata
                                                else {}
                                            ),
                                        )
                                    )
                                    node_count += 1
                                    previous_chunk = current_chunk
                                # Add long sentence as its own node
                                nodes.append(
                                    TextNode(
                                        id_=str(node_count),
                                        text=sentence,
                                        metadata=(
                                            doc.metadata.copy() if doc.metadata else {}
                                        ),
                                    )
                                )
                                node_count += 1
                                previous_chunk = sentence
                                # Start fresh with overlap
                                if self.chunk_overlap > 0:
                                    current_chunk = sentence[-self.chunk_overlap :]
                                else:
                                    current_chunk = ""
                            elif (
                                len(current_chunk) + len(sentence) + 1 > self.chunk_size
                            ):
                                # Save current chunk and start new one
                                if current_chunk.strip():
                                    nodes.append(
                                        TextNode(
                                            id_=str(node_count),
                                            text=current_chunk.strip(),
                                            metadata=(
                                                doc.metadata.copy()
                                                if doc.metadata
                                                else {}
                                            ),
                                        )
                                    )
                                    node_count += 1
                                    previous_chunk = current_chunk
                                # Handle overlap
                                if self.chunk_overlap > 0 and previous_chunk:
                                    current_chunk = (
                                        previous_chunk[-self.chunk_overlap :]
                                        + " "
                                        + sentence
                                    )
                                else:
                                    current_chunk = sentence
                            else:
                                # Add sentence to current chunk
                                if current_chunk:
                                    current_chunk += " " + sentence
                                else:
                                    current_chunk = sentence
                    else:
                        # Paragraph fits, handle overlap and start new chunk
                        if self.chunk_overlap > 0 and previous_chunk:
                            current_chunk = (
                                previous_chunk[-self.chunk_overlap :]
                                + self.paragraph_separator
                                + para
                            )
                        else:
                            current_chunk = para

            # Don't forget the last chunk
            if current_chunk.strip():
                nodes.append(
                    TextNode(
                        id_=str(node_count),
                        text=current_chunk.strip(),
                        metadata=doc.metadata.copy() if doc.metadata else {},
                    )
                )
                node_count += 1

        return nodes


class SimpleMarkdownSplitter:
    """Simple markdown splitter that splits by headings."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 0,
        language: str = "en",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language

    def get_nodes_from_documents(
        self, documents: List[Document], show_progress: bool = False
    ) -> List[TextNode]:
        """Split markdown documents by headings."""
        nodes = []
        node_count = 0

        if not documents:
            return nodes

        for doc in documents:
            text = doc.text if doc.text else ""

            # Handle empty documents
            if not text.strip():
                continue

            # Split by markdown headings (## for h2)
            sections = re.split(r"\n(?=## )", text)

            for section in sections:
                section = section.strip()
                if not section:
                    continue

                # If section is too long, split further by sentences
                if len(section) > self.chunk_size:
                    sentences = _split_into_sentences(section, self.language)
                    current_chunk = ""

                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue

                        # Handle sentences longer than chunk_size
                        if len(sentence) > self.chunk_size:
                            if current_chunk.strip():
                                nodes.append(
                                    TextNode(
                                        id_=str(node_count),
                                        text=current_chunk.strip(),
                                        metadata=(
                                            doc.metadata.copy() if doc.metadata else {}
                                        ),
                                    )
                                )
                                node_count += 1
                            # Add long sentence as its own node
                            nodes.append(
                                TextNode(
                                    id_=str(node_count),
                                    text=sentence,
                                    metadata=(
                                        doc.metadata.copy() if doc.metadata else {}
                                    ),
                                )
                            )
                            node_count += 1
                            current_chunk = ""
                        elif len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                            if current_chunk.strip():
                                nodes.append(
                                    TextNode(
                                        id_=str(node_count),
                                        text=current_chunk.strip(),
                                        metadata=(
                                            doc.metadata.copy() if doc.metadata else {}
                                        ),
                                    )
                                )
                                node_count += 1
                            current_chunk = sentence
                        else:
                            current_chunk += (
                                " " + sentence if current_chunk else sentence
                            )

                    if current_chunk.strip():
                        nodes.append(
                            TextNode(
                                id_=str(node_count),
                                text=current_chunk.strip(),
                                metadata=doc.metadata.copy() if doc.metadata else {},
                            )
                        )
                        node_count += 1
                else:
                    nodes.append(
                        TextNode(
                            id_=str(node_count),
                            text=section,
                            metadata=doc.metadata.copy() if doc.metadata else {},
                        )
                    )
                    node_count += 1

        return nodes


def split_sentence(file_path: str):
    """Split a file into sentence-based chunks."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text:
        raise Exception("No documents found.")

    doc = Document(id_="0", text=text)
    text_splitter = SimpleSentenceSplitter(chunk_size=550, chunk_overlap=15)
    nodes = text_splitter.get_nodes_from_documents([doc])
    return nodes


# Uncomment when needed
# def pdf_split(folder_path: str, filename: str):
#     image_output_path = os.path.join(folder_path, OUTPUT_PDF_IMAGES_PATH)
#     file_path = os.path.join(folder_path, filename)

#     # @TODO pip install unstructured and import partition_pdf
#     def partition_pdf():
#         return [{}]

#     # Extracts elements from pdf (tables, text, images, etc)
#     elements = partition_pdf(
#         filename=file_path,
#         # Find embedded image blocks
#         extract_images_in_pdf=True,
#         # helpers
#         strategy="hi_res",
#         infer_table_structure=True,
#         model_name="yolox",
#         chunking_strategy="by_little",
#         max_characters=4000,
#         new_after_n_chars=3800,
#         combine_text_under_n_chars=2000,
#         image_output_dir_path=image_output_path,
#     )
#     # Extract table data as html since LLM's understand them better
#     table = elements[0].metadata.text_as_html
#     # When encountering images, 2 strategies:
#     # 1. Generate a text summary and embed that text
#     # 2. Generate embeddings for image using pre-trained vision model
#     image_paths = [
#         f
#         for f in os.listdir(folder_path)
#         if os.path.isfile(os.path.join(folder_path, f))
#     ]
#     image_summaries = []
#     for img_path in image_paths:
#         image_str = image_to_base64(img_path)
#         # @TODO Feed this image url to Vision LLM to summarize
#         summary = "This is a summary"  # llm.vision({"url": f"data:image/jpeg;base64,{image_str}"})
#         image_summaries.append(summary)

#     # @TODO Return all results combined
#     return [table, image_summaries]


# Text Splitters

# Uncomment when needed
# def split_chars(text: str):
#     text_splitter = CharacterTextSplitter(
#         chunk_size=35, chunk_overlap=4, seperator="", strip_whitespace=False
#     )
#     docs = text_splitter.create_documents([text])
#     result = []
#     for document in docs:
#         result.append(document.page_content)
#     return result


# Split by similarity (advanced)
# This is handled in llama-index -> SemanticSplitterNodeParser: https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules.html#semanticsplitternodeparser
# Output a pairwise sequence of chunks and embeddings List[{chunk: string, embedding: List[int]}]
# Use this output to compare sentences together and find the largest discrepency (distance) in relationship
# for determining when to break text off for chunks.
def semantic_split(text: str):
    # Split out all sentences
    single_sentences_list = re.split(r"(?<=[.?!])\s+", text)
    # Create a document data type with helpful metadata
    sentences = [
        {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
    ]
    combined_sentences = combine_sentences(sentences)
    return combined_sentences


# Instruct an LLM to chunk the text (advanced)
def agentic_split(text: str):
    return text


# Uncomment when needed
# Recommended for general use
# def recursive_char_split(text: str) -> List[str]:
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=0)
#     docs = text_splitter.create_documents([text])
#     result = []
#     for document in docs:
#         result.append(document.page_content)
#     return docs


# Uncomment when needed
# def code_split(text: str):
#     text_splitter = RecursiveCharacterTextSplitter.from_language(
#         language=Language.JS, chunk_size=65, chunk_overlap=0
#     )
#     docs = text_splitter.create_documents([text])
#     result = []
#     for document in docs:
#         result.append(document.page_content)
#     return


# Parsers / Factory Functions


def markdown_document_split(chunk_size: int = 500, chunk_overlap: int = 0):
    """Recommended for markdown or code documents."""
    return SimpleMarkdownSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def markdown_heading_split(chunk_size: int = 500, chunk_overlap: int = 0):
    """Split along major headings (h2) then by whole sentences."""
    return SimpleSentenceSplitter(
        paragraph_separator="\n## ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
