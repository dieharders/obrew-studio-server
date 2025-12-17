import re
from typing import List
from core.document import Document, TextNode
import pysbd

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
    try:
        segmenter = pysbd.Segmenter(language=language, clean=False)
        return segmenter.segment(text)
    except:
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


class RecursiveCharacterTextSplitter:
    """
    Recursively splits text using a hierarchy of separators.

    Falls through separators when chunks are too large, starting with
    paragraph breaks and progressively using finer separators until
    chunks meet size requirements.

    Best for: General documents, PDFs, unstructured text.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None,
        language: str = "en",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.language = language

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using hierarchy of separators."""
        if not text:
            return []

        # Base case: no more separators, split by characters
        if not separators:
            # Character-level splitting as last resort
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i : i + self.chunk_size]
                if chunk:
                    chunks.append(chunk)
            return chunks

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        if separator:
            splits = text.split(separator)
        else:
            # Empty separator means character split
            return self._split_text_recursive(text, [])

        chunks = []
        current_chunk = ""

        for i, split in enumerate(splits):
            split = split.strip() if separator in ["\n\n", "\n"] else split

            # Calculate what adding this split would look like
            if current_chunk:
                potential_chunk = current_chunk + separator + split
            else:
                potential_chunk = split

            if len(potential_chunk) <= self.chunk_size:
                # Fits in current chunk
                current_chunk = potential_chunk
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)

                # Check if split itself is too large
                if len(split) > self.chunk_size:
                    # Recursively split with next separator
                    sub_chunks = self._split_text_recursive(split, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    # Start new chunk with overlap from previous
                    if chunks and self.chunk_overlap > 0:
                        overlap_text = chunks[-1][-self.chunk_overlap :]
                        current_chunk = overlap_text + separator + split
                        # Trim if overlap made it too long
                        if len(current_chunk) > self.chunk_size:
                            current_chunk = split
                    else:
                        current_chunk = split

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def get_nodes_from_documents(
        self, documents: List[Document], show_progress: bool = False
    ) -> List[TextNode]:
        """Split documents into text nodes using recursive character splitting."""
        nodes = []
        node_count = 0

        if not documents:
            return nodes

        for doc in documents:
            text = doc.text if doc.text else ""

            if not text.strip():
                continue

            # Recursively split text
            chunks = self._split_text_recursive(text, self.separators)

            for chunk in chunks:
                chunk = chunk.strip()
                if chunk:
                    nodes.append(
                        TextNode(
                            id_=str(node_count),
                            text=chunk,
                            metadata=doc.metadata.copy() if doc.metadata else {},
                        )
                    )
                    node_count += 1

        return nodes


class SentenceWindowSplitter:
    """
    Creates small embedding chunks with surrounding window context in metadata.

    Each chunk is small (1-3 sentences) for precise embedding, but stores
    surrounding sentences in metadata for context expansion at retrieval time.

    Best for: QA, detailed extraction, precise retrieval.

    Metadata stored:
    - window_before: preceding sentences
    - window_after: following sentences
    - original_text: full expanded text (window_before + chunk + window_after)
    """

    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 0,
        window_size: int = 3,
        language: str = "en",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.window_size = window_size
        self.language = language

    def get_nodes_from_documents(
        self, documents: List[Document], show_progress: bool = False
    ) -> List[TextNode]:
        """Split documents into sentence-based chunks with window context."""
        nodes = []
        node_count = 0

        if not documents:
            return nodes

        for doc in documents:
            text = doc.text if doc.text else ""

            if not text.strip():
                continue

            # Split into sentences
            sentences = _split_into_sentences(text, self.language)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                continue

            # Build chunks from sentences
            current_chunk_sentences = []
            current_chunk_start_idx = 0
            current_length = 0

            for i, sentence in enumerate(sentences):
                sentence_len = len(sentence)

                # Check if adding this sentence exceeds chunk size
                if (
                    current_length + sentence_len + 1 > self.chunk_size
                    and current_chunk_sentences
                ):
                    # Save current chunk with window context
                    chunk_text = " ".join(current_chunk_sentences)
                    chunk_end_idx = i - 1

                    # Build window context
                    window_before_start = max(
                        0, current_chunk_start_idx - self.window_size
                    )
                    window_after_end = min(len(sentences), i + self.window_size)

                    window_before = " ".join(
                        sentences[window_before_start:current_chunk_start_idx]
                    )
                    window_after = " ".join(sentences[i:window_after_end])

                    # Build original text with full context
                    original_parts = []
                    if window_before:
                        original_parts.append(window_before)
                    original_parts.append(chunk_text)
                    if window_after:
                        original_parts.append(window_after)
                    original_text = " ".join(original_parts)

                    # Create metadata with window info
                    chunk_metadata = doc.metadata.copy() if doc.metadata else {}
                    chunk_metadata.update(
                        {
                            "window_before": window_before,
                            "window_after": window_after,
                            "original_text": original_text,
                            "window_size": self.window_size,
                            "sentence_start_idx": current_chunk_start_idx,
                            "sentence_end_idx": chunk_end_idx,
                        }
                    )

                    nodes.append(
                        TextNode(
                            id_=str(node_count),
                            text=chunk_text,
                            metadata=chunk_metadata,
                        )
                    )
                    node_count += 1

                    # Start new chunk
                    current_chunk_sentences = [sentence]
                    current_chunk_start_idx = i
                    current_length = sentence_len
                else:
                    # Add sentence to current chunk
                    current_chunk_sentences.append(sentence)
                    current_length += sentence_len + 1  # +1 for space

            # Handle last chunk
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunk_end_idx = len(sentences) - 1

                window_before_start = max(0, current_chunk_start_idx - self.window_size)
                window_before = " ".join(
                    sentences[window_before_start:current_chunk_start_idx]
                )

                original_parts = []
                if window_before:
                    original_parts.append(window_before)
                original_parts.append(chunk_text)
                original_text = " ".join(original_parts)

                chunk_metadata = doc.metadata.copy() if doc.metadata else {}
                chunk_metadata.update(
                    {
                        "window_before": window_before,
                        "window_after": "",
                        "original_text": original_text,
                        "window_size": self.window_size,
                        "sentence_start_idx": current_chunk_start_idx,
                        "sentence_end_idx": chunk_end_idx,
                    }
                )

                nodes.append(
                    TextNode(
                        id_=str(node_count),
                        text=chunk_text,
                        metadata=chunk_metadata,
                    )
                )
                node_count += 1

        return nodes


class TokenTextSplitter:
    """
    Splits text based on estimated token count.

    Uses a configurable chars_per_token ratio for estimation.
    No external tokenizer dependencies required.

    Best for: Ensuring chunks fit model context windows.

    Default ratio of 4.0 chars/token works well for GPT-style models.
    For Llama-style models, try 3.5 chars/token.
    """

    DEFAULT_CHARS_PER_TOKEN = 4.0

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 20,
        chars_per_token: float = None,
        language: str = "en",
    ):
        self.chunk_size = chunk_size  # in tokens
        self.chunk_overlap = chunk_overlap  # in tokens
        self.chars_per_token = chars_per_token or self.DEFAULT_CHARS_PER_TOKEN
        self.language = language

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count."""
        return int(len(text) / self.chars_per_token)

    def _estimate_chars(self, tokens: int) -> int:
        """Estimate character count from token count."""
        return int(tokens * self.chars_per_token)

    def get_nodes_from_documents(
        self, documents: List[Document], show_progress: bool = False
    ) -> List[TextNode]:
        """Split documents into token-based chunks."""
        nodes = []
        node_count = 0

        if not documents:
            return nodes

        # Convert token limits to character estimates
        max_chunk_chars = self._estimate_chars(self.chunk_size)
        overlap_chars = self._estimate_chars(self.chunk_overlap)

        for doc in documents:
            text = doc.text if doc.text else ""

            if not text.strip():
                continue

            # Split into sentences first for cleaner boundaries
            sentences = _split_into_sentences(text, self.language)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                continue

            current_chunk = ""
            previous_chunk = ""

            for sentence in sentences:
                # If sentence alone exceeds max, split it by characters
                if len(sentence) > max_chunk_chars:
                    # Save current chunk first
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

                    # Split long sentence by character chunks
                    for i in range(0, len(sentence), max_chunk_chars - overlap_chars):
                        chunk = sentence[i : i + max_chunk_chars]
                        if chunk.strip():
                            nodes.append(
                                TextNode(
                                    id_=str(node_count),
                                    text=chunk.strip(),
                                    metadata=(
                                        doc.metadata.copy() if doc.metadata else {}
                                    ),
                                )
                            )
                            node_count += 1

                    current_chunk = ""
                    continue

                # Check if adding sentence exceeds limit
                test_chunk = (
                    (current_chunk + " " + sentence).strip()
                    if current_chunk
                    else sentence
                )

                if len(test_chunk) > max_chunk_chars:
                    # Save current chunk
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

                    # Start new chunk with overlap
                    if overlap_chars > 0 and previous_chunk:
                        overlap_text = previous_chunk[-overlap_chars:]
                        current_chunk = (overlap_text + " " + sentence).strip()
                        # If overlap made it too long, just use sentence
                        if len(current_chunk) > max_chunk_chars:
                            current_chunk = sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = test_chunk

            # Don't forget last chunk
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


def recursive_character_split(chunk_size: int = 500, chunk_overlap: int = 50):
    """Recommended for general documents (PDF, DOCX, unstructured text)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def sentence_window_split(
    chunk_size: int = 300, chunk_overlap: int = 0, window_size: int = 3
):
    """Precise retrieval with expanded context. Best for QA and detailed extraction."""
    return SentenceWindowSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        window_size=window_size,
    )


def token_split(
    chunk_size: int = 256, chunk_overlap: int = 20, chars_per_token: float = 4.0
):
    """Token-based splitting for precise context window management."""
    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chars_per_token=chars_per_token,
    )
