import os
import json
from datetime import datetime, timezone
from core import classes, common
from fastapi import APIRouter, Request, Depends, File, BackgroundTasks, UploadFile
from embeddings.vector_storage import VECTOR_STORAGE_PATH, Vector_Storage
from embeddings.embedder import Embedder
from . import file_parsers

router = APIRouter()


@router.get("/addCollection")
def create_memory_collection(
    request: Request,
    form: classes.AddCollectionRequest = Depends(),
) -> classes.AddCollectionResponse:
    app = request.app

    try:
        parsed_tags = common.parse_valid_tags(form.tags)
        collection_name = form.collectionName
        if not collection_name:
            raise Exception("You must supply a collection name.")
        if parsed_tags == None:
            raise Exception("Invalid value for 'tags' input.")
        if not common.check_valid_id(collection_name):
            raise Exception(
                "Invalid collection name. No '--', uppercase, spaces or special chars allowed."
            )
        # Create payload. ChromaDB only accepts strings, numbers, bools.
        metadata = {
            "icon": form.icon or "",
            "createdAt": datetime.now(timezone.utc).strftime("%B %d %Y - %H:%M:%S"),
            "tags": parsed_tags,
            "description": form.description,
            "sources": json.dumps([]),
        }
        vector_storage = Vector_Storage(app=app)
        vector_storage.db_client.create_collection(
            name=collection_name,
            metadata=metadata,
        )
        msg = f'Successfully created new collection "{collection_name}"'
        print(f"{common.PRNT_API} {msg}")
        return {
            "success": True,
            "message": msg,
        }
    except Exception as e:
        msg = f'Failed to create new collection "{collection_name}": {e}'
        print(f"{common.PRNT_API} {msg}")
        return {
            "success": False,
            "message": msg,
        }


# Create a memory for Ai.
# This is a multi-step process involving several endpoints.
# It will first process the file, then embed its data into vector space and
# finally add it as a document to specified collection.
@router.post("/addDocument")
async def create_memory(
    request: Request,
    form: classes.EmbedDocumentRequest = Depends(),
    file: UploadFile = File(None),  # File(...) means required
    background_tasks: BackgroundTasks = None,  # This prop is auto populated by FastAPI
) -> classes.AddDocumentResponse:
    tmp_input_file_path = ""
    app = request.app

    try:
        embedder = Embedder(app=app)
        tmp_input_file_path = await embedder.modify_document(
            form=form,
            file=file,
            background_tasks=background_tasks,
            is_update=False,
        )
    except (Exception, KeyError) as e:
        # Error
        msg = f"Failed to create a new memory: {e}"
        print(f"{common.PRNT_API} {msg}")
        return {
            "success": False,
            "message": msg,
        }
    else:
        msg = "A new memory has been added to the queue. It will be available for use shortly."
        print(f"{common.PRNT_API} {msg}", flush=True)
        return {
            "success": True,
            "message": msg,
        }
    finally:
        # Delete uploaded tmp file
        if os.path.exists(tmp_input_file_path):
            os.remove(tmp_input_file_path)
            print(f"{common.PRNT_API} Removed temp file.")


# Re-process and re-embed document
@router.post("/updateDocument")
async def update_memory(
    request: Request,
    form: classes.EmbedDocumentRequest = Depends(),
    file: UploadFile = File(None),  # File(...) means required
    background_tasks: BackgroundTasks = None,  # This prop is auto populated by FastAPI
) -> classes.AddDocumentResponse:
    tmp_input_file_path = ""
    app = request.app

    try:
        embedder = Embedder(app=app)
        tmp_input_file_path = await embedder.modify_document(
            form=form,
            file=file,
            background_tasks=background_tasks,
            is_update=True,
        )
    except (Exception, KeyError) as e:
        # Error
        msg = f"Failed to update memory: {e}"
        print(f"{common.PRNT_API} {msg}", flush=True)
        return {
            "success": False,
            "message": msg,
        }
    else:
        msg = "A memory has been added to the update queue. It will be available for use shortly."
        print(f"{common.PRNT_API} {msg}", flush=True)
        return {
            "success": True,
            "message": msg,
        }
    finally:
        # Delete uploaded tmp file
        if os.path.exists(tmp_input_file_path):
            os.remove(tmp_input_file_path)
            print(f"{common.PRNT_API} Removed temp file.")


@router.get("/getAllCollections")
def get_all_collections(
    request: Request,
) -> classes.GetAllCollectionsResponse:
    app = request.app

    try:
        vector_storage = Vector_Storage(app=app)
        collections = vector_storage.get_all_collections()

        return {
            "success": True,
            "message": f"Returned {len(collections)} collection(s)",
            "data": collections,
        }
    except Exception as e:
        print(f"{common.PRNT_API} Error: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": [],
        }


# Return a collection by id and all its documents
@router.post("/getCollection")
def get_collection(
    request: Request,
    props: classes.GetCollectionRequest,
) -> classes.GetCollectionResponse:
    app = request.app

    try:
        name = props.id
        vector_storage = Vector_Storage(app=app)
        collection = vector_storage.get_collection(name=name)

        return {
            "success": True,
            "message": f"Returned collection(s) {name}",
            "data": collection,
        }
    except Exception as e:
        print(f"{common.PRNT_API} Error: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": {},
        }


# Get all chunks for a source document
@router.post("/getChunks")
def get_chunks(request: Request, params: classes.GetDocumentChunksRequest):
    collection_id = params.collectionId
    document_id = params.documentId
    num_chunks = 0
    app = request.app

    try:
        vector_storage = Vector_Storage(app=app)
        chunks = vector_storage.get_source_chunks(
            collection_name=collection_id, source_id=document_id
        )
        if chunks != None:
            num_chunks = len(chunks)

        return {
            "success": True,
            "message": f"Returned {num_chunks} chunks for document.",
            "data": chunks,
        }
    except Exception as e:
        print(f"{common.PRNT_API} Error: {e}")
        return {
            "success": False,
            "message": f"{e}",
            "data": None,
        }


# Open an OS file exporer on host machine
@router.get("/fileExplore")
def explore_source_file(
    params: classes.FileExploreRequest = Depends(),
) -> classes.FileExploreResponse:
    filePath = params.filePath

    if not os.path.exists(filePath):
        return {
            "success": False,
            "message": "No file path exists",
        }

    # Open a new os window
    common.file_explore(filePath)

    return {
        "success": True,
        "message": "Opened file explorer",
    }


# Delete one or more source documents by id
@router.post("/deleteDocuments")
def delete_document_sources(
    request: Request,
    params: classes.DeleteDocumentsRequest,
) -> classes.DeleteDocumentsResponse:
    app = request.app

    try:
        collection_name = params.collection_id
        source_ids = params.document_ids
        num_documents = len(source_ids)
        # Find source data
        vector_storage = Vector_Storage(app=app)
        collection = vector_storage.get_collection(name=collection_name)
        sources_to_delete = vector_storage.get_sources_from_ids(
            collection=collection, source_ids=source_ids
        )
        # Remove specified source(s)
        embedder = Embedder(app=app)
        embedder.delete_sources(
            collection_name=collection_name, sources=sources_to_delete
        )

        return {
            "success": True,
            "message": f"Removed {num_documents} source(s): {source_ids}",
        }
    except Exception as e:
        print(f"{common.PRNT_API} Error: {e}")
        return {
            "success": False,
            "message": str(e),
        }


# Delete a collection by id
@router.get("/deleteCollection")
def delete_collection(
    request: Request,
    params: classes.DeleteCollectionRequest = Depends(),
) -> classes.DeleteCollectionResponse:
    app = request.app

    try:
        collection_id = params.collection_id
        vector_storage = Vector_Storage(app=app)
        db = vector_storage.db_client
        collection = db.get_collection(collection_id)
        # Remove all the sources in this collection
        sources = vector_storage.get_collection_sources(collection)
        # Remove all associated source files
        embedder = Embedder(app=app)
        embedder.delete_sources(collection_name=collection_id, sources=sources)
        # Remove the collection
        db.delete_collection(name=collection_id)
        # Remove persisted vector index from disk
        common.delete_vector_store(
            target_file_path=collection_id, folder_path=VECTOR_STORAGE_PATH
        )
        return {
            "success": True,
            "message": f"Removed collection [{collection_id}]",
        }
    except Exception as e:
        print(f"{common.PRNT_API} Error: {e}")
        return {
            "success": False,
            "message": str(e),
        }


# Completely wipe database
@router.get("/wipe")
def wipe_all_memories(
    request: Request,
) -> classes.WipeMemoriesResponse:
    app = request.app

    try:
        # Delete all db values
        vector_storage = Vector_Storage(app=app)
        db = vector_storage.db_client
        db.reset()
        # Delete all parsed files in /memories
        file_parsers.delete_all_files()
        # Remove all vector storage collections and folders
        vector_storage.delete_all_vector_storage()
        # Acknowledge success
        return {
            "success": True,
            "message": "Successfully wiped all memories from Ai",
        }
    except Exception as e:
        print(f"{common.PRNT_API} Error: {e}")
        return {
            "success": False,
            "message": str(e),
        }
