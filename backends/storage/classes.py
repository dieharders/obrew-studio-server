from typing import List, Optional
from pydantic import BaseModel


class GetChatThreadResponse(BaseModel):
    success: bool
    message: str
    data: List[dict]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Returned chat messages.",
                    "success": True,
                    "data": [
                        {
                            "id": "06ufWDF",
                            "created_at": "Monday January 21, 2012",
                            "title": "World's tallest buildings",
                            "summary": "A discussion about all the different buildings around the world that are the tallest.",
                            "numMessages": 1,
                            "messages": [
                                {
                                    "id": "shdaDoWj92D501jaufWDFiW",
                                    "content": "What is the tallest building in the world?",
                                    "role": "user",
                                    "created_at": "Monday, January 21, 2012",
                                    "order": 0,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    }


class GetChatThreadRequest(BaseModel):
    threadId: Optional[str | None] = None


class SaveChatThreadRequest(BaseModel):
    threadId: str
    thread: dict


class DeleteChatThreadRequest(BaseModel):
    threadId: Optional[str | None] = None
