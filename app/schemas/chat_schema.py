"""
WHAT: Pydantic models defining the shape of /chat request and response bodies.
WHY: FastAPI uses these to validate incoming JSON and auto-generate API
     docs (Swagger UI), so we don't have to manually check field types.
OUTPUT: `ChatRequest` (input) and `ChatResponse` (output) classes.
CALLED FROM: app/routers/chat.py
WHY CALLED: Type-safe request/response handling for the /chat endpoint.

সহজ ভাষায় (Bengali):
এটা শুধু "ফর্মের নকশা" — /chat এন্ডপয়েন্টে কেউ request পাঠালে ঠিক কী কী
field থাকতে হবে (ChatRequest), আর আমরা response এ কী কী ফেরত দেব
(ChatResponse) সেটা এখানে ঠিক করা আছে।
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="যে user প্রশ্ন করছে, তার unique id")
    session_id: str = Field(..., description="বর্তমান চ্যাট সেশনের id")
    question: str = Field(..., description="ইউজারের প্রশ্ন")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "masum",
                "session_id": "session_abc123",
                "question": "who has won the most world cups?",
            }
        }


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Self-RAG graph এর চূড়ান্ত উত্তর")
    documents_relevant: bool
    generation_grounded: bool
    generation_useful: bool