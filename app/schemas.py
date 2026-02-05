from pydantic import BaseModel, Field

class MedicalQuery(BaseModel):
    question: str = Field(..., example="what is diabetes?")

class MedicalResponse(BaseModel):
    request_id: str
    answer: str 
    disclaimer: str = "This information is for educational purposes only and not a substitute for professional medical advice."

