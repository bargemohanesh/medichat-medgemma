from fastapi import FastAPI
from app.schemas import MedicalQuery, MedicalResponse
from app.medgemma import MedGemmaService
import uuid
import re
import logging


app = FastAPI()

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)

app = FastAPI(
    title="MediChat – MedGemma Powered Medical Assistant", 
    description = "A privacy-preserving medical Q&A system using Googles MedGemma",
    version="1.0.0"
    )

medgemma = MedGemmaService()

@app.get("/")
def health_check():
    return {"status": "ok", "model": "medgemma-1.5-4b-it"}

@app.get("/ready")
def readiness_check():
    return {
        "status": "ready",
        "model": "medgemma-1.5-4b-it",
        "device": "cuda"
    }

def safety_block(question:str) -> str | None:
    q = question.lower()

    # Emergency indicators
    if re.search(r"\b(chest pain|difficulty breathing|unconscious|seizure|strokes)\b", q):
        return (
            "This is medical emergency. Please seek immediate medical attention" 
            "or contact local emergency services."
        )
    # Prescription / dosage requests
    if re.search(r"\b(dosage|dose|how many mg|prescribe|prescription)\b", q):
        return (
            "I can't provide prescription or exact medication dosage."
            "please consult a licensed heathcare professional."
        )
    return None

@app.post("/ask", response_model=MedicalResponse)
def ask_medical_question(payload: MedicalQuery):
    request_id = str(uuid.uuid4())

    blocked = safety_block(payload.question)
    if blocked:
        return {
            "request_id": request_id,
            "answer": blocked,
            "disclaimer": "This is for edicational purpose only and not a substitute for professional medical advice."
        }
    
    logging.info(f"REQUEST {request_id} | question = {payload.question}")
    answer = medgemma.generate(payload.question)
    logging.info(f"Response {request_id} | answer_length = {len(answer)}")

    #return MedicalResponse(answer=answer)
    return {
        "request_id": request_id,
        "answer": answer,
        "disclaimer": "This information is for educational purpose only and not a substitute for professional medical advice."
    }