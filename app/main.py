from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import requests
import tempfile
from app.qa_chain import get_qa_chain
from app.utils import load_pdf  # Assume this turns PDF → LangChain docs

app = FastAPI()

TEAM_TOKEN = "8ad62148045cbf8137a66e1d8c0974e14f62a970b4fa91afb850f461abfbadb8"

# Input schema
class QueryRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

#“When a client sends a POST request to /api/v1/hackrx/run, run the function just below this line (the fastapi route decorator).”
@app.post("/api/v1/hackrx/run")
def run_query(request: QueryRequest, authorization: Optional[str] = Header(None)):
    # 🔐 Auth check
    if not authorization or authorization.split()[-1] != TEAM_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 📥 Download PDF from URL
    try:
        pdf_response = requests.get(request.documents)
        pdf_response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_response.content)
            pdf_path = temp_pdf.name # full absolute path to the temporary file
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF download failed: {e}")

    # 📄 Load the document
    docs = load_pdf(pdf_path)

    # 🧠 Build the QA chain
    qa = get_qa_chain(docs)

    # ❓ Answer all questions
    answers = []
    for q in request.questions:
        try:
            result = qa(q)
            answers.append(result["result"])
        except Exception as e:
            answers.append(f"Error answering question: {str(e)}")

    return {"answers": answers}
