from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from ingest import ingest_document
from retrieval import retrieve_with_rerank
from generator import generate_answer

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.post("/upload_document")
def upload_document(file: UploadFile):
    path = f"data/{file.filename}"

    with open(path, "wb") as f:
        f.write(file.file.read())

    ingest_document(path)

    return {"status": "ingested"}


@app.post("/query")
@app.post("/query")
def query(req: QueryRequest):
    query_text = req.query

    context_chunks = retrieve_with_rerank(query_text)

    if len(context_chunks) == 0:
        return {
            "answer": "The provided context does not contain information to answer this question."
        }

    answer = generate_answer(query_text, context_chunks)

    return {"answer": answer}
