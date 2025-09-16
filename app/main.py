# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .embeddings import get_embedding_from_bytes, emb_to_bytes, bytes_to_emb
from .db import init_db, insert_face, find_best_match

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

@app.post('/register')
async def register(name: str = Form(...), file: UploadFile = File(...)):
    image_bytes = await file.read()
    emb = get_embedding_from_bytes(image_bytes)  # numpy array float32
    emb_blob = emb_to_bytes(emb)
    row_id = insert_face(name, emb_blob, image_bytes)
    return JSONResponse({'status': 'ok', 'id': row_id})

@app.post('/verify')
async def verify(file: UploadFile = File(...)):
    image_bytes = await file.read()
    emb = get_embedding_from_bytes(image_bytes)
    match = find_best_match(emb)
    if match:
        return JSONResponse({'match': match})
    return JSONResponse({'match': None})

if __name__ == '__main__':
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)