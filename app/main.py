from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    from embeddings import get_embedding_from_bytes, emb_to_bytes, bytes_to_emb
    from db import init_db, insert_face, find_best_match
except ImportError:
    from .embeddings import get_embedding_from_bytes, emb_to_bytes, bytes_to_emb
    from .db import init_db, insert_face, find_best_match

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add root endpoint to fix 404
@app.get("/")
async def root():
    return {"message": "Face Recognition API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

init_db()

@app.post('/register')
async def register(name: str = Form(...), file: UploadFile = File(...)):
    image_bytes = await file.read()
    emb = get_embedding_from_bytes(image_bytes)
    emb_blob = emb_to_bytes(emb)
    row_id = insert_face(name, emb_blob, image_bytes)
    return JSONResponse({'status': 'ok', 'id': row_id})

import traceback

@app.post('/verify')
async def verify(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        emb = get_embedding_from_bytes(image_bytes)
        match = find_best_match(emb)
        if match:
            return JSONResponse({'match': match})
        return JSONResponse({'match': None})
    except Exception as e:
        print(f"Error in verify: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse({'error': str(e)}, status_code=500)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=1000, reload=True)