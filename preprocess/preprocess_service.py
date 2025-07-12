from fastapi import FastAPI, HTTPException, Request
from preprocess import PreprocessService


app = FastAPI()
preprocess_service = PreprocessService()

@app.post("/preprocess_text/")
async def preprocess_text(request: Request):
    try:
        data = await request.json()
        text = data.get("text")
        tokens = preprocess_service.preprocess_text(text)
        return {"processed_tokens": tokens}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/preprocess_documents/")
async def preprocess_documents(request: Request):
    try:
        data = await request.json()
        documents = data.get("documents")
        input_file = data.get("input_file")
        output_file = data.get("output_file")
        processed_docs, all_tokens = preprocess_service.preprocess_documents(documents , input_file , output_file)
        
        return {"processed_docs": processed_docs, "all_tokens": all_tokens}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)