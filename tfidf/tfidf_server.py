from contextlib import asynccontextmanager
import sys
from fastapi import FastAPI, HTTPException, Request
from typing import List
import pathlib
from tf_idf_service import TfidfService
from fastapi import FastAPI, HTTPException, Request
import scipy.sparse as sp

tfidf_service = TfidfService()

@asynccontextmanager
async def lifespan(app: FastAPI):

 
    project_root = pathlib.Path(__file__).parent.parent
    sys.path.append(str(project_root))

    antique_folder = project_root / "files" / "antique"
    corpus_folder = project_root / "files" / "corpus"
    antique_csv = project_root / "antique_clean_data.csv"
    corpus_csv = project_root / "corpus_clean_data.csv"

 
    tfidf_service.preload(antique_folder, corpus_folder)
    # tfidf_service.read_data(antique_csv, "antique")
    # tfidf_service.read_data(corpus_csv, "corpus")

    yield
    

app = FastAPI(lifespan=lifespan)


@app.post("/vectorize_query/")
async def vectorize_query(request: Request):
    try:
        data = await request.json()
        query_tokens = data.get("query_tokens")
        dataset = data.get("dataset")
     
        query_vector = tfidf_service.vectorize_query(query_tokens, dataset)
        return {"query_vector": query_vector}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate_similarity/")
async def calculate_similarity(request: Request):
    try:
        data = await request.json()
        query_vector_data = data.get("query_vector")
        dataset = data.get("dataset")

        if not query_vector_data:
            raise HTTPException(status_code=400, detail="Query vector data is missing")

        query_vector_sparse = sp.csr_matrix(
            (query_vector_data["data"], query_vector_data["indices"], query_vector_data["indptr"]),
            shape=query_vector_data["shape"]
        )
        
        similarities = tfidf_service.calculate_similarity_api(query_vector_sparse, dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"similarities": similarities}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


