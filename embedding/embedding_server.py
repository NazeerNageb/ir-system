from contextlib import asynccontextmanager
import sys
from fastapi import FastAPI, HTTPException, Request
from typing import List
import pathlib

import numpy as np
from embedding_service import EmbeddingService
from fastapi import FastAPI, HTTPException, Request
import scipy.sparse as sp

embedding_service = EmbeddingService()

@asynccontextmanager
async def lifespan(app: FastAPI):

    # تحديد مسار الجذر الحقيقي للمشروع بغض النظر عن مكان التشغيل
    project_root = pathlib.Path(__file__).parent.parent
    sys.path.append(str(project_root))
    # المسارات النسبية تصبح صحيحة بناءً على المشروع
    antique_folder = project_root / "files" / "antique"
    corpus_folder = project_root / "files" / "corpus"
    # antique_csv = project_root / "antique_clean_data.csv"
    # corpus_csv = project_root / "corpus_clean_data.csv"

    # تحميل الملفات
    embedding_service.preload(antique_folder, corpus_folder)
    # embedding_service.read_data(antique_csv, "antique")
    # embedding_service.read_data(corpus_csv, "corpus")

    embedding_service.load_chroma_embedding()

    yield
    

app = FastAPI(lifespan=lifespan)



# @app.post("/vectorize_query_embedding/")
# async def vectorize_query(request: Request):
#     try:
#         data = await request.json()
#         query_tokens = data.get("query_tokens")
#         dataset = data.get("dataset")
     
#         query_vector = embedding_service.queryEncode(query_tokens, dataset)
#         return {"query_vector": query_vector}
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/calculate_similarity_embedding/")
# async def calculate_similarity(request: Request):
#     try:
#         data = await request.json()
#         query_vector_data = data.get("query_vector")
#         dataset = data.get("dataset")

#         if not query_vector_data:
#             raise HTTPException(status_code=400, detail="Query vector data is missing")

#         # Reconstruct the sparse matrix
#         query_vector_sparse = sp.csr_matrix(
#             (query_vector_data["data"], query_vector_data["indices"], query_vector_data["indptr"]),
#             shape=query_vector_data["shape"]
#         )
        
#         similarities = embedding_service.consine_similarity(query_vector_sparse, dataset)
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
#     return {"similarities": similarities}


@app.post("/vectorize_query_embedding/")
async def vectorize_query(request: Request):
    print("\n📩 [API] /vectorize_query_embedding/ endpoint called")
    try:
        data = await request.json()
        query_tokens = data.get("query_tokens")
        dataset = data.get("dataset")

        print(f"📥 Received query_tokens: {query_tokens}")
        print(f"📥 Received dataset: {dataset}")

        if not query_tokens or not dataset:
            print("❌ Missing query_tokens or dataset")
            raise HTTPException(status_code=400, detail="query_tokens and dataset are required")

        query_vector = embedding_service.queryEncode(query_tokens)
        print(f"✅ Query vector generated, shape: {len(query_vector)}")

        return {"query_vector": query_vector.tolist()}
    except ValueError as e:
        print(f"❌ ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Exception occurred in vectorize_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate_similarity_embedding/")
async def calculate_similarity(request: Request):
    print("\n📩 [API] /calculate_similarity_embedding/ endpoint called")
    try:
        data = await request.json()
        query_vector_data = data.get("query_vector")
        dataset = data.get("dataset")

        print(f"📥 Received dataset: {dataset}")
        print(f"📥 Received query_vector_data: {type(query_vector_data)}, len: {len(query_vector_data)}")

        if not query_vector_data:
            print("❌ Query vector data is missing")
            raise HTTPException(status_code=400, detail="Query vector data is missing")

        query_vector_array = np.array(query_vector_data)
        print(f"✅ Query vector shape: {query_vector_array.shape}")

        similarities = embedding_service.consine_similarity(query_vector_array, dataset)

        if similarities is None:
            print("❌ Error: Similarity calculation returned None")
            raise HTTPException(status_code=500, detail="Similarity calculation failed")

        print(f"✅ Similarities calculated, total: {len(similarities)}")
        return {"similarities": similarities.tolist()}

    except ValueError as e:
        print(f"❌ ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print(f"❌ Exception occurred in calculate_similarity:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

   
@app.post("/query_chroma/")
async def query_chroma(request: Request):
    print("\n📩 [API] /query_chroma/ endpoint called")
    try:
        data = await request.json()
        query_embedding = data.get("query_vector")
        dataset = data.get("dataset")       
        
        query_vector_array = np.array(query_embedding)      
    
        results = embedding_service.query_chroma(query_vector_array, dataset, 10)

        print(f"✅ Query results: {results}")
        return {"results": results}

    except ValueError as e:
        print(f"❌ ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print(f"❌ Exception occurred in query_chroma:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)


