from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import requests
from fastapi.middleware.cors import CORSMiddleware
import pathlib
import json
from sklearn.neighbors import NearestNeighbors
import pandas as pd


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

QUERY_PROCESSING_URL = "http://127.0.0.1:5000/preprocess_text/"
VECTORIZE_QUERY_URL = "http://localhost:8001/vectorize_query"
VECTORIZE_QUERY_EMBEDDING_URL = "http://localhost:8006/vectorize_query_embedding"
CALCULATE_SIMILARITY_URL = "http://localhost:8001/calculate_similarity"
CALCULATE_SIMILARITY_EMBEDDING_URL = "http://localhost:8006/calculate_similarity_embedding"
CALCULATE_SIMILARITY_CHROMA_URL = "http://localhost:8006/query_chroma"
RANK_DOCUMENTS_URL = "http://localhost:8002/rank_documents/"

# Path to save user queries log
log_dir_path = pathlib.Path.cwd() / "files"

def get_log_path(dataset: str) -> pathlib.Path:
    dataset_log_path = log_dir_path / dataset / "user_queries_log.json"
    return dataset_log_path
        


@app.post("/search_query_hybrid")
async def search_query_hybrid(request: Request):
    try:
        request_body = await request.json()
        query = request_body.get("query")
        dataset = request_body.get("dataset")
        top_k = request_body.get("top_k") or 10

        # التحقق من المدخلات
        if not query or not dataset:
            raise HTTPException(status_code=400, detail="Query and dataset must be provided.")

        # حفظ سجل الاستعلامات
        user_queries_log_path = get_log_path(dataset)
        if user_queries_log_path.exists():
            with open(user_queries_log_path, 'r') as f:
                user_queries_log = json.load(f)
        else:
            user_queries_log = []

        user_queries_log.append(query)
        user_queries_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(user_queries_log_path, 'w') as f:
            json.dump(user_queries_log, f)

        # 1. معالجة الاستعلام
        response = requests.post(QUERY_PROCESSING_URL, json={"text": query})
        response.raise_for_status()
        processed_tokens = response.json().get("processed_tokens")
        if not processed_tokens:
            raise HTTPException(status_code=500, detail="Processed tokens not returned.")

        # 2. توليد المتجهات
        tfidf_vector_resp = requests.post(VECTORIZE_QUERY_URL, json={"query_tokens": processed_tokens, "dataset": dataset})
        embedding_vector_resp = requests.post(VECTORIZE_QUERY_EMBEDDING_URL, json={"query_tokens": processed_tokens, "dataset": dataset})
        tfidf_vector_resp.raise_for_status()
        embedding_vector_resp.raise_for_status()
        tfidf_vector = tfidf_vector_resp.json().get("query_vector")
        embedding_vector = embedding_vector_resp.json().get("query_vector")

        if not tfidf_vector or not embedding_vector:
            raise HTTPException(status_code=500, detail="One or both query vectors are missing.")

        # 3. حساب التشابه
        tfidf_sim_resp = requests.post(CALCULATE_SIMILARITY_URL, json={"query_vector": tfidf_vector, "dataset": dataset})
        embedding_sim_resp = requests.post(CALCULATE_SIMILARITY_EMBEDDING_URL, json={"query_vector": embedding_vector, "dataset": dataset})
        tfidf_sim_resp.raise_for_status()
        embedding_sim_resp.raise_for_status()
        tfidf_sims = tfidf_sim_resp.json().get("similarities", [])
        embedding_sims = embedding_sim_resp.json().get("similarities", [])

        if not tfidf_sims or not embedding_sims:
            raise HTTPException(status_code=500, detail="One or both similarity lists are missing.")


        if dataset == "antique":
            doc_data = pd.read_csv("antique_clean_data.csv")
        else:
            doc_data = pd.read_csv("corpus_clean_data.csv")

        # إنشاء قواميس التشابه
        tfidf_dict = {}
        embedding_dict = {}

        for idx, (_, row) in enumerate(doc_data.iterrows()):
            doc_id = str(row["doc_id"])
            if idx < len(tfidf_sims):
                tfidf_dict[doc_id] = tfidf_sims[idx]
            if idx < len(embedding_sims):
                embedding_dict[doc_id] = embedding_sims[idx]

        all_doc_ids = set(tfidf_dict.keys()).union(embedding_dict.keys())

        combined_scores = []
        for doc_id in all_doc_ids:
            tfidf_score = tfidf_dict.get(doc_id, 0)
            embed_score = embedding_dict.get(doc_id, 0)
            avg_score = (tfidf_score + embed_score) / 2
            combined_scores.append({"doc_id": doc_id, "score": avg_score})

        # 5. ترتيب النتائج وأخذ الأعلى
        combined_scores.sort(key=lambda x: x["score"], reverse=True)
        top_docs = combined_scores[:top_k]

        return {"result_docs": top_docs}

    except requests.HTTPError as http_err:
        raise HTTPException(status_code=response.status_code, detail=str(http_err))
    except requests.RequestException as req_err:
        raise HTTPException(status_code=500, detail=str(req_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      


@app.post("/search_query")
async def docs_ids_search(request: Request):
    try:
        request_body = await request.json()
        query = request_body.get("query")
        dataset = request_body.get("dataset")
        top_k = request_body.get("top_k") or 10
        method = request_body.get("method")

        # Validate input
        if not query or not dataset:
            raise HTTPException(status_code=400, detail="Query and dataset must be provided.")

        # Determine the correct log path based on the dataset
        user_queries_log_path = get_log_path(dataset)

          # Load existing user queries log if available
        if user_queries_log_path.exists():
            with open(user_queries_log_path, 'r') as f:
                user_queries_log = json.load(f)
        else:
            user_queries_log = []

        # Log the user's query
        user_queries_log.append(query)
        
   # Save the updated user queries log to file
        user_queries_log_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        with open(user_queries_log_path, 'w') as f:
            json.dump(user_queries_log, f)


        # Step 1: Process the text
        response = requests.post(QUERY_PROCESSING_URL, json={"text": query, })
        response.raise_for_status()
        processed_tokens = response.json().get('processed_tokens')

        if not processed_tokens:
            raise HTTPException(status_code=500, detail="Processed tokens not returned from text processing service.")

        # Step 2: Vectorize the query
        if method == "tfidf":
            response =   requests.post(VECTORIZE_QUERY_URL, json={"query_tokens": processed_tokens, "dataset": dataset})
        elif method in ("embedding", "chroma"):
            response =   requests.post(VECTORIZE_QUERY_EMBEDDING_URL, json={"query_tokens": processed_tokens, "dataset": dataset})
        else:
            raise HTTPException(status_code=400, detail="Invalid method.")

        response.raise_for_status()
        query_vector = response.json().get('query_vector')

        if not query_vector:
            raise HTTPException(status_code=500, detail="Query vector not returned from vectorization service.")


        
        if method == "chroma":
            
            response = requests.post(
                CALCULATE_SIMILARITY_CHROMA_URL,
                json={"query_vector": query_vector, "dataset": dataset, "top_k": top_k}
            )
            response.raise_for_status()
            result_docs = response.json().get('results')
            if not result_docs:
                raise HTTPException(status_code=500, detail="Result documents not returned from chroma similarity service.")
        else:
            if method == "tfidf":
                response = requests.post(CALCULATE_SIMILARITY_URL, json={"query_vector": query_vector, "dataset": dataset})
            elif method == "embedding":
                response = requests.post(CALCULATE_SIMILARITY_EMBEDDING_URL, json={"query_vector": query_vector, "dataset": dataset})
            else:
                raise HTTPException(status_code=400, detail="Invalid method.")

            response.raise_for_status()
            similarities = response.json().get('similarities')

            if similarities is None:
                raise HTTPException(status_code=500, detail="Similarities not returned from similarity calculation service.")

            response = requests.post(RANK_DOCUMENTS_URL, json={
                "similarities": similarities,
                "query_tokens": processed_tokens,
                "dataset": dataset,
                "top_k": top_k,
                "method": method
            })
            response.raise_for_status()
            result_docs = response.json().get('result_docs')
       

        if not result_docs:
            raise HTTPException(status_code=500, detail="Result documents not returned from ranking service.")

    except requests.HTTPError as http_err:
        raise HTTPException(status_code=response.status_code, detail=str(http_err))
    except requests.RequestException as req_err:
        raise HTTPException(status_code=500, detail=str(req_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return { "result_docs": result_docs }


@app.post("/get_document")
async def get_document(request: Request):
    try:
        request_body = await request.json()
        doc_id = request_body.get("doc_id")
        dataset = request_body.get("dataset")

        if not doc_id or not dataset:
            raise HTTPException(status_code=400, detail="Document ID and dataset must be provided.")

        # تحديد مسار ملف البيانات
        if dataset == "corpus":
            data_file = "corpus_clean_data.csv"
        elif dataset == "antique":
            data_file = "antique_clean_data.csv"
        else:
            raise HTTPException(status_code=400, detail="Invalid dataset.")

        # قراءة البيانات
        import pandas as pd
        import pathlib
        
        project_root = pathlib.Path(__file__).parent.parent
        data_path = project_root / data_file
        
        if not data_path.exists():
            raise HTTPException(status_code=404, detail=f"Data file {data_file} not found.")
        
        df = pd.read_csv(data_path)
        
        # البحث عن المستند
        doc_row = df[df['doc_id'] == doc_id]
        
        if doc_row.empty:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found in {dataset} dataset.")
        
        # استخراج المحتوى
        content = doc_row.iloc[0].get('content', '')
        if pd.isna(content):
            content = 'لا يوجد محتوى متاح'
        
        return {
            "doc_id": doc_id,
            "dataset": dataset,
            "content": str(content)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
