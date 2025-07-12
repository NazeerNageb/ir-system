from contextlib import asynccontextmanager
import pathlib
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import math
import joblib

# خدمة الترتيب
class RankingService:
    def __init__(self):
        self.corpus = pd.DataFrame()
        self.antique = pd.DataFrame()
        self.corpus_inverted_index = {}
        self.antique_inverted_index = {}
        self.antique_chroma_db = {}
        self.corpus_chroma_db = {}
        # إضافة خرائط محسنة مسبقاً
        self.corpus_doc_id_to_index = {}
        self.antique_doc_id_to_index = {}


    def load_corpus(self, file_path):
        print("🔄 Loading corpus data...")
        # تحميل الأعمدة المطلوبة فقط
        self.corpus = pd.read_csv(file_path, usecols=['doc_id'])
        # إنشاء خريطة doc_id إلى index مرة واحدة
        self.corpus_doc_id_to_index = {str(row["doc_id"]): idx for idx, row in self.corpus.iterrows()}
        print(f"✅ Loaded {len(self.corpus)} corpus documents")

    def load_antique(self, file_path):
        print("🔄 Loading antique data...")
        # تحميل الأعمدة المطلوبة فقط
        self.antique = pd.read_csv(file_path, usecols=['doc_id'])
        # إنشاء خريطة doc_id إلى index مرة واحدة
        self.antique_doc_id_to_index = {str(row["doc_id"]): idx for idx, row in self.antique.iterrows()}
        print(f"✅ Loaded {len(self.antique)} antique documents")

    def load_inverted_indexes(self, project_root):
        print(project_root)
        try:
            corpus_index_path = project_root / "files" / "corpus" / "corpus_inverted_index.joblib"
            antique_index_path = project_root / "files" / "antique" / "antique_inverted_index.joblib"
            print(corpus_index_path)
            print(antique_index_path)

            if corpus_index_path.exists():
                self.corpus_inverted_index = joblib.load(open(corpus_index_path, 'rb'))
            if antique_index_path.exists():
                self.antique_inverted_index = joblib.load(open(antique_index_path, 'rb'))
                
        except Exception as e:
            print(f"Error loading inverted indexes: {e}")


    def load_chroma_embedding(self):
            client = chromadb.PersistentClient(path="antique_embeddings_chroma_db")
            self.antique_chroma_db = client.get_collection(name="w2v_embeddings_antique") 

            client = chromadb.PersistentClient(path="corpus_embeddings_chroma_db")
            self.corpus_chroma_db = client.get_collection(name="w2v_embeddings_corpus")
            print("✅ Chroma embeddings loaded successfully")

    def rank_and_sort(self, similarity_scores):
        ranks = {
            i: similarity_scores[i] for i in range(len(similarity_scores))
            if similarity_scores[i] is not None and not math.isnan(similarity_scores[i]) and not math.isinf(similarity_scores[i])
        }
        return sorted(ranks.items(), key=lambda x: x[1], reverse=True)

    def get_doc_id_to_index_map(self, dataset):
        """الحصول على خريطة doc_id إلى index المحسنة مسبقاً"""
        return self.corpus_doc_id_to_index if dataset == "corpus" else self.antique_doc_id_to_index

# إنشاء كائن الخدمة
ranking_service = RankingService()

# نموذج الطلب الجديد
class RankRequest(BaseModel):
    similarities: List[float]         
    query_tokens: List[str]                
    dataset: str
    top_k: Optional[int] = 10
    method: Optional[str] = "tfidf"

# تهيئة التطبيق وتحميل البيانات عند البدء
@asynccontextmanager
async def lifespan(app: FastAPI):
    project_root = pathlib.Path(__file__).parent.parent
    try:
        ranking_service.load_corpus(project_root / "corpus_clean_data.csv")
        ranking_service.load_antique(project_root / "antique_clean_data.csv")
        ranking_service.load_inverted_indexes(project_root)
        ranking_service.load_chroma_embedding()
    except Exception as e:
        print(f"Startup error: {e}")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/rank_documents/")
async def rank_documents(request: RankRequest):
    data = ranking_service.corpus if request.dataset == "corpus" else ranking_service.antique

    if request.method == "embedding":
        # تحسين: استخدام الخريطة المحسنة مسبقاً
        doc_id_to_index = ranking_service.get_doc_id_to_index_map(request.dataset)
        
        # تحسين: تجنب iterrows() البطيء
        doc_ids = data['doc_id'].astype(str).tolist()
        top_results = []
        
        for idx, doc_id in enumerate(doc_ids):
            if idx < len(request.similarities):
                top_results.append((doc_id, request.similarities[idx]))
        
        # ترتيب واختيار top_k
        top_results.sort(key=lambda x: x[1], reverse=True)
        return {"result_docs": top_results[:request.top_k], "status": "success"}




    inverted_index = ranking_service.corpus_inverted_index if request.dataset == "corpus" else ranking_service.antique_inverted_index 

    if data.empty:
       
        raise HTTPException(status_code=400, detail=f"Dataset {request.dataset} is empty or not loaded")

    # تحديد المستندات المرشحة
    candidate_doc_ids = set()
    for token in request.query_tokens:
        postings = inverted_index.get(token, [])
       
        candidate_doc_ids.update(doc_id for doc_id, _ in postings)

   

    if not candidate_doc_ids:
       
        return {"result_docs": [], "status": "no_candidates_found"}

    # استخدام الخريطة المحسنة مسبقاً
    doc_id_to_index = ranking_service.get_doc_id_to_index_map(request.dataset)

    result_candidates = []
    for doc_id in candidate_doc_ids:
        index = doc_id_to_index.get(str(doc_id))
        if index is not None and index < len(request.similarities):
            sim = request.similarities[index]
            result_candidates.append((doc_id, sim))
        else:
            print(f"⚠️ doc_id '{doc_id}' ليس له index مطابق أو خارج النطاق")



    if not result_candidates:
        return {"result_docs": [], "status": "no_similarities_found"}

    # ترتيب النتائج
    top_results = sorted(result_candidates, key=lambda x: x[1], reverse=True)[:request.top_k]
    for i, (doc_id, score) in enumerate(top_results, start=1):
        print(f"  {i}. {doc_id}: {score}")

    return {"result_docs": top_results, "status": "success"}


# تشغيل التطبيق
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
