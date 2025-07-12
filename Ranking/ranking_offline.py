import pathlib
from Ranking.ranking_service import RankingService

class RankingOffline:
    def __init__(self):
        self.ranking_service = RankingService()
        project_root = pathlib.Path(__file__).parent.parent
        
   
        self.ranking_service.load_corpus(project_root / "corpus_clean_data.csv")
        self.ranking_service.load_antique(project_root / "antique_clean_data.csv")
        
        # self.ranking_service.load_inverted_indexes(project_root)
        self.ranking_service.load_chroma_embedding()

    def rank_documents(self, similarities, query_tokens, dataset, top_k, method):
        data = self.ranking_service.corpus if dataset == "corpus" else self.ranking_service.antique

        if method == "embedding":
           
            if similarities is None or (hasattr(similarities, '__len__') and len(similarities) == 0):
               
                return {"result_docs": [], "status": "empty_similarities"}
            
            if data.empty:
              
                return {"result_docs": [], "status": "empty_dataset"}
            
     
            if len(similarities) < len(data):
                print(f"⚠️ [WARNING] عدد قيم التشابه ({len(similarities)}) أقل من عدد المستندات ({len(data)})")
            
           
            top_results = sorted(
                [(str(row["doc_id"]), float(similarities[idx])) for idx, row in data.iterrows() if idx < len(similarities)],
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
         
            return {"result_docs": top_results, "status": "success"}
        else:
            inverted_index = self.ranking_service.corpus_inverted_index if dataset == "corpus" else self.ranking_service.antique_inverted_index 

            if data.empty:
             
                return {"result_docs": [], "status": "no_dataset_found"}

            candidate_doc_ids = set()
            for token in query_tokens:
                postings = inverted_index.get(token, [])
             
                candidate_doc_ids.update(doc_id for doc_id, _ in postings)


            if not candidate_doc_ids:
             
                return {"result_docs": [], "status": "no_candidates_found"}

            doc_id_to_index = {str(row["doc_id"]): idx for idx, row in data.iterrows()}

            result_candidates = []
            for doc_id in candidate_doc_ids:
                index = doc_id_to_index.get(str(doc_id))
                if index is not None and index < len(similarities):
                    sim = similarities[index]
                    result_candidates.append((doc_id, sim))
                else:
                     print(f"⚠️ doc_id '{doc_id}' ليس له index مطابق أو خارج النطاق")

   

            if not result_candidates:
                
                return {"result_docs": [], "status": "no_similarities_found"}

            
            top_results = sorted(result_candidates, key=lambda x: x[1], reverse=True)[:top_k]
          
            for i, (doc_id, score) in enumerate(top_results, start=1):
                print(f"  {i}. {doc_id}: {score}")

            return {"result_docs": top_results, "status": "success"}

