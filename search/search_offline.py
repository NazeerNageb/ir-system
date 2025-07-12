import requests
import json
import pathlib
from preprocess.preprocess import PreprocessService
from tfidf.tf_idf_offline import TfidfOffline
from embedding.embedding_offline import EmbeddingOffline
from Ranking.ranking_offline import RankingOffline




class SearchOffline:

    preprocess_service = PreprocessService()
    tfidf_offline = TfidfOffline()
    embedding_offline = EmbeddingOffline()
    ranking_offline = RankingOffline()
    
    async def docs_ids_search(self, query, dataset, top_k, method):
        try:


            # Step 1: Process the text
            response = self.preprocess_service.preprocess_text(query)
            response.raise_for_status()
            processed_tokens = response.json().get('processed_tokens')

            if not processed_tokens:
                return {"error": "Processed tokens not returned from text processing service.", "success": False}

            # Step 2: Vectorize the query
            if method == "tfidf":
                response = self.tfidf_offline.vectorize_query(processed_tokens, dataset)
            elif method in ("embedding", "chroma"):
                response = self.embedding_offline.vectorize_query(processed_tokens, dataset)
            else:
                return {"error": "Invalid method.", "success": False}

            response.raise_for_status()
            query_vector = response.json().get('query_vector')

            if not query_vector:
                return {"error": "Query vector not returned from vectorization service.", "success": False}

            if method == "chroma":
                response = self.embedding_offline.calculate_similarity_chroma(query_vector, dataset, top_k)
                response.raise_for_status()
                result_docs = response.json().get('results')
                if not result_docs:
                    return {"error": "Result documents not returned from chroma similarity service.", "success": False}
            else:
                if method == "tfidf":
                    response = self.tfidf_offline.calculate_similarity(query_vector, dataset)
                elif method == "embedding":
                    response = self.embedding_offline.calculate_similarity_embedding(query_vector, dataset)
                else:
                    return {"error": "Invalid method.", "success": False}

                response.raise_for_status()
                similarities = response.json().get('similarities')

                if similarities is None:
                    return {"error": "Similarities not returned from similarity calculation service.", "success": False}

                response = self.ranking_offline.rank_documents(similarities, processed_tokens, dataset, top_k, method)
               
                response.raise_for_status()
                result_docs = response.json().get('result_docs')

            if not result_docs:
                return {"error": "Result documents not returned from ranking service.", "success": False}

        except requests.HTTPError as http_err:
            return {"error": f"HTTP Error: {str(http_err)}", "success": False}
        except requests.RequestException as req_err:
            return {"error": f"Request Error: {str(req_err)}", "success": False}
        except Exception as e:
            return {"error": f"Unexpected Error: {str(e)}", "success": False}
        
        return {"result_docs": result_docs, "success": True}