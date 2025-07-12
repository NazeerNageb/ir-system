import pathlib
import numpy as np
from embedding.embedding_service import EmbeddingService
    

class EmbeddingOffline:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        
     
        antique_folder = pathlib.Path("files/antique")
        corpus_folder = pathlib.Path("files/corpus")
        
        
        self.embedding_service.preload(antique_folder, corpus_folder)

        # antique_csv = pathlib.Path("antique_clean_data.csv")
        # corpus_csv = pathlib.Path("corpus_clean_data.csv")
  
        # self.embedding_service.read_data(antique_csv, "antique")
        # self.embedding_service.read_data(corpus_csv, "corpus")
        
        self.embedding_service.load_chroma_embedding()   

    def load_embedding(self, filePath):
        self.bert_embeddings = np.load(filePath)
        return self.bert_embeddings
    

    def process_embedding(self, dataset, filePath):
       
        
        self.embedding_service.process_embedding(dataset)
        self.embedding_service.save_embedding(filePath)
       
        return self.bert_embeddings
    
    def chroma_embedding(self, dataset):
        self.embedding_service.chroma_embedding(dataset)


    def vectorize_query(self, query_tokens, dataset):

        
        
        query_text = ' '.join(query_tokens)

        
        query_vector = self.embedding_service.queryEncode(query_text)

        
        return query_vector
    
    def calculate_similarity_embedding(self, query_vector, dataset):
  
        
        similarities = self.embedding_service.consine_similarity(query_vector, dataset)
  
        if similarities is not None:
            print(f" [EMBEDDING_OFFLINE] Similarities shape: {similarities.shape}, min: {similarities.min()}, max: {similarities.max()}")
        else:
            print("❌ [EMBEDDING_OFFLINE] Similarities is None!")
        
        return similarities
    
    def calculate_similarity_chroma(self, query_vector, dataset, top_k):
        result = self.embedding_service.query_chroma(query_vector, dataset, top_k)
        return result
  
