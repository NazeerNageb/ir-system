import json
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch import cosine_similarity
import torch
import chromadb
from tqdm import tqdm

class EmbeddingService:
    def __init__(self):
        self.docs_texts_antique = None
        self.doc_ids_antique = None
        self.processed_docs_antique = None
        self.tokens_antique = None
        self.docs_texts_corpus = None
        self.doc_ids_corpus = None
        self.processed_docs_corpus = None
        self.tokens_corpus = None
        self.bert_embeddings = None
        # embeddings
        self.antique_bert_embeddings = None
        self.corpus_bert_embeddings = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.chroma_collection_antique = None
        self.chroma_collection_corpus = None


                
    def process_embedding(self, dataset):       
            documents = self.processed_docs_antique if dataset == "antique" else self.processed_docs_corpus
            
                     
            model = SentenceTransformer('all-MiniLM-L6-v2')  

            
            self.bert_embeddings =  model.encode(documents, show_progress_bar=True, batch_size=64)
            return self.bert_embeddings



    def save_embedding(self, filePath):

        np.save(filePath, self.bert_embeddings)

    def load_embedding(self, filePath ):
        bert_embeddings = np.load(filePath)
        return bert_embeddings
    

    def preload(self, antique_folder, corpus_folder):
        try:
          
            self.antique_bert_embeddings = self.load_embedding(antique_folder / "bert_embeddings.npy")
 
            self.corpus_bert_embeddings = self.load_embedding(corpus_folder / "bert_embeddings.npy") 

        except Exception as e:
            self.antique_bert_embeddings = None
            self.corpus_bert_embeddings = None

    def queryEncode (self, query):

         
        query_embedding = self.model.encode([query])
        
        return query_embedding

   
    
    def query_chroma(self, query_embedding, dataset, top_k): 
        if dataset == "antique":
            collection = self.chroma_collection_antique
        else:
            collection = self.chroma_collection_corpus

        if collection is None:
            raise ValueError(f"Chroma collection for dataset '{dataset}' is not loaded")

        return collection.query(
            query_embeddings=query_embedding,
            n_results=top_k  
        )
   

    def consine_similarity(self, query_embedding, dataset):
      
        
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
  

        if dataset == "antique":
            doc_embeddings = self.antique_bert_embeddings
           
        else:
            doc_embeddings = self.corpus_bert_embeddings
           

        if doc_embeddings is None:
            print("❌ Error: Document embeddings not loaded")
            return None

       
        doc_tensor = torch.tensor(doc_embeddings, dtype=torch.float32)

        if query_tensor.ndim == 1:
            query_tensor = query_tensor.unsqueeze(0)  
           

        similarities = torch.nn.functional.cosine_similarity(query_tensor, doc_tensor)
        result = similarities.detach().numpy()
       
        return result
        
        
  

    def chroma_embedding(self, dataset):
      if dataset == "antique":
        doc_embeddings = self.antique_bert_embeddings
        doc_ids = self.doc_ids_antique
        docs_texts = self.docs_texts_antique
        persist_dir = "antique_embeddings_chroma_db"
      else:
        doc_embeddings = self.corpus_bert_embeddings
        doc_ids = self.doc_ids_corpus
        docs_texts = self.docs_texts_corpus
        persist_dir = "corpus_embeddings_chroma_db"

      client = chromadb.PersistentClient(path=persist_dir)
     
      w2v_collection = client.get_or_create_collection(name=f"w2v_embeddings_{dataset}")

      for i, doc_id in enumerate(tqdm(doc_ids, desc="Adding embeddings ")):
        embedding = doc_embeddings[i].tolist()
        doc_text = docs_texts[i]
        if isinstance(doc_text, str) and doc_text.strip():
            w2v_collection.add(
                ids=[str(doc_id)],
                embeddings=[embedding],
                metadatas=[{"doc_id": doc_id}],
                documents=[doc_text]
            )
        else:
            print(f"⚠️ Skipping doc_id {doc_id} due to invalid or empty text: {doc_text}")

     

    def load_chroma_embedding(self):
        try:
           
            
    
            client = chromadb.PersistentClient(path="antique_embeddings_chroma_db")
            self.chroma_collection_antique = client.get_collection(name="w2v_embeddings_antique")
           
            
     
            client = chromadb.PersistentClient(path="corpus_embeddings_chroma_db")
            self.chroma_collection_corpus = client.get_collection(name="w2v_embeddings_corpus")
           
            
           
            
        except Exception as e:
           
           
            
            try:
       
                client = chromadb.PersistentClient(path="antique_embeddings_chroma_db")
                self.chroma_collection_antique = client.get_or_create_collection(name="w2v_embeddings_antique")
               
                
                client = chromadb.PersistentClient(path="corpus_embeddings_chroma_db")
                self.chroma_collection_corpus = client.get_or_create_collection(name="w2v_embeddings_corpus")

                
            except Exception as e2:
               
                self.chroma_collection_antique = None
                self.chroma_collection_corpus = None



    def read_data(self, input_file , dataset):
        """Read processed documents from CSV file"""
        try:

            if not os.path.exists(input_file):
                return False
            
           
            df = pd.read_csv(input_file)
            

            required_columns = ['doc_id', 'original_text', 'clean_text', 'tokens']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False
            
           
            
            # Apply the same filtering logic as in process_csv_file
           
            df['clean_text'] = df['clean_text'].fillna('')
            df = df[df['clean_text'].str.strip() != '']
            
            if len(df) == 0:
                return False
                

            df['tokens'] = df['tokens'].fillna('[]')  

            docs_tokens = [json.loads(text) for text in df['tokens']]
            docs_texts = df['original_text'].tolist()
            processed_docs = df['clean_text'].tolist()
            doc_ids = df['doc_id'].tolist()
            
            
          
            if dataset == "antique":
                self.docs_texts_antique = docs_texts
                self.doc_ids_antique = doc_ids
                self.processed_docs_antique = processed_docs
                self.tokens_antique = docs_tokens
            elif dataset == "corpus":
                self.docs_texts_corpus = docs_texts
                self.doc_ids_corpus = doc_ids
                self.processed_docs_corpus = processed_docs
                self.tokens_corpus = docs_tokens
            
           
            return True
            
        except FileNotFoundError:
            return False
        except Exception as e:
            return False    




















# def read_data(self, input_file , dataset):
#         """Read processed documents from CSV file"""
#         try:
#             # Check if file exists
#             if not os.path.exists(input_file):
#                 print(f"❌ File not found: {input_file}")
#                 print(f"   Current working directory: {os.getcwd()}")
#                 print(f"   Absolute path: {os.path.abspath(input_file)}")
#                 return False
            
#             print(f"📖 Reading data from: {input_file}")
#             df = pd.read_csv(input_file)
            
#             # Check required columns
#             required_columns = ['doc_id', 'original_text', 'clean_text', 'tokens']
#             missing_columns = [col for col in required_columns if col not in df.columns]
#             if missing_columns:
#                 print(f"❌ Missing required columns: {missing_columns}")
#                 print(f"   Available columns: {list(df.columns)}")
#                 return False
            
#             print(f"✅ Found {len(df)} documents in {input_file}")
            
#             # Apply the same filtering logic as in process_csv_file
#             print("🔧 Applying same filtering as TF-IDF processing...")
#             df['clean_text'] = df['clean_text'].fillna('')
#             df = df[df['clean_text'].str.strip() != '']
            
#             if len(df) == 0:
#                 print(f"❌ No valid documents after filtering")
#                 return False
                
#             print(f"✅ After filtering: {len(df)} documents remain")

#             # تحويل نص JSON في tokens إلى قائمة
#             df['tokens'] = df['tokens'].fillna('[]')  # إذا فيه قيم ناقصة، استبدلها بقائمة فارغة

#             docs_tokens = [json.loads(text) for text in df['tokens']]
#             docs_texts = df['original_text'].tolist()
#             processed_docs = df['clean_text'].tolist()
#             doc_ids = df['doc_id'].tolist()
            
            
#             # Update instance variables
#             if dataset == "antique":
#                 self.docs_texts_antique = docs_texts
#                 self.doc_ids_antique = doc_ids
#                 self.processed_docs_antique = processed_docs
#                 self.tokens_antique = docs_tokens
#             elif dataset == "corpus":
#                 self.docs_texts_corpus = docs_texts
#                 self.doc_ids_corpus = doc_ids
#                 self.processed_docs_corpus = processed_docs
#                 self.tokens_corpus = docs_tokens
            
#             print(f"✅ Successfully loaded {len(doc_ids)} documents for dataset '{dataset}'")
#             return True
            
#         except FileNotFoundError:
#             print(f"❌ File not found: {input_file}")
#             return False
#         except Exception as e:
#             print(f"❌ Error reading file {input_file}: {e}")
#             return False    