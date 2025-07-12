import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
import json
import joblib
import os
import numpy as np
from collections import defaultdict

def identity(x):
        return x

def simple_tokenizer(x):
        return x.split()    


class TfidfService:
    def __init__(self):
        self.antique_vectorizer = None
        self.antique_tfidf_matrix = None
        self.corpus_vectorizer = None  
        self.corpus_tfidf_matrix = None
        self.inverted_index = None
        self.docs_texts_antique = None
        self.doc_ids_antique = None
        self.processed_docs_antique = None
        self.tokens_antique = None
        self.docs_texts_corpus = None
        self.doc_ids_corpus = None
        self.processed_docs_corpus = None
        self.tokens_corpus = None

    

    def fit_transform_documents(self, documents):

        self.vectorizer = TfidfVectorizer(
            tokenizer=simple_tokenizer,
            preprocessor=identity,
            lowercase=False,
            token_pattern=None,
            max_df=0.9,                   
            min_df=10,                    
            max_features=100000,          
            ngram_range=(1, 2),           
            norm='l2',                    
            sublinear_tf=True,            
            smooth_idf=True              

        )

        tfidf_matrix = self.vectorizer.fit_transform(documents)
        return tfidf_matrix

    def save_tfidf_matrix(self, tfidf_matrix, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "wb") as f:
                joblib.dump(tfidf_matrix, f)
        except Exception as e:
            print(f"❌ خطأ في حفظ TF-IDF matrix: {e}")
            raise

    def save_vectorizer(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "wb") as f:
                joblib.dump(self.vectorizer, f)
        except Exception as e:
            print(f"❌ خطأ في حفظ Vectorizer: {e}")
            raise

    def load_tfidf_matrix(self, file_path):
        try:
            with open(file_path, "rb") as f:
                tfidf_matrix = joblib.load(f)
            return tfidf_matrix
        except Exception as e:
            print(f"❌ خطأ في تحميل TF-IDF matrix: {e}")
            raise

    def load_vectorizer(self, file_path):
        """Load vectorizer from file"""
        try:
            with open(file_path, "rb") as f:
                vectorizer = joblib.load(f)
            return vectorizer
        except Exception as e:
            print(f"❌ خطأ في تحميل Vectorizer: {e}")
            raise

    def preload(self, antique_folder, corpus_folder):
        try:
            self.antique_vectorizer = self.load_vectorizer(antique_folder  / "vectorizer.joblib")
            self.antique_tfidf_matrix = self.load_tfidf_matrix(antique_folder / "tfidf_matrix.joblib")
            self.corpus_vectorizer = self.load_vectorizer(corpus_folder / "vectorizer.joblib")
            self.corpus_tfidf_matrix = self.load_tfidf_matrix(corpus_folder / "tfidf_matrix.joblib") 

        except Exception as e:
            print(f"Error during preloading: {e}")
   

    def process_csv_file(self, input_file_path, tfidf_matrix_path, vectorizer_path , dataset):
        try:
           
            if not os.path.exists(input_file_path):
                raise FileNotFoundError(f"الملف المدخل غير موجود: {input_file_path}")
            
   
            
            documents = self.processed_docs_antique if dataset == "antique" else self.processed_docs_corpus
          
            

            tfidf_matrix = self.fit_transform_documents(documents)
            
            self.save_tfidf_matrix(tfidf_matrix, tfidf_matrix_path)
            self.save_vectorizer(vectorizer_path)

            
        except Exception as e:
            print(f"❌ خطأ في معالجة الملف: {e}")
            raise
    def vectorize_query(self, query_tokens, dataset):
        if dataset == "antique":
            vectorizer = self.antique_vectorizer
        elif dataset == "corpus":
            vectorizer = self.corpus_vectorizer
        else:
            raise ValueError(f"Dataset {dataset} not found")
        
        processed_query_string = ' '.join(query_tokens)
        query_vector = vectorizer.transform([processed_query_string])
        return {
            "shape": query_vector.shape,
            "data": query_vector.data.tolist(),
            "indices": query_vector.indices.tolist(),
            "indptr": query_vector.indptr.tolist()
        }
    
    # def vectorize_query_evaluation(self, query_tokens, dataset):
    #     if dataset == "antique":
    #         vectorizer = self.antique_vectorizer
    #     elif dataset == "corpus":
    #         vectorizer = self.corpus_vectorizer
    #     else:
    #         raise ValueError(f"Dataset {dataset} not found")
        
    #     if vectorizer is None:
    #         raise ValueError(f"Vectorizer for dataset {dataset} is not loaded")
    #     processed_query_string = ' '.join(query_tokens)
    #     query_vector = vectorizer.transform([processed_query_string])
    #     return query_vector    

    
    def calculate_similarity_api(self, query_vector, dataset):
            if dataset == "antique":
                tfidf_matrix = self.antique_tfidf_matrix
            elif dataset == "corpus":
                tfidf_matrix = self.corpus_tfidf_matrix
            else:
                raise ValueError(f"Dataset {dataset} not found")
            similarity_matrix = cosine_similarity(query_vector, tfidf_matrix)
            return similarity_matrix.flatten().tolist()

        

    def build_inverted_index_tfidf(self, dataset):
        if dataset == "antique":
            tfidf_matrix = self.antique_tfidf_matrix
            vectorizer = self.antique_vectorizer
            doc_ids = self.doc_ids_antique
        elif dataset == "corpus":
            tfidf_matrix = self.corpus_tfidf_matrix
            vectorizer = self.corpus_vectorizer
            doc_ids = self.doc_ids_corpus
        else:
            raise ValueError(f"Dataset {dataset} not found. Available datasets: antique, corpus")



        # Validation
        if tfidf_matrix is None:
            raise ValueError(f"TF-IDF matrix for dataset {dataset} is not loaded. Call preload() first.")
        if vectorizer is None:
            raise ValueError(f"Vectorizer for dataset {dataset} is not loaded. Call preload() first.")
        if doc_ids is None:
            raise ValueError(f"Document IDs for dataset {dataset} are not loaded. Call read_data() first.")
        if len(doc_ids) != tfidf_matrix.shape[0]:
            raise ValueError(f"Mismatch between doc_ids length ({len(doc_ids)}) and TF-IDF matrix rows ({tfidf_matrix.shape[0]})")

        feature_names = vectorizer.get_feature_names_out()
        inverted_index = defaultdict(list)

        for doc_idx, doc_id in enumerate(doc_ids):
            row = tfidf_matrix[doc_idx].tocoo()
            for term_idx, score in zip(row.col, row.data):
                term = feature_names[term_idx]
                inverted_index[term].append((doc_id, float(score)))  # 👈 التحويل لـ float لتفادي مشاكل joblib

        print(f"✅ Inverted index built successfully with {len(inverted_index)} unique terms")
        return inverted_index
    
    def save_inverted_index(self, inverted_index, output_path):
        try:
            joblib.dump(inverted_index, output_path)
            print(f"✅ تم حفظ inverted index في: {output_path}")
        except Exception as e:
            print(f"❌ خطأ في حفظ inverted index: {e}")
            raise
        
    def read_data(self, input_file , dataset):
        """Read processed documents from CSV file"""
        try:
            # Check if file exists
            if not os.path.exists(input_file):
                print(f"❌ File not found: {input_file}")
                print(f"   Current working directory: {os.getcwd()}")
                print(f"   Absolute path: {os.path.abspath(input_file)}")
                return False
            
            print(f"📖 Reading data from: {input_file}")
            df = pd.read_csv(input_file)
            
            # Check required columns
            required_columns = ['doc_id', 'original_text', 'clean_text', 'tokens']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                print(f"   Available columns: {list(df.columns)}")
                return False
            
            print(f"✅ Found {len(df)} documents in {input_file}")
            
            # Apply the same filtering logic as in process_csv_file
            print("🔧 Applying same filtering as TF-IDF processing...")
            df['clean_text'] = df['clean_text'].fillna('')
            df = df[df['clean_text'].str.strip() != '']
            
            if len(df) == 0:
                print(f"❌ No valid documents after filtering")
                return False
                
            print(f"✅ After filtering: {len(df)} documents remain")

            # تحويل نص JSON في tokens إلى قائمة
            df['tokens'] = df['tokens'].fillna('[]')  # إذا فيه قيم ناقصة، استبدلها بقائمة فارغة

            docs_tokens = [json.loads(text) for text in df['tokens']]
            docs_texts = df['original_text'].tolist()
            processed_docs = df['clean_text'].tolist()
            doc_ids = df['doc_id'].tolist()
            
            
            # Update instance variables
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
            
            print(f"✅ Successfully loaded {len(doc_ids)} documents for dataset '{dataset}'")
            return True
            
        except FileNotFoundError:
            print(f"❌ File not found: {input_file}")
            return False
        except Exception as e:
            print(f"❌ Error reading file {input_file}: {e}")
            return False    

    