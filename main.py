# from convert_doc import ConvertDoc 
# from data_function import  DataFunction 
# from preprocess.preprocess import PreprocessService
# from tfidf.tf_idf_offline import TfidfOffline
from embedding.embedding_offline import EmbeddingOffline    
import pathlib



################################### convert dataset antique to utf8 ###################################
# input1_file = 'antique_docs.tsv'           # الملف الأصلي
# output1_file = 'antique_docs_utf8.tsv'    # الملف الجديد بعد التحويل
# ConvertDoc.convertToUtf8(input1_file, output1_file)


################################### preprocess dataset antique ###################################
    
# antiquedata = DataFunction()
# antiquedata.load_documents("antique_docs_utf8.tsv" , "antique")

# preprocess_service = PreprocessService()
# processed_docs, all_tokens, doc_ids, original_docs = preprocess_service.preprocess_documents(antiquedata.docs_texts , antiquedata.doc_ids)

# antiquedata.processed_docs = processed_docs
# antiquedata.tokens = all_tokens
# antiquedata.doc_ids = doc_ids
# antiquedata.docs_texts = original_docs
# antiquedata.save_to_csv("antique_clean_data.csv")

################################### convert dataset corpus to json ###################################

# input_file = 'corpus.jsonl'
# output_file = 'corpus.json'
# ConvertDoc.convert_jsonl_to_json(input_file, output_file)

################################### convert dataset corpus to tsv ###################################
# input_file = 'corpus.json'
# output_file = 'corpus.tsv'
# ConvertDoc.json_to_tsv(input_file, output_file)



################################### preprocess dataset corpus ###################################



# corpusdata = DataFunction()
# corpusdata.load_documents("corpus_docs_utf8.tsv" , "corpus")

# preprocess_service = PreprocessService()
# processed_docs, all_tokens, doc_ids, original_docs = preprocess_service.preprocess_documents(corpusdata.docs_texts , corpusdata.doc_ids)

# corpusdata.processed_docs = processed_docs
# corpusdata.tokens = all_tokens
# corpusdata.doc_ids = doc_ids
# corpusdata.docs_texts = original_docs
# corpusdata.save_to_csv('corpus_clean_data.csv')



################################### read csv file ###################################
# corpusdata = DataFunction()
# corpusdata.read_csv("corpus_clean_data.csv")

# print(corpusdata.processed_docs[0])
# print(corpusdata.tokens[0])
# print(corpusdata.docs_texts[0])
# print(corpusdata.doc_ids[0])


################################### tfidf offline ################################### 
# service = TfidfOffline() 

# # Define file paths
# cwd = pathlib.Path().cwd()
# input_file_path = cwd / "corpus_clean_data.csv"
# tfidf_matrix_path = cwd / "files" / "corpus" / "tfidf_matrix.joblib"
# vectorizer_path = cwd / "files" / "corpus" / "vectorizer.joblib"
# # Process csv file
# service.process_csv_file(input_file_path, tfidf_matrix_path, vectorizer_path , "corpus")
# print("TF-IDF processing of CSV file complete.")

# input_file_path = cwd / "antique_clean_data.csv"
# tfidf_matrix_path = cwd / "files" / "antique" / "tfidf_matrix.joblib"
# vectorizer_path = cwd / "files" / "antique" / "vectorizer.joblib"


# # Process csv file
# service.process_csv_file(input_file_path, tfidf_matrix_path, vectorizer_path , "antique")
# print("TF-IDF processing of CSV file complete.")

 
################################### build inverted index ###################################
# inverted_index = service.build_inverted_index_tfidf("antique" , cwd / "files" / "antique" / "antique_inverted_index.joblib")
# inverted_index = service.build_inverted_index_tfidf("corpus" , cwd / "files" / "corpus" / "corpus_inverted_index.joblib")
# print(inverted_index)




    ######################                bert embedding                  #################################################

# bertembedding = EmbeddingOffline()    

# cwd = pathlib.Path().cwd()

# embeddingPathantique = cwd / "files" / "antique" / "bert_embeddings.npy"
# embeddingPathcorpus    = cwd / "files" / "corpus" / "bert_embeddings.npy"


# bertembedding.process_embedding('antique', embeddingPathantique)
# bertembedding.process_embedding('corpus', embeddingPathcorpus)




    ######################                chroma embedding                  #################################################

chromaembedding = EmbeddingOffline()    

# chromaembedding.chroma_embedding("antique")
chromaembedding.chroma_embedding("corpus")




