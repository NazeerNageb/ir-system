from tfidf.tf_idf_service import TfidfService
import pathlib

class TfidfOffline:
    def __init__(self):
        self.tfidf_service = TfidfService()
        

        antique_folder = pathlib.Path("files/antique")
        corpus_folder = pathlib.Path("files/corpus")
        
        print(antique_folder)
        print(corpus_folder)
        
        self.tfidf_service.preload(antique_folder, corpus_folder)

        # antique_csv = pathlib.Path("antique_clean_data.csv")
        # corpus_csv = pathlib.Path("corpus_clean_data.csv")
  
        # self.tfidf_service.read_data(antique_csv, "antique")
        # self.tfidf_service.read_data(corpus_csv, "corpus")

    def process_csv_file(self, input_file_path, tfidf_matrix_path, vectorizer_path , dataset):
        """
        Process a CSV file to generate TF-IDF matrix and vectorizer.
        
        Args:
            input_file_path (str): Path to the input CSV file
            tfidf_matrix_path (str): Path to save the TF-IDF matrix
            vectorizer_path (str): Path to save the vectorizer
        """
        try:
            self.tfidf_service.process_csv_file(input_file_path, tfidf_matrix_path, vectorizer_path , dataset)
        except Exception as e:
            print(f"Error processing CSV file: {e}")
            raise

    def build_inverted_index_tfidf(self, dataset  , output_path):
        inverted_index = self.tfidf_service.build_inverted_index_tfidf(dataset)
        self.tfidf_service.save_inverted_index(inverted_index, output_path)
        return inverted_index
    
    def vectorize_query(self, query_tokens, dataset):
        query_vector = self.tfidf_service.vectorize_query(query_tokens, dataset)
        return query_vector
    
    def calculate_similarity(self, query_vector, dataset):
        similarities = self.tfidf_service.calculate_similarity_api(query_vector, dataset)
        return similarities



