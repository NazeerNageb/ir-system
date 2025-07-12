import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tqdm import tqdm
import json

# تحميل الموارد الضرورية لـ NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

class PreprocessService:
    
    
    def preprocess_text(self, text):
        """Preprocess single text"""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"<.*?>|\[.*?\]", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\d+", "", text)

        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens]
        return tokens
    
    def preprocess_documents(self , documents , doc_ids):
        """Preprocess multiple documents"""
        processed_docs = []
        all_tokens = []
        doc_ids_list = []
        original_docs = []

        for i, text in enumerate(tqdm(documents, desc="معالجة المستندات", unit="مستند")):
            tokens = self.preprocess_text(text)
            if tokens:
                all_tokens.append(tokens)            
                processed_docs.append(' '.join(tokens))
                doc_ids_list.append(doc_ids[i])
                original_docs.append(text)

        return processed_docs, all_tokens, doc_ids_list, original_docs


    
