import pandas as pd
import json
import csv
# from preprocess_service import preprocess

class DataFunction:
    def __init__(self):
        
        self.docs_texts = []
        self.doc_ids = []
        self.processed_docs = []
        self.tokens = []
    
    def load_documents(self, input_file , dataset_name):
        """Load and preprocess documents from TSV file"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                next(f)  # Skip header if present
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    
                    doc_id = parts[0]
                    text = parts[1]
                    
                    
                    self.docs_texts.append(text)
                    self.doc_ids.append(f'{dataset_name}_{doc_id}')
            
            print(f"تم تحميل  {len(self.docs_texts)} وثيقة.")
            return True
            
        except FileNotFoundError:
            print(f"الملف {input_file} غير موجود.")
            return False
        except Exception as e:
            print(f"حدث خطأ أثناء تحميل الملف: {e}")
            return False
    

    def save_to_csv(self, output_path):
        """Save processed documents to CSV file"""
        if not self.docs_texts:
            print("لا توجد وثائق محملة. يرجى تحميل الوثائق أولاً.")
            return False
        
        try:
            # استبدال علامات الاقتباس داخل النصوص (اختياري)
            self.docs_texts = [text.replace('"','') for text in self.docs_texts]
            # تأكد أن كل نص في clean_text نص وليس NaN
            self.processed_docs = [str(text) if pd.notna(text) else '' for text in self.processed_docs]
            # خزّن التوكنز كـ JSON string
            tokens_json = [json.dumps(tokens) for tokens in self.tokens]
            
            # إنشاء DataFrame
            df = pd.DataFrame({
                'doc_id': self.doc_ids,
                'original_text': self.docs_texts,
                'clean_text': self.processed_docs,
                'tokens': tokens_json
            })
            
            # استخدم quoting
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✅ تم حفظ النتائج في ملف '{output_path}' بنجاح.")
            return True
            
        except Exception as e:
            print(f"حدث خطأ أثناء حفظ الملف: {e}")
            return False
        

    def read_csv(self, input_file):
        """Read processed documents from CSV file"""
        try:
            df = pd.read_csv(input_file)
            
            # تعويض القيم الناقصة في clean_text بنص فارغ
            df['clean_text'] = df['clean_text'].fillna('')
            
            # تحويل نص JSON في tokens إلى قائمة
            df['tokens'] = df['tokens'].fillna('[]')  # إذا فيه قيم ناقصة، استبدلها بقائمة فارغة
            docs_tokens = [json.loads(text) for text in df['tokens']]
            docs_texts = df['original_text'].tolist()
            processed_docs = df['clean_text'].tolist()
            doc_ids = df['doc_id'].tolist()
            
            # Update instance variables
            self.docs_texts = docs_texts
            self.doc_ids = doc_ids
            self.processed_docs = processed_docs
            self.tokens = docs_tokens
            
            print(f"تم تحميل {len(self.docs_texts)} وثيقة من ملف CSV.")
            return True
            
        except FileNotFoundError:
            print(f"الملف {input_file} غير موجود.")
            return False
        except Exception as e:
            print(f"حدث خطأ أثناء قراءة الملف: {e}")
            return False
        
        
        
        
        
        