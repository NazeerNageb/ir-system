# 🔍 نظام استرجاع المعلومات (Information Retrieval System)

نظام متكامل لاسترجاع المعلومات يدعم طرق بحث متعددة مع واجهة ويب تفاعلية.

## 📋 نظرة عامة

هذا المشروع يقدم نظام استرجاع معلومات متقدم يدعم أربع طرق بحث مختلفة:
- **TF-IDF**: البحث التقليدي باستخدام Term Frequency-Inverse Document Frequency
- **BERT Embedding**: البحث الدلالي باستخدام نماذج BERT
- **Chroma**: البحث باستخدام قاعدة بيانات المتجهات Chroma
- **Hybrid**: الجمع بين TF-IDF و BERT للحصول على نتائج محسنة

## 🏗️ بنية المشروع

```
ir_true/
├── 📁 embedding/                 # خدمات التضمين (Embedding)
│   ├── embedding_offline.py     # معالجة التضمين خارج الخط
│   ├── embedding_server.py      # خادم التضمين
│   └── embedding_service.py     # خدمة التضمين
│
├── 📁 preprocess/               # معالجة البيانات الأولية
│   ├── preprocess.py           # معالجة البيانات
│   └── preprocess_service.py   # خدمة المعالجة
│
├── 📁 tfidf/                   # خدمات TF-IDF
│   ├── tf_idf_offline.py       # معالجة TF-IDF  
│   ├── tf_idf_service.py       # خدمة TF-IDF
│   └── tfidf_server.py         # خادم TF-IDF
│
├── 📁 Ranking/                  # خدمات الترتيب
│   ├── ranking_offline.py      # معالجة الترتيب  
│   └── ranking_service.py      # خدمة الترتيب
│
├── 📁 search/                   # خدمات البحث
│   ├── search_offline.py       # معالجة البحث  
│   └── search_service.py       # خدمة البحث
│
├── 📁 files/                    # الملفات المحفوظة
│   ├── 📁 antique/             # ملفات مجموعة Antique
│   │   ├── antique_inverted_index.joblib
│   │   ├── bert_embeddings.npy
│   │   ├── tfidf_matrix.joblib
│   │   ├── user_queries_log.json
│   │   └── vectorizer.joblib
│   └── 📁 corpus/              # ملفات مجموعة Corpus
│       ├── bert_embeddings.npy
│       ├── corpus_inverted_index.joblib
│       ├── tfidf_matrix.joblib
│       ├── user_queries_log.json
│       └── vectorizer.joblib
│
├── 📁 antique_embeddings_chroma_db/  # قاعدة بيانات Chroma لمجموعة Antique
├── 📁 corpus_embeddings_chroma_db/   # قاعدة بيانات Chroma لمجموعة Corpus
│
├── 📄 main.py                   # النقطة الرئيسية للتطبيق
├── 📄 serve_interface.py        # خادم واجهة الويب
├── 📄 search_interface.html     # واجهة البحث التفاعلية
├── 📄 data_function.py          # وظائف معالجة البيانات
├── 📄 convert_doc.py           # تحويل المستندات
├── 📄 test_fix.py              # اختبارات وإصلاحات
│
├── 📊 ملفات البيانات:
│   ├── antique_clean_data.csv
│   ├── antique_docs_utf8.tsv
│   ├── antique_docs.tsv
│   ├── antique_qrels.tsv
│   ├── antique_queries.txt
│   ├── corpus_clean_data.csv
│   ├── corpus_docs_utf8.tsv
│   ├── corpus_docs.tsv
│   ├── corpus_qrels.tsv
│   ├── corpus_queries.jsonl
│   ├── corpus.json
│   └── corpus.jsonl
│
├── 📈 ملفات التقييم:
│   ├── evaluation_bert.ipynb
│   ├── evaluation_hybrid.ipynb
│   └── evaluation_tfidf.ipynb
│
└── 📄 ملفات النتائج:
    ├── search_results_chroma_antique_*.json
    ├── search_results_chroma_corpus_*.json
    ├── search_results_embedding_antique_*.json
    ├── search_results_embedding_corpus_*.json
    ├── search_results_hybrid_antique_*.json
    ├── search_results_hybrid_corpus_*.json
    ├── search_results_tfidf_antique_*.json
    └── search_results_tfidf_corpus_*.json
```

## 🔧 الخدمات المتاحة

### 1. خدمة البحث الرئيسية (`serach.py`)
- **المنفذ**: 8004
- **الوظائف**:
  - `/search_query` - البحث باستخدام TF-IDF, BERT, Chroma
  - `/search_query_hybrid` - البحث الهجين
  - `/get_document` - جلب محتوى المستند


### 2. خدمة TF-IDF (`tfidf/tfidf_server.py`)
- **المنفذ**: 8001
- **الوظائف**:
  - بناء فهرس TF-IDF
  - البحث باستخدام TF-IDF
  - حساب أوزان المصطلحات

### 3. خدمة التضمين (`embedding/embedding_server.py`)
- **المنفذ**: 8002
- **الوظائف**:
  - توليد تضمينات BERT
  - البحث الدلالي
  - حساب التشابه بين المستندات

### 4. خدمة الترتيب (`Ranking/ranking_service.py`)
- **المنفذ**: 8003
- **الوظائف**:
  - دمج نتائج البحث
  - ترتيب النتائج
  - تطبيق خوارزميات الترتيب

## 📊 مجموعات البيانات المدعومة

### 1. Antique Dataset
- **الوصف**: مجموعة بيانات للأسئلة والأجوبة باللغة الإنجليزية
- **الملفات**:
  - `antique_docs.tsv` - المستندات
  - `antique_queries.txt` - الاستعلامات
  - `antique_qrels.tsv` - العلاقات المرجعية

### 2. Corpus Dataset
- **الوصف**: مجموعة بيانات نصية متنوعة
- **الملفات**:
  - `corpus_docs.tsv` - المستندات
  - `corpus_queries.jsonl` - الاستعلامات
  - `corpus_qrels.tsv` - العلاقات المرجعية

## 🔍 طرق البحث المدعومة

### 1. TF-IDF
- **الوصف**: البحث التقليدي باستخدام أوزان المصطلحات
- **المميزات**:
  - سريع وفعال
  - مناسب للبحث النصي المباشر
  - يدعم البحث باللغة العربية والإنجليزية

### 2. BERT Embedding
- **الوصف**: البحث الدلالي باستخدام نماذج BERT
- **المميزات**:
  - فهم دلالي للاستعلامات
  - دعم اللغات المتعددة
  - نتائج أكثر دقة للاستعلامات المعقدة

### 3. Chroma
- **الوصف**: البحث باستخدام قاعدة بيانات المتجهات
- **المميزات**:
  - تخزين فعال للمتجهات
  - بحث سريع في المساحات المتجهية
  - دعم الاستعلامات المعقدة

### 4. Hybrid
- **الوصف**: الجمع بين TF-IDF و BERT
- **المميزات**:
  - دمج مميزات الطريقتين
  - نتائج أكثر شمولية
  - تحسين دقة البحث

## 🎯 واجهة المستخدم

### المميزات:
- **واجهة عربية**: تصميم مخصص للغة العربية مع دعم RTL
- **تصميم متجاوب**: يعمل على جميع الأجهزة
- **بحث تفاعلي**: نتائج فورية مع إمكانية عرض المستندات
- **طرق بحث متعددة**: اختيار طريقة البحث المناسبة
- **عرض النتائج**: ترتيب النتائج مع النقاط والروابط

### الاستخدام:
1. اختر مجموعة البيانات (Antique أو Corpus)
2. اختر طريقة البحث (TF-IDF, BERT, Chroma, أو Hybrid)
3. أدخل استعلام البحث
4. حدد عدد النتائج المطلوبة
5. اضغط "ابدأ البحث"
6. استعرض النتائج واضغط على "عرض محتوى المستند" لعرض المحتوى الكامل

## 📈 التقييم والأداء

### ملفات التقييم:
- `evaluation_tfidf.ipynb` - تقييم أداء TF-IDF
- `evaluation_bert.ipynb` - تقييم أداء BERT
- `evaluation_hybrid.ipynb` - تقييم الأداء الهجين

### مقاييس التقييم:
- **Precision@K**: دقة النتائج في المواضع الأولى
- **Recall@K**: استرجاع النتائج الصحيحة
- **MRR**: متوسط الترتيب المتبادل (Mean Reciprocal Rank)
- **MAP**: متوسط الدقة

## 🔧 التخصيص والتطوير

### إضافة مجموعة بيانات جديدة:
1. أضف ملفات البيانات إلى المجلد الرئيسي
2. عدّل `data_function.py` لمعالجة البيانات الجديدة
3. أضف خيارات جديدة في واجهة المستخدم
4. اختبر النظام مع البيانات الجديدة


### السجلات:
- راجع سجلات الخدمات للتفاصيل
- استخدم `user_queries_log.json` لتتبع الاستعلامات



