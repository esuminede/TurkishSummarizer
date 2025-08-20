import pymupdf
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from config import FILE_NAME, STOPWORDS_FILE, EMBEDDING_MODEL
import re
 
#from snowballstemmer import TurkishStemmer
#---------------------------------------------------------------------#
doc = pymupdf.open(FILE_NAME)
full_text = ""

for page in doc:
    full_text += page.get_text().lower()

sentences = re.split(r'(?<=[.!?])\s+', full_text)
sentence = [s.strip() for s in sentences if s.strip() != ""]


import nltk
import nltk.tokenize 
nltk.download('punkt_tab')  # For tokenization
#nltk.download('stopwords')  # For stopword removal
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
stop_words = stopwords.words('turkish')

### Buradaki işlem noktalama işaretlerini ortadan kaldırıyor.
### cümleler halinde belirlenen chunklar bozuluyor ve bu yüzden cümle bazlı bütünlük ortadan kalkıyor.
### Bu yüzden tfidf için uygunluğu sağlanamadı. TfidfVectorizer kullanılacak.
tokenize_words_without_stopwords = []
for line in sentence: # sentence değişkeni pdfin çekildiği kısımda oluşturuldu.
        tokenize_pdf_file = word_tokenize(line)
        #print(tokenize_pdf_file)
        for word in tokenize_pdf_file:
            if word not in stop_words:
                tokenize_words_without_stopwords.append(word)  #Dosyadaki bütün kelimeleri teker teker alıyor. Listede cümleler değil kelimeler tutuluyor.
                                                               #Noktalama işaretleri de kelime olarak ayrıştırıldı. Nerede yaptık BİLMİYORUM şu an.

#print(tokenize_words_without_stopwords) : kelimelerin birleştirildiği kontrolü. 
### NOT: ' '.join(word) yapınca word ifadesini harflerine bölüyormuş. Bu yüzden cümle birleştirme sırasında sıkıntı çıkarttı.

combined_sentences_from_tokenize_words = []
current_sentence = []
for word in tokenize_words_without_stopwords:
    current_sentence.append(word)  #elimdeki liste kelimelerden oluşuyor.
    #print(current_sentence) #current_sentence dizisi kelimelerden oluşturuluyor
    if word in ['.', '!', '?']: #burada harfe dönüşüyor. Bozuluyor. OLMAMALI!
        new_sentece = ' '.join(current_sentence)
        combined_sentences_from_tokenize_words.append(new_sentece)
        current_sentence = []
# Son cümle için
if current_sentence:
    sentence = ' '.join(current_sentence)
    combined_sentences_from_tokenize_words.append(sentence)

#print(combined_sentences_from_tokenize_words)  # Tüm cümleler
#print(combined_sentences_from_tokenize_words)

### --------------------------------------------------------------------###
# Bütün metin stopwordlerden temizlendi. Ve cümleler halinde dizide tutuluyor.
# Şimdi sıra cümleleri chunklar için birleştirmek. (1 chunk = 25 cümle)
 
chunk_sentence_size = 1
chunk_list_of_25s = []
#for i in range(len(combined_sentences_from_tokenize_words)):
chunk_list_of_25s = [combined_sentences_from_tokenize_words[i:i+chunk_sentence_size] for i in range(0,len(combined_sentences_from_tokenize_words), chunk_sentence_size)]

#aşağıdaki parçacık chunkların doğru ayrılıp ayrılmadığını test etmek için oluşturuldu.
"""for i, chunk in enumerate(chunk_list_of_25s):
    print(f"--- Chunk {i+1} ---")
    for sentence in chunk:
        print(sentence)
    print()  # Boş satır"""

### -------------------------------------------------------------- ###
# Topic Clıusterig ve Ardışıl Bağlam işlemlerini birleştireceğim. 
# Bu sayede konu bütünlüğünü sağlayacak chunkları dil modeline iletebileceğiz.

from sentence_transformers import SentenceTransformer
import numpy as np

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
embedded_chunks = []
volitile_chunk : str
for i in range(len(chunk_list_of_25s)-1):
    volitile_vector = embedding_model.encode(chunk_list_of_25s[i])
    embedded_chunks.append(volitile_vector)
    volitile_vector = ""

#for i in range(len(embedded_chunks)):
#     print(embedded_chunks[i].shape)
embeddings_and_chunks = []
embeddings_and_chunks = zip(embedded_chunks, chunk_list_of_25s)

embedded_chunks_2d = np.mean(embedded_chunks, axis=1)

#print("Veri şekli (shape):", embedded_chunks_2d.shape)
#print("Boyut sayısı:", embedded_chunks_array.ndim)

#np.savetxt("embedded_chunks.txt", embedded_chunks, fmt = '%.8f')  daha sonra denerim. şimdi çok şart değil    

#print(embedded_chunks) # mis gibi vektörler. embedding modelini beğenmezsem bakarız yine.

### ----------------------------------------------------------------###
# Chunklar vektöre dönüştürüldü. Şimdi sıra KÜMELEMEde. KMeans

from sklearn.cluster import KMeans

num_clusters = 1
clustering_model = KMeans(n_clusters = num_clusters)
clustering_model.fit(embedded_chunks_2d)
cluster_labels = clustering_model.labels_

"""
with open("embedded_chunks.txt", "r") as f:        
    for i in range(len(embedded_chunks)):
        data = f.readline()
        data = f
        """
#print(cluster_labels)

##-----------------------------------------------------------##
# Skor Hesaplama : 2 çeşit: 
# 1. küme merkezine yakın olanların seçimi
# 2. Tfidf ile en skorlu vektörün seçimi
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

cluster_scores = []
for i, (embedding, cluster_id) in enumerate(zip(embedded_chunks_2d, cluster_labels)):
    center = clustering_model.cluster_centers_[cluster_id]
    distance = np.linalg.norm(embedding - center)
    score = 1 / (1 + distance)
    cluster_scores.append(score)

#print(cluster_scores)

top_percentage = 0.1
n_chunks = len(cluster_scores)
n_select = int(n_chunks * top_percentage)

sorted_indices = np.argsort(cluster_scores)[::-1]
top_indices= sorted_indices[:n_select]

extended_indices_for_context = []
selected_chunks = []
for idx in top_indices:
    extended_indices_for_context.append(idx)

    if idx > 0:  #bağlamı korumak için seçilen chunkın önü ve arkasındaki chunklar da alınıyor
        extended_indices_for_context.append(idx - 1)
        selected_chunks.append(chunk_list_of_25s[idx])

    if idx < n_chunks - 1:
        extended_indices_for_context.append(idx + 1)
        selected_chunks.append(chunk_list_of_25s[idx])

    selected_chunks.append(chunk_list_of_25s[idx])
print(selected_chunks)
#print(extended_indices_for_context)

##--------------------------------------------------------##
#Dil modeli ile etkileşim

import ollama
response = ollama.chat(model='mistral:7b-instruct', messages = [
    {
    'role':'user',
    'content':f"""Aşağıdaki metni özetle. KESİNLİKLE KENDİ YORUMUNU EKLEME. SADECE METİNDEKİ BİLGİLERİ KULLAN.
    ÖZETLEME TALİMATLARI:
    1. Metnin ana konusunu belirt
    2. Önemli olayları kronolojik sırayla anlat
    3. Karakter gelişimlerini ve ilişkilerini özetle
    4. Metnin atmosferini ve tonunu koru
    
    METİN: 
    {selected_chunks}""",
    }],
    options={
        'temperature': 0.3,  # Daha düşük değer daha deterministik sonuçlar verir
        'top_p': 0.9,
        'min_tokens': 1000   # Maksimum yanıt uzunluğunu artır
    }
)
with open("summary.txt", "w", encoding="utf-8") as f:
    f.write(response['message']['content'])
#print(response['message']['content'])