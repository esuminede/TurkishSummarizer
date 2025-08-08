import pymupdf
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from config import FILE_NAME, STOPWORDS_FILE
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
 
chunk_sentence_size = 25
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
embedding_model = SentenceTransformer('sentence-transformers/LaBSE')
embedded_chunks = []
volitile_chunk : str
for i in range(len(chunk_list_of_25s)):
    volitile_vector = embedding_model.encode(chunk_list_of_25s[i])
    embedded_chunks.append(volitile_vector)
    volitile_vector = ""
#print(embedded_chunks) # mis gibi vektörler. embedding modelini beğenmezsem bakarız yine.

### ----------------------------------------------------------------###
# Chunklar vektöre dönüştürüldü. Şimdi sıra KÜMELEMEde. KMeans

from sklearn import KMeans

num_clusters = 10
clustering_model = KMeans(n_clusters = num_clusters)
clustering_model.fit(embedded_chunks)
cluster_labels = clustering_model.labels_

print(cluster_labels)