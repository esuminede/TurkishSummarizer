from nltk.corpus import stopwords
stop_words = set(stopwords.words('turkish'))

from nltk.tokenize import word_tokenize, sent_tokenize
text=["merhaba", "bir", "emine"]
tokenize_words = word_tokenize(text)

tokenize_words_without_stopwords =[]
for word in tokenize_words:
    if word not in stop_words:
        tokenize_words_without_stopwords.append(word)

print(tokenize_words_without_stopwords)



#stopword_df = pd.read_csv(STOPWORDS_FILE, header=0)
#stopwords = set(stopword_df["Stopword"].str.strip().tolist())
#seperated_words = re.findall(r'\b\w+\b', full_text) #kelime kelime ayrıldı. tfidf için kullanılabilir. noktalama işaretleri yok.

#print(sentence)

#cleaned_stopwords = [k for k in seperated_words if k not in stopword_df]
#clean_from_stopwords = " ".join(cleaned_stopwords) 
"""clean_from_stopwords = []
for i in stopwords:
    for j in seperated_words:
        if(j == i):
            j=""
        else:
            clean_from_stopwords.append(j)
print(clean_from_stopwords)
"""

-------------------------

combined_sentences = []
chunked_sentences = []
current_sentence = []
for word in tokenize_words_without_stopwords:
    #print(word) :her bir word gerçekten kelimeye karşılık geliyor.
    current_sentence.append(word)
    if word in ['.', '!', '?']: #burada harfe dönüşüyor. Bozuluyor. OLMAMALI!
        collect_sentence = ' '.join(word)
        combined_sentences.append(collect_sentence)
        current_sentence = []
print(combined_sentences[1])

"""        chunked_sentences.append(combined_sentence)
        combined_sentence = []
"""
chunks = []
chunk = [] # bütün metni tek cümle halide tutuyor. SIKINTI!!
chunk_size = 10 # tüm metni 150 karakterlik yapılara (chunk) bölmeyi hedefledim. Sonra her chunk vektör edilecek GALİBA
"""chunks = [combined_sentences[i:i+chunk_size]
          for i in range(0, len(combined_sentences), chunk_size)]
print(chunks[1])
united_sentences_as_text = [''.join(chunk) for chunk in chunks]
"""
#print(united_sentences_as_text)
for paragraph in range(len(combined_sentences)):
    if paragraph == chunk_size:
        chunk = re.split(r'(?<=[.!?])\s+', combined_sentences)
        chunks.append(chunk)
        chunk = []    

print(len(chunks))
united_sentences_as_chunks_25 = [chunks[i:i+chunk_size] for i in range(0, len(chunks), chunk_size)]
#print(united_sentences_as_chunks_25[0])
print(combined_sentences[4])
"""         
for i in range
"""
#for_chunks = 
#print(chunks)

--------------
if(i % chunk_sentence_size != 0):
        for item in combined_sentences_from_tokenize_words:
            chunk_list_of_25s = ' '.join(combined_sentences_from_tokenize_words[i]) 
    else:
        chunk_list_of_25s.append(i)
