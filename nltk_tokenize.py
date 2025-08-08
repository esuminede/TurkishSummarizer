
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
for line in sentence: # bu değişken pdfin çekildiği kısımda oluşturuldu.
        tokenize_pdf_file = word_tokenize(line)
        for word in tokenize_pdf_file:
            if word not in stop_words:
                tokenize_words_without_stopwords.append(word)
 