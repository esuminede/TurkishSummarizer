import pymupdf
import re
import nltk
import nltk.tokenize
import pandas as pd

from config import FILE_NAME, STOPWORDS_FILE

def file_handler(file_name = FILE_NAME):
    
    doc = pymupdf.open(file_name)
    print(range(len(doc)))
    
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    return full_text

   #txt dosyası oluşturup yazdırmak çok efor harcatır gibi geldi. Bu fikir (gpt verdi) daha mantıklı.
   #ayrıca regex fonksiyonunda dosya ile düzeltme yapamadım. yukarıdaki yöntem daha yapılabilir.
    
    """out = open("output.txt", "wb")
    for page in doc:
        full_text = page.get_text().encode("utf-8")
        out.write(text)
        out.write(bytes((12)))  #byte((12,)) ifadesi form feed karakteri
    out.close()
"""
def edit_text(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    #boş olmayan cümlelerin baş ve sonunda varsa boşlukları siler.
    return [sentence.strip() for sentence in sentences if sentence.strip() != ""]

def dataframe(stopwords_file=STOPWORDS_FILE):
    
    stopwords_df = pd.read_csv(stopwords_file)
    print(stopwords_df)
    """
    stopwords = set(stopwords_df[0].str.strip().tolist())
    print(stopwords)
"""
