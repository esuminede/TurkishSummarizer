from file_handle import file_handler, edit_text, dataframe
from config import FILE_NAME,STOPWORDS_FILE
import sklearn

text = file_handler(FILE_NAME)
sentences = edit_text(text)
enumerate(sentences)

for sentence in sentences:
    print(sentence + "\n")

dataframe(STOPWORDS_FILE)