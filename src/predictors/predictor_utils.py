import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 

tokenizer = TfidfVectorizer().build_tokenizer() # Return a function that splits a string into a sequence of tokens considering unicode characters
stemmer = SnowballStemmer("spanish") 
lemmatizer = WordNetLemmatizer()

##stop-words: el vectorizador ya las remueve pero para mostrar las palabras más frecuente tiene sentido activarlo
remove_stops_here = True

def my_pre_processer(text, stemming=False, lemmatize=True):
    results = []
    for token in tokenizer(text):
        clean_token = token.lower().strip().strip('-').strip('_')
        if remove_stops_here and (clean_token in stopwords.words('spanish')):
          continue
        if stemming:
          token_pro = stemmer.stem(clean_token) #podemos probar stemming en vez de lematizacion
        if lemmatize:
          token_pro = lemmatizer.lemmatize(clean_token)
        if len(token_pro) > 2 and not token_pro[0].isdigit() and token_pro != 'http': #elimina palabras largo menor a 2
            results.append(token_pro)
    result = " ".join(results)
    result = re.sub('@[^\s]+', '', result)
    return result

def clean_text(text, special_chars=["\n", "\t"]):
    special_chars += ["#"]
    for char in special_chars:
        text = text.replace(char, " ")
    return my_pre_processer(text)