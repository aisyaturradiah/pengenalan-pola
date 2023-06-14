import pandas as pd
import numpy as np
from string import punctuation
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
import os
import pickle

nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords

def remove_text_special (text:str)->str:
    text = text.replace('\\t',"").replace('\\n',"").replace('\\u',"").replace('\\',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    return text.replace("http://"," ").replace("https://", " ")

def remove_tanda_baca(text:str)->str:
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text)
    return text
    

def remove_numbers (text:str)->str:
    return re.sub(r"\d+", "", text)

"""
    CASE FOLDING 
"""

#proses casefolding
def casefolding(Comment:str)->str:
    Comment = Comment.lower()
    return Comment

"""
    TOKENIZING
"""

def tokenizing(sentence:str)->list:
    return nltk.word_tokenize(sentence)



"""
    NORMALIZE
"""

# print(os.getcwd())
normalize = pd.read_csv("https://raw.githubusercontent.com/RBellaApriliaDamayanti22/projectt/main/Normalization%20Data.csv")
normalize_word_dict = {}

for row in normalize.iterrows():
    if row[0] not in normalize_word_dict:
        normalize_word_dict[row[0]] = row[1]

def normalized_term(comment:list)->list:
  return [normalize_word_dict[term] if term in normalize_word_dict else term for term in comment]


"""
    STOPWARD REMOVAL
"""

txt_stopwords = stopwords.words('indonesian')
def stopwords_removal(filtering:list)->list:
    filtering = [word for word in filtering if word not in txt_stopwords]
    return filtering

#stopword removal 2

data_stopwords = pd.read_csv("https://raw.githubusercontent.com/RBellaApriliaDamayanti22/projectt/main/list_stopwords.csv")
data_stopwords = data_stopwords.to_numpy().flatten().tolist()

def stopwords_removal2(filter:list)->list :
    filter = [word for word in filter if word not in data_stopwords]
    return filter

"""
    STEMMING
"""

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming (document:list)->list:
    result = []
    for term in document:
        result.append(stemmer.stem(term))
    return result

def preprocessing(text:str)->list:
    text = remove_text_special(text)
    text = remove_tanda_baca(text)
    text = remove_numbers(text)
    text = casefolding(text)
    tokenize = tokenizing(text)
    tokenize = normalized_term(tokenize)
    tokenize = stopwords_removal(tokenize)
    tokenize = stopwords_removal2(tokenize)
    tokenize = stemming(tokenize)

    return tokenize

with open("word2vec.pickle", 'rb') as file:
    vectorizer = pickle.load(file)

with open("model.pickle", 'rb') as file:
    model = pickle.load(file)  
    