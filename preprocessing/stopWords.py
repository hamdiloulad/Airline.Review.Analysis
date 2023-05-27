from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu','verified','british','airway','wa','ha', 'ba','us', 'took', 'would', 'sure', 'get', 'got','even', 'lhr','flight'])
    stop_words = set(stopwords.words('english'))
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]
