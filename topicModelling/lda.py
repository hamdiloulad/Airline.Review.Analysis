import gensim
from gensim import corpora

def train_lda_model(data_words, num_topics):
    id2word = corpora.Dictionary(data_words)
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
    doc_lda = lda_model[corpus]

    return lda_model, doc_lda
