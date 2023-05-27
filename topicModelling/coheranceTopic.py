from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary

def get_topic_keywords(lda_model, corpus, dictionary, texts, num_topics, top_n, coherence='c_v'):
    coherence_values = []
    topic_keywords = {}

    for topic in range(num_topics):
        topic_words = lda_model.show_topic(topic, topn=top_n)
        topic_keywords[f"Topic {topic+1}"] = [word for word, _ in topic_words]
        
        if coherence == 'c_v':
            cm = CoherenceModel(topics=[topic_keywords[f"Topic {topic+1}"]], texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(cm.get_coherence())
        elif coherence == 'u_mass':
            cm = CoherenceModel(topics=[topic_keywords[f"Topic {topic+1}"]], corpus=corpus, dictionary=dictionary, coherence='u_mass')
            coherence_values.append(cm.get_coherence())
        elif coherence == 'c_uci':
            cm = CoherenceModel(topics=[topic_keywords[f"Topic {topic+1}"]], corpus=corpus, dictionary=dictionary, coherence='c_uci')
            coherence_values.append(cm.get_coherence())
        elif coherence == 'c_npmi':
            cm = CoherenceModel(topics=[topic_keywords[f"Topic {topic+1}"]], texts=texts, dictionary=dictionary, coherence='c_npmi')
            coherence_values.append(cm.get_coherence())

    return topic_keywords, coherence_values