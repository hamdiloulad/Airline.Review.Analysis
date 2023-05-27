import gensim
import re
def get_most_frequent_topic(lda_model, reviews):
    num_reviews = len(reviews)
    topics = lda_model.show_topics(num_topics=-1, formatted=False)
    topic_frequencies = {i:0 for i in range(len(topics))}
    for review in reviews:
        doc = lda_model.id2word.doc2bow(review.split())
        topic_distribution = lda_model[doc]
        most_probable_topic = max(topic_distribution, key=lambda x: x[1])[0]
        topic_frequencies[most_probable_topic] += 1
    for topic_id in topic_frequencies:
        topic_frequencies[topic_id] /= num_reviews
    return topic_frequencies