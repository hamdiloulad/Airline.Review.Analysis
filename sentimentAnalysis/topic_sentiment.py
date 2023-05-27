from nltk.sentiment import SentimentIntensityAnalyzer

def get_topic_sentiment(lda_model, topic_labels):
    sia = SentimentIntensityAnalyzer()
    topic_sentiments = []
    for i, topic in enumerate(lda_model.show_topics(num_topics=len(topic_labels), formatted=False)):
        topic_label = topic_labels[i]
        topic_words = [word[0] for word in topic[1]]
        topic_text = ' '.join(topic_words)
        sentiment_scores = sia.polarity_scores(topic_text)
        topic_sentiments.append((topic_label, sentiment_scores))
    return topic_sentiments
