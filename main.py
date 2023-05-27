import pandas as pd
import os
import re
import nltk
from others.wordCloud import create_wordclouds
from topicModelling.mostFrequent import get_most_frequent_topic
from sentimentAnalysis.sentiment import get_sentiment_scores
from sentimentAnalysis.sentimentCount import count_reviews_sentiments
from preprocessing.sentToWords import sent_to_words
from preprocessing.lemmatization import lemmatize_text
from preprocessing.stopWords import remove_stopwords
from topicModelling.lda import train_lda_model
from sentimentAnalysis.topic_sentiment import get_topic_sentiment
from topicModelling.nbrTopic import optimal_topic_nbre
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


#import data
papers = pd.read_csv('D:/DataProject/venv/scarping-nlp/Airline.Review.Analysis/import/main.csv')

#Filtre data
papers = papers.drop(columns=['Aircraft','Type Of Traveller','Seat Type','Route','Date Flown','Seat Comfort','Wifi & Connectivity','Food & Beverages','Inflight Entertainment','Ground Service','Cabin Staff Service','Value For Money','Value For Money','Recommended'], axis=1)

#Data Preprocessing
def main():
        #Removing punctuation
    papers['paper_text_processed'] = \
    papers['review_text'] = papers['review_text'].astype(str).map(lambda x: re.sub('[,\.!?]', '', x))
    data = papers.paper_text_processed.values.tolist()
    data_words = list(sent_to_words(data))
    data_words = [lemmatize_text(word) for word in data_words]
    data_words = remove_stopwords(data_words)
    #model_list, coherence_values, perplexity_values = optimal_topic_nbre(data_words, num_topics=10)
    lda_model,doc_lda= train_lda_model(data_words, num_topics=6) 

    vis_data = gensimvis.prepare(lda_model, doc_lda, dictionary=lda_model.id2word)
    file_path = os.path.join('D:/DataProject/venv/scarping-nlp/result', 'lda_visualization.html')
    pyLDAvis.save_html(vis_data, file_path)
    topic_labels = {
        0: 'In-flight experience',
        1: 'Cabin crew',
        2: 'Class',
        3: 'Airport experience',
        4: 'Business class ',
        5: 'Economy class ',
    }
    create_wordclouds(lda_model, 6, topic_labels)
    
    reviews = papers['review_text'].tolist()
    most_frequent_topic = get_most_frequent_topic(lda_model, reviews)
    sentiment_scores = get_sentiment_scores(reviews)
    topic_sentiments = get_topic_sentiment(lda_model, topic_labels)
    sentiment_reviews = count_reviews_sentiments(reviews)
    
    print(most_frequent_topic)
    print(sentiment_scores)
    print(topic_sentiments)
    print(sentiment_reviews)
    
if __name__ == "__main__":
    main()
