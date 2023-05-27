from nltk.sentiment.vader import SentimentIntensityAnalyzer

def count_reviews_sentiments(reviews):
    sid = SentimentIntensityAnalyzer()
    negative_reviews = 0
    neutral_reviews = 0
    positive_reviews = 0
    
    for review in reviews:
        sentiment_scores = sid.polarity_scores(review)
        if sentiment_scores['neg'] > sentiment_scores['pos']:
            negative_reviews += 1
        elif sentiment_scores['pos'] > sentiment_scores['neg']:
            positive_reviews += 1
        else:
            neutral_reviews += 1
    
    return negative_reviews, neutral_reviews, positive_reviews
