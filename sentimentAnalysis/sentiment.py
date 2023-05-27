from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_sentiment_scores(reviews):
    num_reviews = len(reviews)
    analyzer = SentimentIntensityAnalyzer()
    neg_score = 0
    neu_score = 0
    pos_score = 0
    
    for review in reviews:
        vs = analyzer.polarity_scores(review)
        neg_score += vs['neg']
        neu_score += vs['neu']
        pos_score += vs['pos']
            
    neg_avg = neg_score / num_reviews
    neu_avg = neu_score / num_reviews
    pos_avg = pos_score / num_reviews
    
    return {'Negative': neg_avg, 'Neutral': neu_avg, 'Positive': pos_avg}
