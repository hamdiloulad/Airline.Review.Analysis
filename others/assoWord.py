import nltk
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import stopwords

def get_associated_words_frequency(reviews, negative_words, num_words=50):
    # preprocess the reviews
    reviews = [review.lower() for review in reviews]
    reviews = [nltk.word_tokenize(review) for review in reviews]
    reviews = [[word for word in review if word.isalnum()] for review in reviews]
    stop_words = set(stopwords.words('english'))
    reviews = [[word for word in review if word not in stop_words] for review in reviews]

    # create bigrams
    finder = BigramCollocationFinder.from_documents(reviews)
    finder.apply_word_filter(lambda word: word in negative_words)
    
    # measure association between words using Pointwise Mutual Information
    bigram_measures = BigramAssocMeasures()
    collocations = finder.nbest(bigram_measures.pmi, num_words)

    # get the associated words from the bigrams
    associated_words = [word for collocation in collocations for word in collocation if word not in negative_words]
    
    # count the frequency of associated words
    freq_dist = nltk.FreqDist(associated_words)
    
    return freq_dist.most_common()
