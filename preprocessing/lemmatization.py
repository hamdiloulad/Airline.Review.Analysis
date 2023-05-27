from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def lemmatize_text(sentences):
    lemmatizer = WordNetLemmatizer()
    lemmatized_texts = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        lemmatized_text = ' '.join(lemmatized_words)
        lemmatized_texts.append(lemmatized_text)
    return lemmatized_texts