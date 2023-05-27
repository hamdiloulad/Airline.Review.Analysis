import numpy as np
import matplotlib.pyplot as plt
import gensim
from gensim.models import CoherenceModel
import gensim.corpora as corpora

def optimal_topic_nbre(data_words, num_topics):
    # Create dictionary
    id2word = corpora.Dictionary(data_words)

    # Create corpus
    corpus = [id2word.doc2bow(text) for text in data_words]

    # Initialize variables
    coherence_values = []
    perplexity_values = []
    model_list = []

    for num_topics in range(2, num_topics+1):
        print(num_topics)
        # Generate LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        # Calculate coherence and perplexity values
        coherence_model = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
        perplexity = lda_model.log_perplexity(corpus)
        perplexity = np.exp2(-perplexity)
        perplexity_values.append(perplexity)

        # Add model to list
        model_list.append(lda_model)

    # Plot coherence and perplexity values
    fig, ax1 = plt.subplots()
    x = range(2, num_topics+1)
    color = 'tab:red'
    ax1.set_xlabel('Num Topics')
    ax1.set_ylabel('Coherence Values', color=color)
    ax1.plot(x, coherence_values, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Perplexity Values', color=color)
    ax2.plot(x, perplexity_values, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)  # set x-tick labels to integer values

    plt.show()

    return model_list, coherence_values, perplexity_values
