import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def create_wordclouds(lda_model, num_topics, topic_labels):
    for topic in range(num_topics):
        plt.figure(figsize=(10, 6))
        wc = WordCloud(background_color="white", width=800, height=400, max_words=200, colormap="tab10", prefer_horizontal=0.8)
        wc.generate_from_frequencies(dict(lda_model.show_topic(topic, 200)))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(topic_labels[topic])
        plt.tight_layout(pad=0)
        file_path = os.path.join('D:/DataProject/venv/scarping-nlp/result', f"wordcloud_topic_{topic}.png")
        plt.savefig(file_path, dpi=300)

        # Close the figure to free up memory
        plt.close()
