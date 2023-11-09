from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    sc = SparkContext(appName="PythonStreamingSentimentAnalysis")
    ssc = StreamingContext(sc, 1)

    # Create initial plot
    personality_types = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ', 'ISTP', 'ISFP', 'INFP', 'INTP',
                         'ESTP', 'ESFP', 'ENFP', 'ENTP', 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']
    avg_polarity = [0] * len(personality_types)

    fig, ax = plt.subplots()
    plt.ion()
    rects = ax.bar(personality_types, avg_polarity)
    ax.set_ylim(-1, 1)
    ax.set_title('Average Polarity by Personality Type')

    def update_plot(rdd):
        # Update plot data
        scores = rdd.flatMap(lambda line: line.split(" ")) \
            .map(lambda word: (word[-4:], TextBlob(word).sentiment.polarity)) \
            .reduceByKey(lambda a, b: a + b)

        personality_counts = scores.map(lambda x: (x[0], x[1])) \
            .reduceByKey(lambda a, b: a + b)

        avg_polarity = []
        for p in personality_types:
            polarity_sum = personality_counts.filter(lambda x: x[0] == p).map(lambda x: x[1]).reduce(lambda a, b: a + b)
            count = personality_counts.filter(lambda x: x[0] == p).count()
            avg_polarity.append(polarity_sum / count if count > 0 else 0)

        # Update plot with new data
        for i, rect in enumerate(rects):
            rect.set_height(avg_polarity[i])

        fig.canvas.draw()

    lines = ssc.textFileStream("localhost", 8080)
    scores = lines.window(20, 10) \
        .transform(lambda rdd: rdd.filter(lambda line: line != '')) \
        .transform(lambda rdd: rdd.map(lambda line: line.lower().strip())) \
        .transform(lambda rdd: rdd.flatMap(lambda line: line.split(" "))) \
        .map(lambda word: (word, TextBlob(word).sentiment.polarity)) \
        .reduceByKey(lambda a, b: a + b)

    personality_counts = scores.map(lambda x: (x[0][-4:], x[1])) \
        .reduceByKey(lambda a, b: a + b)

    personality_counts.foreachRDD(update_plot)

    ssc.start()
    plt.show()
    ssc.awaitTermination()