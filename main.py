from topic import train_model_from_csv
from classifier import extend_csv_with_sentiment
# save_csv('data/geotagged_tweets.jsons', 'data/tweets.csv', 'data/error_tweets.txt')
# lda_model = train_model_from_csv('data/tweets.csv')
#
# print(lda_model.print_topics(num_topics=20, num_words=6))

extend_csv_with_sentiment('data/tweets.csv', 'data/senti_tweets.csv')
