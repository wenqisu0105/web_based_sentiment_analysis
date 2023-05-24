import joblib
from .input_processing import process_input
import os

def abs_path(file):
    current_file = os.path.abspath(__file__)
    pkl_file = os.path.join(os.path.dirname(current_file), file)
    return pkl_file


# Load models
topic_model = joblib.load(abs_path('topic_classifier.pkl'))
movie = joblib.load(abs_path('movies_sentiment.pkl'))
climate = joblib.load(abs_path('climate_sentiment.pkl'))
stock = joblib.load(abs_path('stock_sentiment.pkl'))
tweets = joblib.load(abs_path('tweets_sentiment.pkl'))
yelp = joblib.load(abs_path('yelp_sentiment.pkl'))

map_topic_to_classifier = {0: climate, 1: movie, 2: stock, 3: tweets, 4: yelp}

topic_tfidf = joblib.load(abs_path('topic_tfidf.pkl'))

tfidf_vetorizer_list = joblib.load(abs_path('tfidf.pkl'))

chi2_list = joblib.load(abs_path('feature_reduce_param.pkl'))

encoder_map = joblib.load(abs_path('encoder_map.pkl'))

def predict_sent(input_text):
    tokens = process_input(input_text)
    lemma_str = ' '.join(tokens)
    vectorized_token = topic_tfidf.transform([lemma_str])
    predicted_topic = topic_model.predict(vectorized_token)[0]# use training topic classifier to prdict the topic first
    print(f"topic predicted is {predicted_topic}")
    feature_extracted = tfidf_vetorizer_list[predicted_topic].transform([lemma_str]) # extract features beased on the topic predicted
    feature_reduced = chi2_list[predicted_topic].transform(feature_extracted) # reduce features based on the topic predicted
    predicted_sentiment = map_topic_to_classifier[predicted_topic].predict(feature_reduced)[0] # predict sentiment using the topic-specific classifier
    return encoder_map[predicted_topic][predicted_sentiment]

