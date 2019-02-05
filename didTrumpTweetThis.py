import pickle
import preprocess
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from nltk.tokenize import TweetTokenizer

dictionary = pickle.load(open("dictionary.pickle", "rb"))
tokenizer = TweetTokenizer()

def prepare(tweet):
    tokenized_tweet = tokenizer.tokenize(tweet)
    tweet_vec = preprocess.create_tweet_vector(dictionary=dictionary, tweet=tokenized_tweet)
    one_sample = []
    one_sample.append(tweet_vec)
    padded_vec = pad_sequences(sequences=one_sample, maxlen=86, dtype='int32', padding='post', truncating='post', value=0)
    print(padded_vec)
    model = tf.keras.models.load_model("Tweeets-CNN.model")

    prediction = model.predict([padded_vec])

    print(prediction[0])

    if prediction[0][0] > prediction[0][1]:
        print("Trump did not tweet this")
    else:
        print("Trump did tweet this")

if __name__ == '__main__':
    prepare("Just signed one of the most important, and largest, Trade Deals in U.S. and World History. The United States, Mexico and Canada worked so well together in crafting this great document. The terrible NAFTA will soon be gone. The USMCA will be fantastic for all!")
