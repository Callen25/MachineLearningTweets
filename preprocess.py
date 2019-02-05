import numpy as np
import os
import glob
import pickle
from nltk.tokenize import TweetTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from random import shuffle


def create_training_data():
    trump_tweets = glob.glob(os.path.join(os.getcwd(), "trumpTweets", "*.txt"))
    control_tweets = glob.glob(os.path.join(os.getcwd(), "controlTweets", "*.txt"))

    trump_arr = []
    control_arr = []
    tokenizer = TweetTokenizer()
    maxLength = 0
    for tweet in trump_tweets:
        with open(tweet, 'r', encoding='utf8') as text:
            token = tokenizer.tokenize(text.read())
            if(len(token) > maxLength):
                maxLength = len(token)
            trump_arr.append( token )

    for tweet in control_tweets:
        with open(tweet, 'r', encoding='utf8') as text:
            token = tokenizer.tokenize(text.read())
            if(len(token) > maxLength):
                maxLength = len(token)
            control_arr.append( token )

    shuffle(trump_arr)
    shuffle(control_arr)
    testTrump = trump_arr[16280:]
    testControl = control_arr[16000:]
    trump_arr = trump_arr[:16280]
    control_arr = control_arr[:16000]


    print("Creating dictionary now")
    word_list = create_dictionary(data=trump_arr + control_arr)
    print(len(word_list))
    pickle_out = open("dictionary.pickle", "wb")
    pickle.dump(word_list, pickle_out)
    pickle_out.close()
    print("dictionary saved!")
    print(len(word_list))
    print("Formatting training data")
    create_training(dictionary=word_list, vec_length=maxLength,
    control_data=control_arr, trump_data=trump_arr)
    create_test(dictionary=word_list, vec_length=maxLength,
    control=testControl, trump=testTrump)


def create_dictionary(data):
    dictionary = {}
    for tweet in data:
        for text in tweet:
            if text not in dictionary:
                dictionary[text] = 0
            dictionary[text] += 1

    return [words[0] for words in sorted(dictionary.items(), key=lambda item: item[1]) if words[1] > 8]

def create_training(dictionary, vec_length, control_data, trump_data):
    training_data = []
    for tweet in control_data:
        training_data.append([create_tweet_vector(dictionary=dictionary, tweet=tweet), 0])
    for tweet in trump_data:
        training_data.append([create_tweet_vector(dictionary=dictionary, tweet=tweet), 1])
    x = []
    y = []
    for features, labels in training_data:
        x.append(features)
        y.append(labels)
    print("All sentences vectorized")
    X = pad_sequences(sequences=x, maxlen=vec_length, dtype='int32',
    padding='post', truncating='post', value=0)

    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
def create_test(dictionary, vec_length, control, trump):
    test_data = []
    for tweet in control:
        test_data.append([create_tweet_vector(dictionary=dictionary, tweet=tweet), 0])
    for tweet in trump:
        test_data.append([create_tweet_vector(dictionary=dictionary, tweet=tweet), 1])
    x = []
    y = []
    for features, labels in test_data:
        x.append(features)
        y.append(labels)
    print("All sentences vectorized")
    X = pad_sequences(sequences=x, maxlen=vec_length, dtype='int32',
    padding='post', truncating='post', value=0)

    pickle_out = open("X_test.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y_test.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

def create_tweet_vector(dictionary, tweet):
    vector = []
    for word in tweet:
        try:
            vector.append(dictionary.index(word) + 1)
        except:
            vector.append(0)
    return vector


if __name__ == '__main__':
    create_training_data()
