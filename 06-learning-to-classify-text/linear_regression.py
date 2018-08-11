import os
import gzip
import json
import numpy
import urllib.request
from nltk.sentiment.util import mark_negation
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from nltk.corpus import stopwords, opinion_lexicon
from nltk.tokenize import sent_tokenize, word_tokenize


def download_data(dataset_name, directory_name):
    """
    Download a given dataset name into a given directory
    :param dataset_name: input dataset name
    :param directory_name: input directory name
    :return: None
    """
    filename = 'reviews_%s_5.json' % dataset_name
    filepath = os.path.join(directory_name, filename)
    if os.path.exists(filepath):
        print("Dataset %s has already been downloaded to %s" % (dataset_name, directory_name))
    else:
        url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/%s.gz' % filename
        urllib.request.urlretrieve(url, filepath + ".gz")
        with gzip.open(filepath + ".gz", 'rb') as fin:
            with open(filepath, 'wb') as fout:
                fout.write(fin.read())
        print("Downloaded dataset %s and saved it to %s" % (dataset_name, directory_name))


def load_data(dataset_name, directory_name):
    """
    Load data in JSON format into memory
    :param dataset_name: input dataset name
    :param directory_name: input directory name
    :return:
    """
    filepath = os.path.join(directory_name, 'reviews_%s_5.json' % dataset_name)
    if not os.path.exists(filepath):
        download_data(dataset_name, directory_name)
    data = []
    with open(filepath, 'r') as f:
        for line in f:                            # read file line by line
            item_hash = hash(line)                # we will use this later for partitioning our data
            item = json.loads(line)               # convert JSON string to Python dict
            item['hash'] = item_hash              # add hash for identification purposes
            data.append(item)
    print("Loaded %d data for dataset %s" % (len(data), dataset_name))
    return data


def partition_train_validation_test(data):
    """
    Partition a dataset into a training set, a validation set and a testing set
    :param data: input dataset
    :return: training, validation and testing set
    """
    # 60% : modulus is 0, 1, 2, 3, 4, or 5
    data_train = [item for item in data if item['hash'] % 10 <= 5]
    # 20% : modulus is 6 or 7
    data_valid = [item for item in data if item['hash'] % 10 in [6, 7]]
    # 20% : modulus is 8 or 9
    data_test = [item for item in data if item['hash'] % 10 in [8, 9]]
    return data_train, data_valid, data_test


# Examples of negations
def print_negated_examples():
    negation_examples = [
        "This product wasn't bad.",
        "This is not a bad product.",
        "This product was bad.",
        "This is a bad product."
    ]
    for sentence in negation_examples:
        negated_tokens = mark_negation(word_tokenize(sentence.lower()))
        print("Sentence = {}".format(sentence))
        print(negated_tokens)


def tokenize_with_negation(text):
    """
    Split a text into lower-case tokens, removing all punctuation tokens and stopwords
    :param text: input text
    :return: lowercase word tokens, without punctuation or stopwords
    """
    # List of stop words in English
    english_stopwords = set(stopwords.words('english'))
    # Set of stopwords marked as negated
    negated_stopwords = set(word + "_NEG" for word in english_stopwords)
    # List of all stopwords, including negated words
    all_stopwords = english_stopwords.union(negated_stopwords)

    tokens = []
    for sentence in sent_tokenize(text):
        pretokens = word_tokenize(sentence.lower())
        # exclude punctuation
        pretokens = [token for token in pretokens if any(char.isalpha() for char in token)]
        # exclude negated stop words (tagged as negated)
        pretokens = mark_negation(pretokens)
        tokens.extend(token for token in pretokens if token not in all_stopwords)
    return tokens


def pos_neg_fraction_with_negation(text):
    """
    Compute the fraction of positive and negative words in a text, including negated words
    :param text: input text
    :return: a fraction of positive and negative words in the text
    """
    # Sets of already known positive and negative words
    positive_words = set(opinion_lexicon.positive())
    negative_words = set(opinion_lexicon.negative())
    # Set of all positive words including negated negative words
    all_positive_words = positive_words.union({tag + "_NEG" for tag in negative_words})
    # Set of all positive words including negated positive words
    all_negative_words = negative_words.union({tag + "_NEG" for tag in positive_words})

    tokens = tokenize_with_negation(text)
    # count how many positive and negative words occur in the text
    count_pos, count_neg = 0, 0
    for token in tokens:
        if token in all_positive_words:
            count_pos += 1
        if token in all_negative_words:
            count_neg += 1
    count_all = len(tokens)
    if count_all != 0:
        return count_pos/count_all, count_neg/count_all
    else:  # avoid division by zero
        return 0., 0.


def dataset_to_targets(data):
    """
    Convert training dataset to target vector
    :param data:
    :return:
    """
    return numpy.array([item['overall'] for item in data])


def dataset_to_matrix_with_negation(data):
    """
    Convert training dataset (with negation) into a matrix
    :param data: training dataset
    :return: matrix representation of training dataset
    """
    # assuming the training dataset has a 'reviewText' key
    return numpy.array([list(pos_neg_fraction_with_negation(item['reviewText'])) for item in data])


if __name__ == '__main__':
    # Dataset information
    datadir, dataset = './data/', 'Baby'
    # Download data and load into memory
    baby_dataset = load_data(dataset, datadir)
    baby_train, baby_valid, baby_test = partition_train_validation_test(baby_dataset)

    # Compute mean absolute error on training set
    Y_train = dataset_to_targets(baby_train)
    X_train = dataset_to_matrix_with_negation(baby_train)
    linear_regression_model = LinearRegression().fit(X_train, Y_train)
    prediction_train = linear_regression_model.predict(X_train)
    mae_train = mean_absolute_error(prediction_train, Y_train)
    print("Now the mean absolute error on the training data is %f stars" % mae_train)
