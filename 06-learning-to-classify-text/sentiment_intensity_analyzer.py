import os
import gzip
import json
import numpy
import urllib.request
from nltk.sentiment.util import mark_negation
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from nltk.corpus import stopwords, opinion_lexicon
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer


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


def tokenize_with_negation(text):
    """
    Split a text into lower-case tokens, removing all punctuation tokens and stopwords
    :param text: input text
    :return: lowercase word tokens, without punctuation or stopwords
    """
    # List of stop words in English
    english_stopwords = set(stopwords.words('english'))
    # Set of negated stopwords
    negated_stopwords = set(word + "_NEG" for word in english_stopwords)
    # List of all stopwords, including negated words
    all_stopwords = english_stopwords.union(negated_stopwords)

    tokens = []
    for sent in sent_tokenize(text):
        pretokens = word_tokenize(sent.lower())
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


def extract_analyzer_features(dataset):
    """
    For each review text in the dataset, extract:
        (1) the mean positive sentiment over all sentences
        (2) the mean neutral sentiment over all sentences
        (3) the mean negative sentiment over all sentences
        (4) the maximum positive sentiment over all sentences
        (5) the maximum neutral sentiment over all sentences
        (6) the maximum negative sentiment over all sentences
    :param dataset: input set of data
    :return: a matrix representing those values
    """
    feature_count = 6
    feature_matrix = numpy.empty((len(dataset), feature_count))
    for i in range(len(dataset)):
        sentences = sent_tokenize(dataset[i]['reviewText'])
        sentence_count = len(sentences)
        if sentence_count:
            polarity_count = 3
            sentence_polarities = numpy.empty((sentence_count, polarity_count))
            for j in range(sentence_count):
                polarity = analyzer.polarity_scores(sentences[j])
                sentence_polarities[j, 0] = polarity['pos']
                sentence_polarities[j, 1] = polarity['neu']
                sentence_polarities[j, 2] = polarity['neg']
            # compute the mean over the columns
            feature_matrix[i, 0:3] = numpy.mean(sentence_polarities, axis=0)
            # compute the maximum over the columns
            feature_matrix[i, 3:6] = numpy.max(sentence_polarities, axis=0)
        else:
            feature_matrix[i, 0:6] = 0.0
    return feature_matrix


def get_additional_features(dataset):
    """
    Add two features to our training model:
        (1) length of the review (in thousands of characters) - truncate at 2500
        (2) percentage of exclamation marks (%)
    :param dataset: input set of data
    :return: a matrix with those features
    """
    feature_count = 2
    feature_matrix = numpy.empty((len(dataset), feature_count))
    for i in range(len(dataset)):
        text = dataset[i]['reviewText']
        feature_matrix[i, 0] = len(text) / 1000.
        if text:
            feature_matrix[i, 1] = 100. * text.count('!') / len(text)
        else:
            feature_matrix[i, 1] = 0.0
    feature_matrix[feature_matrix > 2.5] = 2.5
    return feature_matrix


if __name__ == '__main__':
    # Dataset information
    datadir, datasetname = './data/', 'Baby'
    # Download data and load into memory
    baby_dataset = load_data(datasetname, datadir)
    baby_train, baby_valid, baby_test = partition_train_validation_test(baby_dataset)

    # TRAINING DATASETS
    X_train = dataset_to_matrix_with_negation(baby_train)
    Y_train = dataset_to_targets(baby_train)

    # Use the NLTK built-in SentimentIntensityAnalyer
    analyzer = SentimentIntensityAnalyzer()
    sample_text = baby_train[5000]['reviewText']
    for sentence in sent_tokenize(sample_text):
        print("\n", sentence)
        print(analyzer.polarity_scores(sentence), "\n")

    # Apply features to the training set
    analyzer_train = extract_analyzer_features(baby_train)
    # Additional features
    additional_train = get_additional_features(baby_train)
    # Let's see what these values look like
    print(X_train.shape, analyzer_train.shape, additional_train.shape)

    # LINEAR REGRESSION ON TRAINING DATASET
    # Stack training sets horizontally
    X_train_augmented = numpy.concatenate((X_train, analyzer_train, additional_train), axis=1)
    # Use Linear Regression
    linear_regression_augmented = LinearRegression().fit(X_train_augmented, Y_train)
    # Compute prediction
    pred_train_augmented = linear_regression_augmented.predict(X_train_augmented)
    # Compute the mean absolute error
    mae_train_augmented = mean_absolute_error(pred_train_augmented, Y_train)
    print("Linear Regression | Mean absolute error on the training data is %f stars" % mae_train_augmented)

    # RANDOM FOREST REGRESSION ON TRAINING DATASET
    random_forest_augmented = RandomForestRegressor().fit(X_train_augmented, Y_train)
    # Compute prediction
    rf_pred_train_augmented = random_forest_augmented.predict(X_train_augmented)
    # Compute the mean absolute error
    rf_mae_train_augmented = mean_absolute_error(rf_pred_train_augmented, Y_train)
    print("Random Forest Regressor | Mean absolute error on the training data is %f stars" % rf_mae_train_augmented)

    # VALIDATION DATASET
    X_valid_neg = dataset_to_matrix_with_negation(baby_valid)
    Y_valid = dataset_to_targets(baby_valid)
    analyzer_valid = extract_analyzer_features(baby_valid)
    additional_valid = get_additional_features(baby_valid)
    X_valid_augmented = numpy.concatenate((X_valid_neg, analyzer_valid, additional_valid), axis=1)

    # LINEAR REGRESSION & RANDOM FOREST REGRESSOR ON VALIDATION DATASET
    pred_valid_augmented = linear_regression_augmented.predict(X_valid_augmented)
    rf_pred_valid_augmented = random_forest_augmented.predict(X_valid_augmented)
    mae_valid_augmented = mean_absolute_error(pred_valid_augmented, Y_valid)
    print("Linear Regression | On the validation set, we get %f error" % mae_valid_augmented)
    rf_mae_valid_augmented = mean_absolute_error(rf_pred_valid_augmented, Y_valid)
    print("Random Forest Regressor | On the validation set, we get  %f error" % rf_mae_valid_augmented)
