import os
import gzip
import json
import urllib.request
from nltk.corpus import stopwords
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import mark_negation
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment.util import extract_unigram_feats
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


def tokenize_with_negation(text):
    """
    Split a text into lower-case tokens, removing all punctuation tokens and stopwords
    :param text: input text
    :return: lowercase word tokens, without punctuation or stopwords
    """
    tokens = []
    for sentence in sent_tokenize(text):
        pretokens = word_tokenize(sentence.lower())
        # exclude punctuation
        pretokens = [token for token in pretokens if any(char.isalpha() for char in token)]
        # exclude negated stop words (tagged as negated)
        pretokens = mark_negation(pretokens)
        tokens.extend(token for token in pretokens if token not in all_stopwords)
    return tokens


def generate_labeled_documents(data, label):
    """
    Return a list of 'labeled' documents (tuples) for a given dataset
    :param data: input dataset
    :param label: input label to characterize the generated documents
    :return: list of labeled documents
    """
    documents = []
    for i in range(len(data)):
        # we're interested in the 'reviewText' component of our dataset's elements
        documents.extend(
            ((tokenize_with_negation(sent), label) for sent in sent_tokenize(data[i]['reviewText']))
        )
    return documents


if __name__ == '__main__':
    # Dataset information
    datadir, dataset = './data/', 'Baby'
    # Download data and load into memory
    baby_dataset = load_data(dataset, datadir)
    baby_train, baby_valid, baby_test = partition_train_validation_test(baby_dataset)

    # List of stop words in English
    english_stopwords = set(stopwords.words('english'))
    # Set of stopwords marked as negated
    negated_stopwords = set(word + "_NEG" for word in english_stopwords)
    # List of all stopwords, including negated words
    all_stopwords = english_stopwords.union(negated_stopwords)

    # Sentiment analyzer
    baby_train_docs_subj = generate_labeled_documents(baby_train[:800], 'subj')
    baby_train_docs_obj = generate_labeled_documents(baby_train[800:1000], 'obj')
    baby_train_docs = baby_train_docs_subj + baby_train_docs_obj
    # print("baby train docs [0]", baby_train_docs[0])

    baby_test_docs_subj = generate_labeled_documents(baby_test[:200], 'subj')
    baby_test_docs_obj = generate_labeled_documents(baby_test[200:400], 'obj')
    baby_test_docs = baby_test_docs_subj + baby_test_docs_obj
    # print("baby test docs [0]", baby_test_docs[0])

    analyzer = SentimentAnalyzer()
    all_words_with_negation = analyzer.all_words([doc for doc in baby_train_docs])
    print("\nVocabulary size: {}".format(str(len(all_words_with_negation))))

    # Unigram features
    unigram_features = analyzer.unigram_word_feats(all_words_with_negation, min_freq=4)
    print("Unigram features size: {}".format(str(len(unigram_features))))
    analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_features)

    # Apply features and build the training set
    training_set = analyzer.apply_features(baby_train_docs)
    test_set = analyzer.apply_features(baby_test_docs)

    # Train our classifier
    trainer = NaiveBayesClassifier.train
    classifier = analyzer.train(trainer, training_set)

    # Evaluation results
    for key, value in sorted(analyzer.evaluate(test_set).items()):
        print('{0}: {1}'.format(key, value))
