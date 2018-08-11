# noinspection PyUnresolvedReferences
from util import data_util
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import extract_unigram_feats
from nltk.tokenize import sent_tokenize


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
            ((data_util.tokenize_with_negation(sent), label) for sent in sent_tokenize(data[i]['reviewText']))
        )
    return documents


if __name__ == '__main__':
    # Dataset information
    datadir, dataset = './data/', 'Baby'
    # Download data and load into memory
    baby_dataset = data_util.load_data(dataset, datadir)
    baby_train, baby_valid, baby_test = data_util.partition_train_validation_test(baby_dataset)

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
