import numpy
# noinspection PyUnresolvedReferences
from util import data_util
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def analyzer_features(dataset):
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


def additional_features(dataset):
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
    baby_dataset = data_util.load_data(datasetname, datadir)
    baby_train, baby_valid, baby_test = data_util.partition_train_validation_test(baby_dataset)

    # TRAINING DATASETS
    X_train = data_util.dataset_to_matrix_with_negation(baby_train)
    Y_train = data_util.dataset_to_targets(baby_train)

    # Use the NLTK built-in SentimentIntensityAnalyer
    analyzer = SentimentIntensityAnalyzer()
    sample_text = baby_train[5000]['reviewText']
    for sentence in sent_tokenize(sample_text):
        print("\n", sentence)
        print(analyzer.polarity_scores(sentence), "\n")

    # Apply features to the training set
    analyzer_train = analyzer_features(baby_train)
    # Additional features
    additional_train = additional_features(baby_train)
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
    X_valid_neg = data_util.dataset_to_matrix_with_negation(baby_valid)
    Y_valid = data_util.dataset_to_targets(baby_valid)
    analyzer_valid = analyzer_features(baby_valid)
    additional_valid = additional_features(baby_valid)
    X_valid_augmented = numpy.concatenate((X_valid_neg, analyzer_valid, additional_valid), axis=1)

    # LINEAR REGRESSION & RANDOM FOREST REGRESSOR ON VALIDATION DATASET
    pred_valid_augmented = linear_regression_augmented.predict(X_valid_augmented)
    rf_pred_valid_augmented = random_forest_augmented.predict(X_valid_augmented)
    mae_valid_augmented = mean_absolute_error(pred_valid_augmented, Y_valid)
    print("Linear Regression | On the validation set, we get %f error" % mae_valid_augmented)
    rf_mae_valid_augmented = mean_absolute_error(rf_pred_valid_augmented, Y_valid)
    print("Random Forest Regressor | On the validation set, we get  %f error" % rf_mae_valid_augmented)
