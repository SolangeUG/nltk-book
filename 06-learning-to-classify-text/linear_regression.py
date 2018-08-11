# noinspection PyUnresolvedReferences
from util import data_util
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    # Dataset information
    datadir, dataset = './data/', 'Baby'
    # Download data and load into memory
    baby_dataset = data_util.load_data(dataset, datadir)
    baby_train, baby_valid, baby_test = data_util.partition_train_validation_test(baby_dataset)

    # Compute mean absolute error on training set
    Y_train = data_util.dataset_to_targets(baby_train)
    X_train = data_util.dataset_to_matrix_with_negation(baby_train)
    linear_regression_model = LinearRegression().fit(X_train, Y_train)
    prediction_train = linear_regression_model.predict(X_train)
    mae_train = mean_absolute_error(prediction_train, Y_train)
    print("Linear Regression | Mean absolute error on the training data is %f stars" % mae_train)
