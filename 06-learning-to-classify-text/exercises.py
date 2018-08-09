import nltk
import random
from nltk.corpus import movie_reviews

# List of documents, labeled with the appropriate categories,
# chosen from the Movie Reviews Corpus, which categories each review as positive or negative
documents = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]
random.shuffle(documents)

# List of the 2000 most frequent words in the overall Movie Reviews Corpus
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
# Test set to classify all words
word_features = list(all_words)[:2000]


def document_features(document):
    """
    A feature extractor for documents.
    This is so that a classifier will know which aspects of the data it should pay attention to.
    :param document: input document
    :return: a feature extactor
    """
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


# Let's use our feature exractor above to train a classifier to label new movie reviews.
# To check how reliable the resulting classifier is, we will compute its accuracy on a test set in (1).
featuresets = [
    (document_features(document), category)
    for (document, category) in documents
]
# Train set and test set (1)
train_set, test_set = featuresets[100:], featuresets[:100]
# Our (Naive Bayes) classifier becomes:
movie_classifier = nltk.NaiveBayesClassifier.train(train_set)
# Compute our classifier's accuracy:
accuracy = nltk.classify.accuracy(movie_classifier, test_set)
print("\nMovie classifier accuracy:", accuracy)
# Let's retrieve which features the classifier found to be most informative:
movie_classifier.show_most_informative_features(5)
