import re
from nltk import word_tokenize
from nltk.corpus import gutenberg


def print_wh_word_types():
    """
    Read in some text from a corpus, tokenize it, and print the list of all wh-word types that occur.
    (wh-words in English are used in questions, relative clauses and exclamations: who, which, what, and so on).
    Print them in order.
    Are any words duplicated in this list, because of the presence of case distinctions or punctuation?
    :return: None
    """
    bryant = gutenberg.raw('bryant-stories.txt')
    tokens = word_tokenize(bryant)
    wh_words = []
    for token in tokens:
        wh_word = re.findall(r'^[w|W]h[aA-zZ]*', token)
        if wh_word:
            wh_words.append(wh_word[0])
    print(set(wh_words))


def main():
    """
    Main program entry
    :return: None
    """
    print_wh_word_types()


if __name__ == '__main__':
    main()
