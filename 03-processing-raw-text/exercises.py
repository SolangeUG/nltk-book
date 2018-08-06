import re
import json
import nltk
from urllib import request
from bs4 import BeautifulSoup
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


def convert_to_celsius(temp):
    """
    Convert temperature from Farenheit to Celcius degrees
    :param temp: input temperature in Farenheit
    :return: equivalent temperature in Celsius
    """
    return ((temp - 32) * 5) / 9


def print_weather_forecast():
    """
    Write code to access a favorite webpage and extract some text from it.
    For example, access a weather site and extract the forecast top temperature for your town or city today.
    :return: None
    """
    weather_url = "https://samples.openweathermap.org/data/2.5/weather" \
                  "?q=London,uk&appid=b6907d289e10d714a6e88b30761fae22"
    response = request.urlopen(weather_url)
    page = response.read().decode('utf-8')

    # extract text from the above page content
    content = BeautifulSoup(page, features="html.parser").get_text()
    # use json utilities to convert the json content to python dictionary
    content = json.loads(content)

    # extract desired information
    weather = content['weather'][0]
    print("Today's weather in London:", weather['description'])

    # extract temperature measures
    temp = float(content['main']['temp'])
    print("Current temperature (°C):", "%.2f" % convert_to_celsius(temp))
    min_temp = float(content['main']['temp_min'])
    print("Minimum temperature (°C):", "%.2f" % convert_to_celsius(min_temp))
    max_temp = float(content['main']['temp_max'])
    print("Maximum temperature (°C):", "%.2f" % convert_to_celsius(max_temp))


def unknown(url):
    """
    Return a list of unknown words that occur on a webpage.
    Extract all substrings consisting of lowercase letters (using re.findall()) and remove any items from this set
    that occur in the Words Corpus (nltk.corpus.words).
    :param url: url of the webpage to analyze
    :return: list of unkown words
    """
    page = request.urlopen(url).read().decode('UTF-8')
    strings = re.findall(r'\b[a-z][a-z]+\b', page)
    strings = set(strings)
    unknown_words = [word for word in strings if word not in nltk.corpus.words.words('en')]
    return sorted(unknown_words)


def main():
    """
    Main program entry
    :return: None
    """
    print()
    print_wh_word_types()

    print()
    print_weather_forecast()

    print()
    print(unknown("http://news.bbc.co.uk"))


if __name__ == '__main__':
    main()
