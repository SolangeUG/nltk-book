import os
import traceback
import urllib.request
from bs4 import BeautifulSoup


def download_pictures(celebrity, directory_name):
    """
    Download all pictures of the requested celebrity posted on the Zimbio website into a given directory
    :param celebrity: requested celebrity name
    :param directory_name: input directory name
    :return: None
    """

    # We'd like to download a given celebirty pictures posted on the wwww.zimbio.com website
    # To do so, from the http://www.zimbio.com/photos/{celebrity}/browse, we'll first determine the gallery size
    # Then, we'll parse all gallery pages, starting
    # from http://www.zimbio.com/photos/{celebrity}/browse?Page=1
    # to   http://www.zimbio.com/photos/{celebrity}/browse?Page=size


def main():
    folder = './data/'
    celebrity = 'Jennifer Garner'
    download_pictures(celebrity, folder)


if __name__ == '__main__':
    main()
