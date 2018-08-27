import os
import traceback
import urllib.request
from bs4 import BeautifulSoup


def download_pictures(celebrity, directory_name, gallery_size):
    """
    Download all pictures of the requested celebrity posted on the Zimbio website into a given directory
    :param celebrity: requested celebrity name
    :param directory_name: input directory name
    :param gallery_size: number of pages in the gallery
    :return: None
    """

    # We'd like to download a given celebirty pictures posted on the wwww.zimbio.com website
    # To do so, from the http://www.zimbio.com/photos/{celebrity}/browse, we'll first determine the gallery size
    # Then, we'll parse all gallery pages, starting
    # from http://www.zimbio.com/photos/{celebrity}/browse?Page=1
    # to   http://www.zimbio.com/photos/{celebrity}/browse?Page=gallery_size

    if gallery_size > -1:
        for i in range(1, gallery_size):
            pass


def get_gallery_size(celebrity):
    """
    Determine the gallery pages size of all pictures of the requested celebrity posted on the Zimbio website
    :param celebrity: requested celebrity name
    :return: gallery pages size
    """
    # We'd like to download a given celebirty pictures posted on the wwww.zimbio.com website
    # To do so, from the http://www.zimbio.com/photos/{celebrity}/browse, we'll first determine the gallery size
    # Then, we'll parse all gallery pages, starting
    # from http://www.zimbio.com/photos/{celebrity}/browse?Page=1
    # to   http://www.zimbio.com/photos/{celebrity}/browse?Page=size

    # Assuming celebrity is of the form 'Firstname Lastname', e.g: 'Jennifer Garner',
    # we need to replace the space between the first and last name with a + sign
    gallery_address = 'http://www.zimbio.com/photos/%s/browse' % celebrity.replace(' ', '+')

    # Let's connect to and parse the gallary front page to determine the gallery size
    try:
        with urllib.request.urlopen(gallery_address) as response:
            html = response.read()

        page = BeautifulSoup(html, 'html.parser')
        gallery = page.select('.pagination')
        links = gallery.select('a')
        last_gallery_link = links[len(links) - 1]

        # The last gallery address is of the form /photos/{celebrity}/browse?Page=size
        last_gallery_address = last_gallery_link.select('href')
        gallery_size = int(last_gallery_address.split('=')[1])

        return gallery_size

    except Exception as e:
        print('Error Type: %s' % str(e))
        print(traceback.format_exc())
        return -1


def main():
    folder = './data/'
    celebrity = 'Jennifer Garner'
    size = get_gallery_size(celebrity)
    download_pictures(celebrity, folder, size)


if __name__ == '__main__':
    main()
