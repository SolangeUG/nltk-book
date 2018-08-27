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

    # count will keep track of total number of pictures in the entire gallery
    count = 0
    if gallery_size > -1:
        for i in range(1, gallery_size):
            page_address = 'http://www.zimbio.com/photos/%s/browse?Page=%s' \
                           % (celebrity.replace(' ', '+'), str(i))
            with urllib.request.urlopen(page_address) as response:
                html = response.read()

            # parse each gallery page and retrieve picture URLs
            page = BeautifulSoup(html, 'html.parser')
            links = page.select('.thumbnail-item')

            for link in links:
                # increment count for each link to a picture in the gallery
                count += 1

                # The page element containing the picture URL of our interest is of the form:

                # <a class="thumbnail-item"
                #    href="/photos/{celebrity}/{celebrity}+gallery+name/pictureID"
                #                  |---------------------------------------------|
                #                                       suffix
                #    title="title of the gallery name">
                #    <img alt="picture descriptive text"
                #         src="http://www3.pictures/zimbio.com/bg/{celebrity}+some+title/pictureIDs.jpg">
                #              |---------------------------------|
                #                           prefix
                # </a>

                # So, the URL to the large size version of the picture we're interested in is of the form:
                # prefix + suffix + 'x.jpg' where prefix and suffix are as shown above.

                href = link.get('href')
                image = link.select('img')[0]
                source = image.get('src')
                try:
                    prefix = source.split(celebrity.replace(' ', '+'))[0]
                    suffix = href.split('/photos/')[1]
                    image_url = prefix + suffix + 'x.jpg'

                    imagefilename = '%s_%s.jpg' % (celebrity.replace(' ', '_'), str(count))
                    imagefilepath = os.path.join(directory_name, imagefilename)
                    if not os.path.exists(imagefilepath):
                        urllib.request.urlretrieve(image_url, imagefilepath)
                except Exception as e:
                    print('Error type: %s' % str(e))
                    print(traceback.format_exc())


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
