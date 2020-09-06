from lib.imageExtractor import ImageExtractor
from bs4 import BeautifulSoup
import urllib.parse, urllib.request
import requests
import requests.exceptions
import re
import os
import shutil


class WebsiteExtractor(ImageExtractor):
    """
    This class allows to download all the images from a given website.
    """

    def __init__(self):
        """
        Initiate the Image extractor by creating the folder for the downloaded images in data/imageExtractor and
        initialize other important parameters.
        """
        # List of all visited urls
        self.url_list = []
        # Names of all visited images
        self.img_visited = []
        # Max number of images
        self.num = 0
        # Directory to save the images
        self.folder = "data/imageExtractor/download"
        # if the directory does not exist create it
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def extract(self, website, parameters):
        """
        Extract the given images from the given website. If parameters do not correspond to the implementation raise a
        ValueError. All the images must be saved locally.

        :param website: Website URL (http://www.google.com/)
        :param parameters: Parameters for the implementation
        :return: List of local path for the extracted images
        """
        # Check parameters
        if len(parameters) == 1:
            try:
                depth = int(parameters[0])
                try:
                    # An user-agent is necessary because otherwise websites return ERROR 403 FORBIDDEN
                    request = urllib.request.Request(website, headers={'User-Agent': 'Mozilla/5.0'})
                    request.get_method = lambda: 'HEAD'
                    urllib.request.urlopen(request)
                except (urllib.request.HTTPError, urllib.request.URLError, ValueError):
                    raise AttributeError
                self.__download_images(website, depth)
                names = [name for name in os.listdir(self.folder)]
                return names
            except ValueError:
                raise ValueError
        else:
            raise ValueError

    def clear(self):
        """
         Delete all the extracted and saved images.

        :return: Nothing
        """
        shutil.rmtree(os.path.join(self.folder))

    # ############################ HELPER ############################

    def __download_images(self, website, depth):
        """
        Download images from the given website. Recursive function until depth is 0.

        :param website: Website URL
        :param depth: Depth of the recursive function (how many link must be visited)
        :return: Nothing
        """
        # if the url is already visited or the number of saved images is greater than the max number, stop the process
        if website in self.url_list:
            return
        # add current site to the list of visited urls
        self.url_list.append(website)
        # extract all images from current url
        self.__single_page(website)
        # if depth is greater than 0 continue with the images extraction
        if depth > 0:
            # create html parser
            soup = BeautifulSoup(requests.get(website, allow_redirects=True).text, 'html.parser')
            # extract all the url from a tags
            links = soup.findAll('a')
            # Loop over all links
            for link in links:
                # read href and clean the content
                try:
                    url = link['href']
                    url = re.sub(r'[\t\n\r]', '', url)
                    res = urllib.parse.urlparse(url)
                    res_site = urllib.parse.urlparse(website)
                    # continue the process only if the url is in the same website
                    if res_site.netloc is not '' and res_site.netloc in res.geturl():
                        # check if website www.xxx.yy is not empty
                        if res.netloc is not '':
                            # extract images from this url (recursion)
                            self.__download_images(res.geturl(), depth - 1)
                        # Check if the url is relative and if it is concat with the base url
                        if res.netloc is '' and res.path is not '':
                            # extract images from this url (recursion)
                            self.__download_images(urllib.parse.urljoin(website, url), depth - 1)
                except KeyError:
                    pass

    def __single_page(self, site):
        """
        Downloads images from a single page.

        :param site: URL to process
        :return: Nothing
        """
        # create html parse
        soup = BeautifulSoup(requests.get(site, allow_redirects=True).text, 'html.parser')
        # find all images
        images = soup.findAll('img')
        # List of all images urls
        urls = []
        # loops over all images and extract the url
        for img in images:
            if img.has_attr('src'):
                urls.append(img['src'])
        # loops over all urls
        for url in urls:
            response = None
            # certain img have control characters in the url WTF ;)
            url = re.sub(r'[\t\n\r]', '', url)
            # if the image is already processed skip
            if url in self.img_visited:
                continue
            # add the image url to visited images
            self.img_visited.append(url)
            # parse the url
            res = urllib.parse.urlparse(url)
            # if website is without http or https
            if res.scheme is '':
                try:
                    # Add the http and check if it works
                    response = requests.get('http://' + str(res.geturl().lstrip("/")), allow_redirects=True)
                    # if url does not exist
                    if response.status_code != 200:
                        response = None
                        raise requests.exceptions.InvalidURL
                    # print('http://' + res.geturl().lstrip("/"))

                except requests.exceptions.RequestException:
                    # check if the url contains the netloc -> www.cwi.nl/%7Eguido/Python.html -> netlog = ''
                    if res.netloc is '':
                        try:
                            # concat the base url with the img url (without the initial /)
                            response = requests.get(site + res.geturl().lstrip("/"), allow_redirects=True)
                            if response.status_code != 200:
                                response = None
                                raise requests.exceptions.InvalidURL
                            # print(site + res.geturl().lstrip("/"))
                        except requests.exceptions.RequestException:
                            try:
                                # Concat the base url (only the www.xxx.yy) with the img url (without the initial /)
                                res_site = urllib.parse.urlparse(site)
                                response = requests.get(
                                    'http://' + res_site.netloc.lstrip("/") + res.geturl().lstrip("/"),
                                    allow_redirects=True)
                                if response.status_code != 200:
                                    response = None
                                    raise requests.exceptions.InvalidURL
                                # print('http://' + res_site.netloc.lstrip("/") + res.geturl())
                            except requests.exceptions.RequestException:
                                response = None
                                # the image is discarded
            # if website has http or https
            else:
                try:
                    response = requests.get(url, allow_redirects=True)
                    if response.status_code != 200:
                        response = None
                        raise requests.exceptions.InvalidURL
                    # print(url)
                except requests.exceptions.RequestException:
                    response = None
                    # the image is discarded
            # if there is a valid response continue with the process
            if response is not None:
                # get img extension from the header
                extension = self.__get_extension(response.headers.get('content-type'))
                # if the extension is known continue
                if extension is not '':
                    # save the image
                    f = open(
                        self.folder + "/" + urllib.parse.quote(urllib.parse.urlparse(site).netloc, '') + "_" + str(
                            self.num) + extension,
                        'wb')
                    f.write(response.content)
                    f.close()
                    # increment the number of saved images
                    self.num += 1

    @staticmethod
    def __get_extension(content_type):
        """
        Given the content type return the image extension.

        :param content_type: Content type (from the header)
        :return: Image extension
        """
        if 'svg' in content_type.lower():
            return ".svg"
        if 'jpeg' in content_type.lower():
            return ".jpeg"
        if 'gif' in content_type.lower():
            return ".gif"
        if 'png' in content_type.lower():
            return ".png"
        else:
            return ''
