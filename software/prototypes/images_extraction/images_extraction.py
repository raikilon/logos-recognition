from bs4 import BeautifulSoup
import requests
import requests.exceptions
import os
from urllib import parse
import re


class ImagesExtraction(object):
    def __init__(self, images_folder, max_images):
        """
        Instantiate parameters
        """
        # List of all visited urls
        self.url_list = []
        # Names of all visited images
        self.img_visited = []
        # Max number of iteration (depth)
        self.num = 0
        # Max number of downloaded images
        self.max_images = max_images
        # Directory to save the images
        self.folder = images_folder
        # if the directory does not exist create it
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

    def single_page(self, site):
        """
        Downloads images from a single page
        :param site: url to process
        :return: nothing
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
            res = parse.urlparse(url)
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
                                res_site = parse.urlparse(site)
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
                extension = self.get_extention(response.headers.get('content-type'))
                # if the extension is known continue
                if extension is not '':
                    # save the image
                    f = open(
                        self.folder + parse.quote(parse.urlparse(site).netloc, '') + "_" + str(self.num) + extension,
                        'wb')
                    f.write(response.content)
                    f.close()
                    # increment the number of saved images
                    self.num += 1
            # if the number of saved images is greater than the max number stop the process
            if self.num > self.max_images:
                break

    def download_images(self, site, depth):
        """ Download images from a website by scanning all the url contained in it

        :param site: site to scan
        :param depth: max depth of the research (1 is equals to scan the given url and all the urls contained in it)
        :return:
        """
        # if the url is already visited or the number of saved images is greater than the max number, stop the process
        if site in self.url_list or self.num > self.max_images:
            return
        # add current site to the list of visited urls
        self.url_list.append(site)
        # extract all images from current url
        self.single_page(site)
        # if depth is greater than 0 continue with the images extraction
        if depth > 0:
            # create html parser
            soup = BeautifulSoup(requests.get(site, allow_redirects=True).text, 'html.parser')
            # extract all the url from a tags
            links = soup.findAll('a')
            # Loop over all links
            for link in links:
                # read href and clean the content
                url = link['href']
                url = re.sub(r'[\t\n\r]', '', url)
                res = parse.urlparse(url)
                res_site = parse.urlparse(site)
                # continue the process only if the url is in the same website
                if res_site.netloc is not '' and res_site.netloc in res.geturl():
                    # check if website www.xxx.yy is not empty
                    if res.netloc is not '':
                        # extract images from this url (recursion)
                        self.download_images(res.geturl(), depth - 1)
                    # Check if the url is relative and if it is concat with the base url
                    if res.netloc is '' and res.path is not '':
                        # extract images from this url (recursion)
                        self.download_images(parse.urljoin(site, url), depth - 1)

    def get_extention(self, content_type):
        """ Given the content type return the image extension

        :param content_type: content type (from the header)
        :return: the image extension
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


def main():
    # ----------------- CHANGE DATA HERE -----------------
    images_extractions = ImagesExtraction('images/', 200) # Folder to save the image
    images_extractions.download_images("WEBSITE", 1) #Website, number of level (how many link inside the page must process)


if __name__ == "__main__":
    main()
