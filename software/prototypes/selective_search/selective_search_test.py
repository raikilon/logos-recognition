import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import cv2


def AlpacaDB(img):
    """
    Generate bounding box and show images with them
    :param img: image to process
    :return: nothing
    """
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=100, sigma=0.8, min_size=10)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding rectangle with height = 0 or width = 0
        if r['rect'][2] == 0 or r['rect'][3] == 0:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()


def OpenCV(im):
    """
    Generate bounding box and show images with them
    :param im: image to process
    :return: nothing
    """
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # set input image on which we will run segmentation
    ss.setBaseImage(im)

    ss.switchToSelectiveSearchQuality()

    # run selective search segmentation on input image
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))

    # number of region proposals to show
    numShowRects = 100

    # create a copy of original image
    imOut = im.copy()

    # iterate over all the region proposals
    for i, rect in enumerate(rects):
        # draw rectangle for region proposal till numShowRects
        if i < numShowRects:
            x, y, w, h = rect
            cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            break
    # show output
    cv2.imshow("Output", imOut)
    cv2.waitKey()


def main():
    # AlpacaDB(skimage.data.load("C:/Dev/git/Logos-Recognition-for-Webshop-Services/logorec/resources/images/banner/other/stockLogoBannerDesignServices.jpg"))
    OpenCV(cv2.imread("C:/Dev/git/Logos-Recognition-for-Webshop-Services/logorec/resources/images/banner/logos/pic_013.jpg"))


if __name__ == "__main__":
    main()