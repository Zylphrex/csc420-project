import numpy as np

from algorithm import detect_document_region
from algorithm import extract_word_bounds
from algorithm import warp_document
from image import ColorImage


def main():
    img = ColorImage.imread('images/test9.png', resize=(3840,2160))
    document_region = detect_document_region(img)
    region = img.draw_quadrilateral(document_region)
    region.imsave('results/region.png')
    warped_region = warp_document(img, document_region, 1700, 2200)
    warped_region.imsave('results/document.png')

    words = extract_word_bounds(warped_region.copy())
    img = warped_region.copy()
    for i, word in enumerate(words):
        img = word.visualize_bounds(img.img)
        word.crop(warped_region.copy().img).imsave('words/word{}.png'.format(i))
    img.imsave('results/words.png')


if __name__ == '__main__':
    main()
