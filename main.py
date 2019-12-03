import argparse

import numpy as np

from algorithm import detect_document_region
from algorithm import extract_word_bounds
from algorithm import warp_document
from image import ColorImage
from nn import extract_text


def run(args):
    # assume the image is 4K, crop if its larger
    img = ColorImage.imread(args.file_name, resize=(3840,2160))

    # step 1: find a bounding box region for the document
    document_region = detect_document_region(img)
    # visualize the document region
    region = img.draw_quadrilateral(document_region)
    region.imsave('results/region.png')

    # step 2: find text regions
    # step 2.1: perform homography to flatten out document
    # 1700 x 2200 is size of a letter size paper at 200 DPI
    warped_region = warp_document(img, document_region, 1700, 2200)
    # visualize the flattened document
    warped_region.imsave('results/document.png')
    # extract word regions from document
    words = extract_word_bounds(warped_region.copy())

    # step 3: CRNN word extraction
    texts = []
    for i, word in enumerate(words):
        # crop each word from the document
        word_img = word.crop(warped_region.img)
        # visualize each cropped word
        word_img.imsave('words/word{}.png'.format(i))
        # CRNN text extraction from cropped word
        text = extract_text(word_img.img)
        texts.append(text)
    print(' '.join(texts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run text extractor')
    parser.add_argument('file_name')
    args = parser.parse_args()
    run(args)
