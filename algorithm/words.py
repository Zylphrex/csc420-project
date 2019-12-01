import datetime

import algorithm


def extract_word_bounds(img):
    gray = img.gray()
    gradient = gray.gradient()
    canny = gradient.canny(100, 200)
    gradient.imsave('results/word_edges.png')

    words = algorithm.detect_words(canny)

    words = algorithm.try_merge_words(words)

    words = sorted(words, key=lambda w: (w.min_y, w.min_x))

    last_y = None
    map_y = {}
    for word in words:
        if last_y is None or word.min_y - last_y >= 30:
            last_y = word.min_y

        map_y[word.min_y] = last_y

    words = sorted(words, key=lambda w: (map_y[w.min_y], w.min_x))

    return words
