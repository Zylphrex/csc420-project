import datetime

import algorithm


def extract_word_bounds(img):
    gray = img.gray()
    gradient = gray.gradient()
    canny = gradient.canny(100, 200)
    gradient.imsave('results/word_edges.png')

    # 2.2: character detection
    words = algorithm.detect_words(canny)

    vis_characters = img.copy()
    for i, word in enumerate(words):
        vis_characters = word.visualize_bounds(vis_characters.img)
    vis_characters.imsave('results/characters_bounds.png')

    # 2.3:  word detection
    words = algorithm.try_merge_words(words)

    # sort the words so we can output them in the order on
    # the paper
    words = sorted(words, key=lambda w: (w.min_y, w.min_x))

    last_y = None
    map_y = {}
    for word in words:
        if last_y is None or word.min_y - last_y >= 30:
            last_y = word.min_y

        map_y[word.min_y] = last_y

    words = sorted(words, key=lambda w: (map_y[w.min_y], w.min_x))

    vis_words = img.copy()
    for i, word in enumerate(words):
        vis_words = word.visualize_bounds(vis_words.img)
    vis_words.imsave('results/word_bounds.png')

    return words
