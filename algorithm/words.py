import algorithm


def extract_word_bounds(img):
    gray = img.gray()
    gradient = gray.gradient()
    canny = gradient.canny(100, 200)
    gradient.imsave('results/word_edges.png')

    words = algorithm.detect_words(canny)
    words = algorithm.try_merge_words(words)
    words = sorted(words, key=lambda w: (w.min_y, w.min_x))
    return words
