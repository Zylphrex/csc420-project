from algorithm import detect_document
from image import ColorImage


def main():
    img = ColorImage.imread('images/test6.png')
    document = detect_document(img)
    img.draw_quadrilateral(document).imsave('results/test.png')


if __name__ == '__main__':
    main()
