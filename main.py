from algorithm import detect_document_region
from algorithm import warp_document
from image import ColorImage


def main():
    img = ColorImage.imread('images/test8.png')
    document_region = detect_document_region(img)
    region = img.draw_quadrilateral(document_region)
    region.imsave('results/region.png')
    warped_region = warp_document(img, document_region)
    warped_region.imsave('results/document.png')


if __name__ == '__main__':
    main()
