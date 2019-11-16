import algorithm
import geometry
import image


def detect_document(img):
    small_img = img.resize(640, 360)
    gray_img = small_img.gray()

    gradient = gray_img.gradient()
    canny = gradient.canny(100, 200)
    point_groups = algorithm.detect_point_groups(canny, gradient.direction)

    point_groups = list(filter(algorithm.at_least(10), point_groups))

    point_groups = algorithm.try_merge_point_groups(small_img, point_groups)

    point_groups = list(filter(algorithm.at_least(60), point_groups))
    to_line = lambda group: geometry.Line.from_points(group * 6)
    lines = list(map(to_line, point_groups))

    quadrilateral = algorithm.detect_quadrilateral(img, lines)

    return quadrilateral
