import image
import image.gradient as gradient


class GrayImage(image.Image):
    def gradient(self):
        return gradient.ImageGradient(self.img)


