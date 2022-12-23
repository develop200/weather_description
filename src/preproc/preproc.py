from torchvision.transforms import ToTensor, Compose, Resize
import cv2

classes = ['cloudy', 'snow', 'rain', 'fogsmog', 'shine']

image_size = 256
transform = Compose([
                    ToTensor(),
                    Resize((image_size, image_size)),
])

def preproc_image(file):
    open("image.jpg", "wb").write(file.read())
    img = cv2.imread("image.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)
    return img[None, :, :, :]
