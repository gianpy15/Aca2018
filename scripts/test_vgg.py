from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import os

model = VGG16(weights='imagenet')

img_path = '../resources/test_imgs/'
images = []
for img in os.listdir(img_path):
    tmp = image.load_img(img_path + img, target_size=(224, 224))
    x = image.img_to_array(tmp)
    x = preprocess_input(x)
    images.append(x)

images = np.array(images)

preds = model.predict(images)

for pred in preds:
    print('Predicted:', decode_predictions(np.expand_dims(pred, axis=0), top=3)[0])
