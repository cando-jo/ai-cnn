from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from google.colab import files
import io

def tahmin_etme(data):
  # Predicts the model
  prediction = model.predict(data)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]

  print("Class:", class_name[2:], end="")
  if (class_name[2:] == "Your class name"):
    print("nohut, buğday, arpa, tohum, karabuğday, darı, bezelye, mercimek ikram ediyorum :D")
  else:
    print("kırık mısır, tahıl taneleri, yulaf, buğday, pirinç ve kurutulmuş böcekler ikram ediyorum")
  print("Confidence Score:", confidence_score)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = files.upload()

# Get the first uploaded image
image_name = next(iter(image))
image_data = image[image_name]

# Create an Image object from the uploaded image data
image = Image.open(io.BytesIO(image_data))

# Resize the image
size = (224, 224)

image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

tahmin_etme(data)

