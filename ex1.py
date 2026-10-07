import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.utils import load_img, img_to_array

# ---------------------------------------------------
# 1. Load Pre-trained VGG16 model
# ---------------------------------------------------

model = ResNet50(weights="imagenet")

print("Model loaded successfully")


# ---------------------------------------------------
# 2. Load an image
# ---------------------------------------------------

image_path = "dog.png"

img = load_img(image_path, target_size=(224, 224))

print("Original image size:", img.size)


# ---------------------------------------------------
# 3. Convert image to NumPy array
# ---------------------------------------------------

img_array = img_to_array(img)

print("After converting to array:", img_array.shape)


# ---------------------------------------------------
# 4. Add batch dimension
# ---------------------------------------------------

img_batch = np.expand_dims(img_array, axis=0)

print("After adding batch:", img_batch.shape)


# ---------------------------------------------------
# 5. Preprocess image for VGG16
# ---------------------------------------------------

processed_image = preprocess_input(img_batch)

print("After preprocessing:", processed_image.shape)


# ---------------------------------------------------
# 6. Display input image
# ---------------------------------------------------

plt.imshow(img)
plt.axis("off")
plt.title("Input Image")
plt.show()


# ---------------------------------------------------
# 7. Predict ImageNet class
# ---------------------------------------------------

predictions = model.predict(processed_image)


# ---------------------------------------------------
# 8. Get Top-5 predictions
# ---------------------------------------------------

top5 = decode_predictions(predictions, top=5)[0]

print("\nTop-5 Predictions:")

for i, (class_id, class_name, probability) in enumerate(top5):
    print(
        i + 1,
        class_name,
        " : ",
        round(probability * 100, 2),
        "%"
    )


# ---------------------------------------------------
# 9. Extract intermediate representation
# ---------------------------------------------------

# Get output from the last convolutional layer
feature_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("block2_conv2").output
)

features = feature_model.predict(processed_image)

print("\nIntermediate Representation:")
print("Shape:", features.shape)


# ---------------------------------------------------
# 10. Display one feature map
# ---------------------------------------------------

plt.imshow(features[0, :, :, 0], cmap="viridis")
plt.title("Intermediate CNN Feature Map")
plt.show()