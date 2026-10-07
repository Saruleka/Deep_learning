import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array


# --------------------------------------------------
# 1. Load pre-trained VGG16
# --------------------------------------------------

model = VGG16(weights="imagenet")

print("VGG16 loaded successfully")


# --------------------------------------------------
# 2. Multiple input images
# --------------------------------------------------

image_paths = [
    "dog.png",
    "cat.png",
    "car.png",
    "bird.png",
    "horse.png"
]


# --------------------------------------------------
# 3. Select CNN layers
# --------------------------------------------------

layer_names = [
    "block1_conv1",    # Early layer
    "block3_conv3",    # Intermediate layer
    "block5_conv3"     # Deep layer
]

# --------------------------------------------------
# 4. Create feature extraction model
# --------------------------------------------------

outputs = [
    model.get_layer(name).output
    for name in layer_names
]

feature_model = tf.keras.Model(
    inputs=model.input,
    outputs=outputs
)


# --------------------------------------------------
# 5. Process each image
# --------------------------------------------------

for image_path in image_paths:

    print("\n====================================")
    print("Processing:", image_path)
    print("====================================")

    # Load image
    img = load_img(
        image_path,
        target_size=(224, 224)
    )

    # Convert image to array
    img_array = img_to_array(img)

    # Add batch dimension
    img_batch = np.expand_dims(
        img_array,
        axis=0
    )

    # Preprocess for VGG16
    processed_image = preprocess_input(
        img_batch
    )


    # --------------------------------------------------
    # Display original image
    # --------------------------------------------------

    plt.figure(figsize=(4, 4))

    plt.imshow(img)

    plt.title("Input: " + image_path)

    plt.axis("off")

    plt.show()


    # --------------------------------------------------
    # Extract feature maps
    # --------------------------------------------------

    feature_maps = feature_model.predict(
        processed_image,
        verbose=0
    )


    # --------------------------------------------------
    # Display dimensions
    # --------------------------------------------------

    print("\nFeature Map Dimensions:")

    for name, feature in zip(
        layer_names,
        feature_maps
    ):

        print(
            name,
            "->",
            feature.shape
        )


    # --------------------------------------------------
    # Visualize feature maps
    # --------------------------------------------------

    for name, feature in zip(
        layer_names,
        feature_maps
    ):

        plt.figure(figsize=(12, 6))

        # Display first 6 feature maps
        for i in range(6):

            plt.subplot(2, 3, i + 1)

            plt.imshow(
                feature[0, :, :, i],
                cmap="viridis"
            )

            plt.axis("off")

            plt.title(
                "Feature Map " + str(i + 1)
            )

        plt.suptitle(
            image_path + " - " + name
        )

        plt.show()

# --------------------------------------------------
# 6. Visualize learned filters
# --------------------------------------------------

layer = model.get_layer("block1_conv1")

filters, biases = layer.get_weights()

print("\nFirst layer filter shape:")
print(filters.shape)


# Normalize filters
filters = (
    filters - filters.min()
) / (
    filters.max() - filters.min()
)

# Display first 8 filters
plt.figure(figsize=(12, 6))

for i in range(8):

    plt.subplot(2, 4, i + 1)

    plt.imshow(filters[:, :, :, i])

    plt.axis("off")

    plt.title("Filter " + str(i + 1))

plt.suptitle(
    "VGG16 Learned Filters - block1_conv1"
)

plt.show()