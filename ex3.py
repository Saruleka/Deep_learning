import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Images and labels

images = [
    "bird.png",
    "car.png",
    "cat.png",
    "dog.png",
    "horse.png"
]

labels = [
    "bird",
    "car",
    "cat",
    "dog",
    "horse"
]


# 2. Load VGG16

vgg = VGG16(
    weights="imagenet",
    include_top=False
)

# Remove original classifier
# and use Global Average Pooling

model = Model(
    vgg.input,
    GlobalAveragePooling2D()(vgg.output)
)

print("VGG16 loaded successfully")


# 3. Extract features

features = []

for image in images:

    img = load_img(
        image,
        target_size=(224, 224)
    )

    img = img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    feature = model.predict(
        img,
        verbose=0
    )

    features.append(feature[0])


features = np.array(features)


# 4. Display feature information

print("\nFeature shape:", features.shape)
print("Feature vector size:", features.shape[1])


# 5. Convert labels to numbers

encoder = LabelEncoder()

y = encoder.fit_transform(labels)

print("\nLabels:", y)


# 6. Train classifier

classifier = LogisticRegression(
    max_iter=1000
)

classifier.fit(features, y)

print("\nClassifier trained successfully")


# 7. Predict

prediction = classifier.predict(features)


# 8. Show predictions

print("\nPredictions:")

for i in range(len(images)):

    name = encoder.inverse_transform(
        [prediction[i]]
    )[0]

    print(images[i], "->", name)


# 9. Calculate results

accuracy = accuracy_score(y, prediction)
precision = precision_score(
    y, prediction, average="weighted"
)
recall = recall_score(
    y, prediction, average="weighted"
)
f1 = f1_score(
    y, prediction, average="weighted"
)


print("\n========== RESULTS ==========")

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1-score :", round(f1 * 100, 2), "%")


# 10. Confusion Matrix

cm = confusion_matrix(y, prediction)

print("\nConfusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
).plot()

plt.title("Confusion Matrix")
plt.show()