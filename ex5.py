import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, Reshape, Conv2DTranspose, Conv2D, LeakyReLU, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import load_img, img_to_array

# ---------------------------------------------------
# 1. Define Local Images and Text
# ---------------------------------------------------
image_paths = ["dog.jpg", "tiger.jpg", "elephant.jpg", "car.jpg", "airplane.jpg"]
captions = ["a photo of a dog", "a photo of a tiger", "a photo of an elephant", "a photo of a car", "a photo of an airplane"]

# ---------------------------------------------------
# 2. Load Images
# ---------------------------------------------------
images = np.array([img_to_array(load_img(path, target_size=(64, 64))) / 127.5 - 1 for path in image_paths])
print("Image shape:", images.shape)

# ---------------------------------------------------
# 3. Text Encoding
# ---------------------------------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
sequences = pad_sequences(tokenizer.texts_to_sequences(captions), maxlen=6, padding="post")
vocab_size = len(tokenizer.word_index) + 1

text_input = Input(shape=(6,))
x = Embedding(vocab_size, 128)(text_input)
x = GlobalAveragePooling1D()(x)
text_encoder = Model(text_input, x)

text_features = text_encoder.predict(sequences, verbose=0)
print("Text feature shape:", text_features.shape)

# ---------------------------------------------------
# 4. Conditional Generator
# ---------------------------------------------------
noise_input = Input(shape=(100,))
text_input = Input(shape=(128,))
x = Concatenate()([noise_input, text_input])
x = Dense(8 * 8 * 256, activation="relu")(x)
x = Reshape((8, 8, 256))(x)
x = Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu")(x)
x = Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu")(x)
output = Conv2DTranspose(3, 3, padding="same", activation="tanh")(x)
generator = Model([noise_input, text_input], output)

# ---------------------------------------------------
# 5. Conditional Discriminator
# ---------------------------------------------------
image_input = Input(shape=(64, 64, 3))
text_input = Input(shape=(128,))
x = Conv2D(64, 4, strides=2, padding="same")(image_input)
x = LeakyReLU(0.2)(x)
x = Conv2D(128, 4, strides=2, padding="same")(x)
x = LeakyReLU(0.2)(x)
x = GlobalAveragePooling2D()(x)
x = Concatenate()([x, text_input])
output = Dense(1, activation="sigmoid")(x)
discriminator = Model([image_input, text_input], output)

# ---------------------------------------------------
# 6. Compile Discriminator
# ---------------------------------------------------
discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ---------------------------------------------------
# 7. Generate Image
# ---------------------------------------------------
noise = np.random.normal(0, 1, (1, 100))
generated_image = generator.predict([noise, text_features[1].reshape(1, -1)], verbose=0)

# ---------------------------------------------------
# 8. Display Generated Image
# ---------------------------------------------------
generated_image = (generated_image[0] + 1) / 2
plt.imshow(generated_image)
plt.title("Generated: a photo of a tiger")
plt.axis("off")
plt.show()