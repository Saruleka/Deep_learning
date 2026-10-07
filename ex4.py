import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

# 1. Load Pre-trained CNN
cnn = VGG16(weights="imagenet", include_top=False, pooling="avg")
print("CNN loaded successfully")
print("CNN output shape:", cnn.output_shape)

# 2. Define Images and Labels
image_paths = ["dog.png", "tiger.png", "elephant.png", "car.png", "airplane.png"]
labels = ["dog", "tiger", "elephant", "car", "airplane"]

# 3. Create Captions
captions = ["<START> a photo of a " + label + " <END>" for label in labels]
print("\nCaptions:")
for caption in captions: print(caption)

# 4. Load Images
images = np.array([img_to_array(load_img(path, target_size=(224, 224))) for path in image_paths])
print("\nImage shape:", images.shape)

# 5. Preprocess Images
processed_images = preprocess_input(images)
print("Preprocessed shape:", processed_images.shape)

# 6. Extract CNN Features
features = cnn.predict(processed_images, verbose=0)
print("Feature shape:", features.shape)

# 7. Display Images
plt.figure(figsize=(12, 6))
for i in range(len(images)):
    plt.subplot(1, 5, i + 1); plt.imshow(images[i].astype("uint8")); plt.title(labels[i]); plt.axis("off")
plt.tight_layout(); plt.show()

# 8. Tokenization
tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
print("\nVocabulary:", tokenizer.word_index)
print("Vocabulary size:", vocab_size)

# 9. Convert Captions to Sequences
sequences = tokenizer.texts_to_sequences(captions)
print("\nSequences:", sequences)

# 10. Create Input and Output Sequences
input_sequences = []; output_words = []; image_features = []
for i, sequence in enumerate(sequences):
    for j in range(1, len(sequence)):
        input_sequences.append(sequence[:j]); output_words.append(sequence[j]); image_features.append(features[i])

# 11. Padding
max_length = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding="pre")
output_words = np.array(output_words)
image_features = np.array(image_features)
print("\nInput shape:", input_sequences.shape)
print("Output shape:", output_words.shape)
print("Image feature shape:", image_features.shape)

# 12. CNN-LSTM Architecture
image_input = Input(shape=(features.shape[1],))
caption_input = Input(shape=(max_length,))
hidden_state = Dense(256, activation="tanh")(image_input)
cell_state = Dense(256, activation="tanh")(image_input)
embedding = Embedding(vocab_size, 128)(caption_input)
lstm_output = LSTM(256)(embedding, initial_state=[hidden_state, cell_state])
output = Dense(vocab_size, activation="softmax")(lstm_output)
model = Model([image_input, caption_input], output)

# 13. Compile Model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 14. Train Model
history = model.fit([image_features, input_sequences], output_words, epochs=100, batch_size=4, verbose=1)

# 15. Generate Caption
def generate_caption(feature):
    current = [tokenizer.word_index["<start>"]]; words = []
    for _ in range(max_length):
        sequence = pad_sequences([current], maxlen=max_length, padding="pre")
        prediction = model.predict([feature.reshape(1, -1), sequence], verbose=0)
        predicted_id = np.argmax(prediction[0])
        if predicted_id == tokenizer.word_index["<end>"]: break
        word = next((w for w, i in tokenizer.word_index.items() if i == predicted_id), None)
        if word is None: break
        words.append(word); current.append(predicted_id)
    return " ".join(words)

# 16. Test Caption Generation
test_index = 0
generated_caption = generate_caption(features[test_index])
print("\nOriginal Caption:", captions[test_index])
print("Generated Caption:", generated_caption)

# 17. Display Result
plt.figure(figsize=(6, 6)); plt.imshow(images[test_index].astype("uint8")); plt.axis("off"); plt.title("Generated: " + generated_caption); plt.show()