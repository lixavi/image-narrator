import tensorflow as tf

class RNNModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.model = self.build_model()

    def build_model(self):
        embedding_dim = 256
        units = 512

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, embedding_dim, mask_zero=True),
            tf.keras.layers.LSTM(units, return_sequences=True),
            tf.keras.layers.Dense(self.vocab_size)
        ])
        return model

    def train(self, captions):
        # Tokenize captions and preprocess if necessary
        # Implement training code here
        pass

    def generate_caption(self, image_features):
        # Implement code to generate caption using the RNN model
        start_token = '<start>'
        end_token = '<end>'
        max_length = 20

        input_sequence = [start_token]
        image_features = tf.expand_dims(image_features, axis=0)

        for _ in range(max_length):
            sequence = [self.tokenizer.word_index[word] for word in input_sequence]
            sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], padding='post', maxlen=max_length)
            predictions = self.model.predict([image_features, sequence])
            predicted_id = tf.argmax(predictions, axis=-1).numpy()
            if predicted_id == self.tokenizer.word_index[end_token]:
                break
            input_sequence.append(self.tokenizer.index_word[predicted_id[0][0]])

        return ' '.join(input_sequence)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
