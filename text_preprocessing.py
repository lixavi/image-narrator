from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_text(captions):
    # Initialize Tokenizer
    tokenizer = Tokenizer()
    # Fit tokenizer on captions
    tokenizer.fit_on_texts(captions)
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(captions)
    # Get word index
    word_index = tokenizer.word_index
    # Pad sequences to ensure uniform length
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences, word_index
