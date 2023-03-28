from tensorflow.keras import layers
import tensorflow.keras as keras
from python.transformer.layers import *


class TransformerModel:

    def __init__(self, epochs=5, vocab_size=100, sequence_length=20, batch_size=64,
                 embed_dim=256, latent_dim=2048, num_heads=8):
        self.epochs = epochs
        self.model = None
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads

    # todo: add dimension information as argument
    def initialize_model(self):
        encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
        x = PositionalEmbedding(self.sequence_length, self.vocab_size, self.embed_dim)(encoder_inputs)
        encoder_outputs = TransformerEncoder(self.embed_dim, self.latent_dim, self.num_heads)(x)
        encoder = keras.Model(encoder_inputs, encoder_outputs)

        decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
        encoded_seq_inputs = keras.Input(shape=(None, self.embed_dim), name="decoder_state_inputs")
        x = PositionalEmbedding(self.sequence_length, self.vocab_size, self.embed_dim)(decoder_inputs)
        x = TransformerDecoder(self.embed_dim, self.latent_dim, self.num_heads)(x, encoded_seq_inputs)
        x = layers.Dropout(0.5)(x)
        decoder_outputs = layers.Dense(self.vocab_size, activation="softmax")(x)
        decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

        decoder_outputs = decoder([decoder_inputs, encoder_outputs])
        self.model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="transformer")
        self.model.compile("rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model.summary()

    @property
    def is_model_created(self):
        return self.model is not None

    def train(self, data: tf.data.Dataset, val_data: tf.data.Dataset):
        self.model.fit(data, epochs=self.epochs, validation_data=val_data)
