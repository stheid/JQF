import logging
import numpy as np
import struct
import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import TextVectorization
from more_itertools import flatten
from tensorflow.keras import layers
from typing import List

from transformer.dataset import Dataset
from transformer.layers import *

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TransformerModel:

    def __init__(self, vocab_size=100, sequence_length=20, batch_size=64,
                 embed_dim=256, latent_dim=2048, num_heads=8, max_output_len=20):
        self.doc_index_lookup = None
        self.transformer = None
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.max_output_len = max_output_len
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        self.seq_vectorization = TextVectorization(
            max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length,
        )
        self.doc_vectorization = TextVectorization(
            max_tokens=vocab_size,
            output_mode="int",
            output_sequence_length=sequence_length + 1
        )

    # todo: add dimension information as argument
    def initialize_model(self):
        encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
        x = PositionalEmbedding(self.sequence_length, self.vocab_size, self.embed_dim)(encoder_inputs)
        encoder_outputs = TransformerEncoder(self.embed_dim, self.latent_dim, self.num_heads)(x)
        # encoder = keras.Model(encoder_inputs, encoder_outputs)

        decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
        encoded_seq_inputs = keras.Input(shape=(None, self.embed_dim), name="decoder_state_inputs")
        x = PositionalEmbedding(self.sequence_length, self.vocab_size, self.embed_dim)(decoder_inputs)
        x = TransformerDecoder(self.embed_dim, self.latent_dim, self.num_heads)(x, encoded_seq_inputs)
        x = layers.Dropout(0.5)(x)
        decoder_outputs = layers.Dense(self.vocab_size, activation="softmax")(x)
        decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

        decoder_outputs = decoder([decoder_inputs, encoder_outputs])
        self.transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="transformer")
        self.transformer.compile("rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.transformer.summary()

    @property
    def is_model_created(self):
        return self.transformer is not None

    def train(self, data: Dataset, val_data: Dataset, epochs, pre_train=True):
        logger.debug("preprocessing train data")
        train = self.preprocess_data_transformer(data, pretrain=pre_train)
        logger.debug("preprocessing val data")
        val = self.preprocess_data_transformer(val_data)
        logger.debug("fitting transformer")
        self.transformer.fit(train, epochs=epochs, validation_data=val)
        doc_vocab = self.doc_vectorization.get_vocabulary()
        self.doc_index_lookup = dict(zip(range(len(doc_vocab)), doc_vocab))

    def predict(self, seq, bitsize=8) -> bytes:
        """

            :param seq: sequence of events List[int]
            :return: document
            """
        # todo: convert string to int, replace unk with random num
        tokenized_input_sentence = self.seq_vectorization([" ".join(map(str, seq))])
        decoded_sentence = "[start]"
        for i in range(self.max_output_len):
            tokenized_target_sentence = self.doc_vectorization([decoded_sentence])[:, :-1]
            predictions = self.transformer([tokenized_input_sentence, tokenized_target_sentence])

            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = self.doc_index_lookup[sampled_token_index]
            decoded_sentence += " " + sampled_token

            if sampled_token == "[end]" or sampled_token == "end":
                break
        return self.str_to_bytes(decoded_sentence, bitsize)

    def preprocess_data_transformer(self, data: Dataset, pretrain: bool = False) -> tf.data.Dataset:
        seqs = data.X
        docs = data.y
        data_pair = self.bytes_to_str(seqs, docs)
        text_pairs = []
        for inp, out in data_pair:
            out = "[start] " + out + " [end]"
            text_pairs.append((inp, out))

        if pretrain:
            train_seq_texts = [pair[0] for pair in text_pairs]
            train_doc_texts = [pair[1] for pair in text_pairs]
            self.seq_vectorization.adapt(train_seq_texts)
            self.doc_vectorization.adapt(train_doc_texts)

        def format_dataset(seq, doc):
            seq = self.seq_vectorization(seq)
            doc = self.doc_vectorization(doc)
            return {"encoder_inputs": seq, "decoder_inputs": doc[:, :-1], }, doc[:, 1:]

        def make_dataset(pairs):
            in_texts, out_texts = zip(*pairs)
            in_texts = list(in_texts)
            out_texts = list(out_texts)
            dataset = tf.data.Dataset.from_tensor_slices((in_texts, out_texts))
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.map(format_dataset)
            return dataset.shuffle(2048).prefetch(16).cache()

        train_ds = make_dataset(text_pairs)

        return train_ds

    @staticmethod
    def bytes_to_str(seqs: List[int], files: bytes):
        logger.debug(f"converting bytes to string {len(seqs)} {len(files)}")

        _in, _out = [], []
        for i, o in zip(seqs, files):
            _in.append(" ".join(map(str, i)))
            _out.append(" ".join(map(str, flatten(struct.iter_unpack(">B", o)))))  # corpus/

        logger.debug("finished converting bytes to string")

        return list(zip(_in, _out))

    @staticmethod
    def str_to_bytes(seqs: str, bitsize: int = 8):
        if len(seqs) == 0:
            return seqs

        res = bytearray()
        seqs = seqs.replace("[UNK]", "")
        for seq in seqs.split()[1:]:
            try:
                if seq != "[end]" and seq != "end":
                    if bitsize == 8:
                        res.extend(struct.pack(">B", int(seq)))
                    if bitsize == 16:
                        res.extend(struct.pack(">H", int(seq)))
                    if bitsize == 32:
                        res.extend(struct.pack(">I", int(seq)))
            except struct.error as e:
                logger.error(f'failed to decode sequence to bytes: {e}')
        return bytes(res)
