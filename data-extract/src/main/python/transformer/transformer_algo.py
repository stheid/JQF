import logging
import numpy as np
from tqdm import tqdm
from typing import List

from sock.rpc_interface import RPCInterface
from transformer.base import BaseFuzzer
# from transformer.dataset import Dataset
from transformer.model import *
from transformer.model import TransformerModel

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

remote = RPCInterface()


class TransformerFuzzer(BaseFuzzer):
    """
    This class implements a fuzzer using Transformer
    """

    def __init__(self, max_input_len=500, n_sample_candidates=10,
                 n_sample_positions=100, epochs=10, exp=6, vocab_size=100, sequence_length=20, batch_size=64,
                 embed_dim=256, latent_dim=2048, num_heads=8):
        super().__init__()
        self.currfile = None
        self.n_discarded_inputs = 0
        self.exp = exp
        self.n_sample_candidates = n_sample_candidates
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_input_len = max_input_len
        self.n_sample_positions = n_sample_positions

        self.model = TransformerModel(vocab_size=vocab_size, sequence_length=sequence_length,
                                      batch_size=batch_size, embed_dim=embed_dim, latent_dim=latent_dim,
                                      num_heads=num_heads, max_output_len=sequence_length)

        # uint8, float32, samples×width
        self.batch = []
        self.events = 100
        self.new_seqs = []
        self.new_files = []
        self.event_bitsize = 8  # bytes
        self.val_data = Dataset(bitsize=self.event_bitsize)
        self.train_data = Dataset(bitsize=self.event_bitsize)

        remote.obj = self

    @remote.register("bitsize")
    def set_bitsize(self, d: int):
        if d == 8 or d == 16 or d == 32:
            self.event_bitsize = d
        else:
            self.event_bitsize = 8
            logger.warn(
                "Incorrect event_bitsize. Currently only supported: BYTES, SHORT and INT. Defaulting to BYTES")

    @remote.register("totalevents")
    def get_total_events(self, n: int):
        logger.debug(f"total events {n}")
        self.events = n

    @remote.register("pretrain")
    def pretrain(self, seqs: List[bytes], status: List[bytes], files: List[bytes]):
        """
        :param files: List of input files as bytes, represents output to the model
        :param seqs: represents input to the model as sequences of events
        :return: None
        """
        # TODO convert sequences to list of integers

        self.train_data, self.val_data = Dataset(X=seqs, y=files, bitsize=self.event_bitsize) \
            .split(frac=0.8)

        # Initialize model and update train and val data for training
        if not self.model.is_model_created:
            logger.debug("initializing model")
            self.model.initialize_model()

        # train NN
        logger.info("Begin pretraining")
        self.model.train(self.train_data, self.val_data, epochs=self.epochs)
        logger.info("Finished pretraining")

    @remote.register("geninput")
    def geninput(self) -> List[int]:
        if not self.batch:
            logger.info(f'creating a new batch of training data')
            self.batch = self.create_inputs()
            logger.info(f'Finished creating a new batch of training data')
        logger.debug(f'geninput: len(batch) = {len(self.batch)}')
        self.currfile = self.batch.pop(0)
        return self.currfile

    def create_inputs(self) -> List[bytes]:
        create_batch_size = self.batch_size
        logger.debug(f'train_data: {len(self.train_data)}')
        logger.debug(f'batch_size: {self.model.batch_size}')
        if self.train_data.is_empty:
            raise Exception("No training data available, cannot create inputs. Pre-train first!")

        result = []
        for seq in tqdm(self.train_data.sample(create_batch_size)):
            mutated_seq = self._mutate(seq, size=10, mode="sub")
            result.append(self.model.predict(mutated_seq, self.event_bitsize))
        return result

    def _mutate(self, seqs, alpha=2, beta=1, size=5, mode="sub", seq_type: int = 1) -> List[int]:
        """
        :param seqs: an event with a sequence of bytes
        :param alpha and beta: parameters to shift sampling towards left, right, up or down
        :param size: number of samples to extract
        :param mode: substitution, removal or addition of bytes
        :param seq_type:  1 for int, 2 for short and so on
        :return: a mutated sequence of bytes for a particular event
        """
        # todo: additional mutation cases for: remove, add
        if mode == "sub":  # Substitute
            # sample geometric positions
            n_pos = np.random.beta(a=alpha, b=beta, size=size % len(seqs)) * len(seqs)

            # sample values (uniform)
            values = np.random.choice(min(self.events, (1 << self.event_bitsize) - 1), len(n_pos), replace=False)

            # mutate
            res = seqs
            for pos, val in zip(n_pos, values):
                res[int(pos)] = val

            return res

        logger.info("incorrect mode, no mutation")
        return None

    @remote.register("observe")
    def observe_single(self, status: int, seq: bytes):
        logger.debug(f'len(batch): {len(self.batch)}')
        if status != 0:
            self.new_files.append(self.currfile)
            self.new_seqs.append(seq)
        else:
            # result.status != SUCCESS, so discarding inputs
            self.n_discarded_inputs += 1

        if len(self.new_files) == self.batch_size:
            logger.info(f'discarded inputs: {self.n_discarded_inputs}')
            self.n_discarded_inputs = 0

            logger.info("collected enough data to train again.")
            self.update(Dataset(self.new_seqs, self.new_files, bitsize=self.event_bitsize))
            self.new_seqs.clear()
            self.new_files.clear()

    def update(self, new_data: Dataset):
        train_data, val_data = new_data.split(frac=0.9)

        # Initialize model and update train and val data for training
        if self.model.is_model_created:
            self.train_data += train_data
            self.val_data += val_data
        else:
            self.model.initialize_model()
            self.train_data = train_data
            self.val_data = val_data

        # train NN
        self.model.train(self.train_data, self.val_data, epochs=self.epochs)


if __name__ == '__main__':
    gen = TransformerFuzzer(max_input_len=500, epochs=1, exp=6, vocab_size=100, sequence_length=20,
                            batch_size=64, embed_dim=256, latent_dim=2048, num_heads=8)
    # gen.set_bitsize(16)
    # # seqs, files = load_jqf("/home/ajrox/Programs/pylibfuzzer/examples/transformer_jqf/data/fuzz-results/")
    # seqs, files = load_jqf("/home/ajrox/Programs/JQF/fuzz-results-runner/")
    # gen.pretrain(seqs, None, files)
    # gen.get_total_events(65536)
    # a = gen.geninput()
    # print(a)
    # import os
    # import random
    # #
    # for i in range(128):
    #     seq = [random.randint(0, 20) for _ in range(10)]
    #     sc = 0
    #     if i % 2 == 0:
    #         sc = 1
    #     gen.observe_single(sc, seq)
    # create
    remote.run()
