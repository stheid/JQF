import logging
import numpy as np
from typing import List

from sock.rpc_interface import RPCInterface
from transformer.base import BaseFuzzer
from transformer.model import *
from utils import remove_lsb

logger = logging.getLogger(__name__)

remote = RPCInterface()


class TransformerFuzzer(BaseFuzzer):
    """
    This class implements a fuzzer using Transformer
    """

    def __init__(self, max_input_len=500, n_sample_candidates=10,
                 n_sample_positions=100, epochs=10, exp=6, vocab_size=100, sequence_length=20, batch_size=64,
                 embed_dim=256, latent_dim=2048, num_heads=8):
        super().__init__()
        self.n_sample_candidates = n_sample_candidates
        self.events = []
        self.uncovered_bits = None
        self.exp = exp
        self.batch = []
        self.epochs = epochs
        self.max_input_len = max_input_len
        self.n_sample_positions = n_sample_positions

        self.model = TransformerModel(vocab_size=vocab_size, sequence_length=sequence_length,
                                      batch_size=batch_size, embed_dim=embed_dim, latent_dim=latent_dim,
                                      num_heads=num_heads)

        # uint8, float32, samples×width
        self.train_data = Dataset()
        self.val_data = Dataset()

    @remote.register("totalevents")
    def get_total_events(self, n: int):
        self.events = n

    @remote.register("pretrain")
    def pretrain(self, files: List[bytes], seqs: List[bytes]):
        """
        :param files: List of input files as bytes, represents output to the model
        :param seqs: represents input to the model as sequences of events
        :return: None
        """
        # todo: use Dataset() to store and split train and val data

        # prepare training and validation data from the data through the socket
        # Initialize model and update train and val data for training
        if not self.model.is_model_created:
            self.model.initialize_model()

        # train NN
        logger.info("Begin training on pre-given dataset")
        self.model.train(self.train_data, self.val_data, epochs=self.epochs)
        logger.info("Finished training on pre-given dataset")

    @remote.register("geninput")
    def create_inputs(self) -> List[bytes]:
        batch = []
        # 1. sample candidates
        mask = np.random.choice(len(self.train_data), self.model.batch_size, replace=False)
        for inputs, target in self.train_data.take[mask]:
            mutated_seq = self._mutate(inputs["encoder_inputs"])

        pass

    def _mutate(self, seqs):
        return seqs
        # todo:
        # # 2. sample geometric positions
        #
        # a = inputs["encoder_inputs"].numpy().flatten()
        # pos = np.random.geometric(p=0.3, size=self.n_sample_positions)
        # # 3. sample values (uniform)
        # values = np.random.choice(self.events, len(pos), replace=False)

    @remote.register("observe")
    def observe(self, fuzzing_result: List[bytes]):
        data = Dataset(np.array([np.frombuffer(b, dtype=np.uint8) for b in self.batch]),
                       np.array(fuzzing_result))

        if self.uncovered_bits is None:
            self.uncovered_bits = np.ones_like(fuzzing_result[0], dtype=np.uint8)

        candidate_indices = []
        for i, result in enumerate(fuzzing_result):
            rmsb = remove_lsb(result)

            if np.any(rmsb & self.uncovered_bits):
                self.uncovered_bits &= ~rmsb
                candidate_indices.append(i)

        # select only covered edges from indices calculated above
        new_data = data[tuple(candidate_indices)]

        # if there is no data then no need for training again
        if new_data.is_empty:
            logger.info("No newly covered edges: all inputs discarded.")
            return

        # split
        train_data, val_data = new_data.split(frac=0.8)

        # Initialize model and update train and val data for training
        if self.model.is_model_created:
            self.train_data += train_data
            self.val_data += val_data
        else:
            self.model.initialize_model()
            self.train_data = train_data
            self.val_data = val_data

        # train NN
        self.model.train(self.train_data, self.val_data)


if __name__ == '__main__':
    gen = TransformerFuzzer(max_input_len=500, epochs=10, exp=6, vocab_size=100, sequence_length=20,
                            batch_size=64, embed_dim=256, latent_dim=2048, num_heads=8)
    remote.run()
