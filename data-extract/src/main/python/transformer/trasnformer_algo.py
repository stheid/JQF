import logging
import numpy as np
from typing import List

from tqdm import trange, tqdm

from python.transformer.base import BaseFuzzer
from python.transformer.datasetio import DatasetIO
from python.transformer.model import *

logger = logging.getLogger(__name__)


class TransformerFuzzer(BaseFuzzer):
    """
    This class implements a fuzzer using Transformer
    """

    def __init__(self, initial_dataset_len, dataset=None, max_input_len=500, n_mutation_candidates=10,
                 n_mutation_positions=100, epochs=10, exp=6, vocab_size=100, sequence_length=20, batch_size=64,
                 embed_dim=256, latent_dim=2048, num_heads=8):
        super().__init__()
        self.exp = exp
        self.batch = []
        self.epochs = epochs
        self._do_warmup = True
        self.dataset = dataset
        self.initial_dataset_len = initial_dataset_len
        self.max_input_len = max_input_len
        self.n_mutation_positions = n_mutation_positions
        self.n_mutation_candidates = n_mutation_candidates

        self.model = TransformerModel(epochs=epochs, vocab_size=vocab_size, sequence_length=sequence_length,
                                      batch_size=batch_size, embed_dim=embed_dim, latent_dim=latent_dim,
                                      num_heads=num_heads)

        # uint8, float32, samples×width
        self.train_data = None
        self.val_data = None

    def prepare(self):
        """ prepare Runner to be able to create inputs before loading seeds.
            - takes care of things like: initialize model and pre-train model if dataset exists """

        if self.dataset is not None:
            zip_pairs = DatasetIO.load_jqf(dataset_path=self.dataset)
            self.train_data, self.val_data = DatasetIO.preprocess_data_transformer(zip_pairs)

        # Initialize model and update train and val data for training
        if not self.model.is_model_created:
            # todo: add dimension information as argument
            self.model.initialize_model()

        # train NN
        logger.info("Begin training on pre-given dataset")
        self.model.train(self.train_data, self.val_data)
        logger.info("Finished training on pre-given dataset")

        self._do_warmup = False

    def load_seed(self, seedfiles):
        pass
        # inputs = []
        # # todo: remove hard-coding below
        # pbar = tqdm(total=50000)
        # with open(os.path.join(seedfiles, "events.bin"), "rb") as f:
        #     while length := f.read(4):
        #         n = struct.unpack(">i", length)[0]
        #         pbar.update(1)
        #         # todo: missing padding part because the seeds are loaded differently than neuzz
        #         inputs.append(" ".join(map(str, flatten(struct.iter_unpack(">h", f.read(n * 2))))))
        # pbar.close()
        #
        # # cut and pad seedfiles to be exact self.max_input_length
        # # input = input[:self.max_input_len] + bytes(bytearray(max(0, self.max_input_len - len(input))))
        #
        # self.batch.append(inputs)

    def create_inputs(self) -> List[bytes]:
        """
        - use JQF to create (input, output)
        """
        if self._do_warmup:
            self._do_warmup = False

            batch = self.batch
            # fill up with random inputs to match desired size
            for _ in trange(len(batch), self.initial_dataset_len):
                file_len = np.random.randint(self.max_input_len)
                batch.append(np.random.bytes(file_len) + bytes(bytearray(self.max_input_len - file_len)))
        else:
            # todo: pending
            batch = self._generate_inputs()

        self.batch = batch
        return self.batch

    def observe(self, fuzzing_result: List[bytes]):
        # data = Dataset(np.array([np.frombuffer(b, dtype=np.uint8) for b in self.batch]),
        #                np.array(fuzzing_result))
        #
        # if self.uncovered_bits is None:
        #     self.uncovered_bits = np.ones_like(fuzzing_result[0], dtype=np.uint8)
        #
        # candidate_indices = []
        # for i, result in enumerate(fuzzing_result):
        #     rmsb = remove_lsb(result)
        #
        #     if np.any(rmsb & self.uncovered_bits):
        #         self.uncovered_bits &= ~rmsb
        #         candidate_indices.append(i)
        #
        # # select only covered edges from indices calculated above
        # new_data = data[tuple(candidate_indices)]
        #
        # # if there is no data then no need for training again
        # if new_data.is_empty:
        #     logger.info("No newly covered edges: all inputs discarded.")
        #     return
        #
        # # split
        # train_data, val_data = new_data.split(frac=0.8)
        #
        # # Initialize model and update train and val data for training
        # if self.model.is_model_created:
        #     self.train_data += train_data
        #     self.val_data += val_data
        # else:
        #     self.model.initialize_model(train_data.xdim, train_data.ydim, network=self.network)
        #     self.train_data = train_data
        #     self.val_data = val_data
        #
        # # train NN
        # self.model.train(self.train_data, self.val_data)
        pass

    # todo: pending
    @staticmethod
    def _gen_mutations(candidate, grad, exp=6) -> List[np.array]:
        candidate = candidate.squeeze()
        grad = grad.squeeze()
        successors = []

        # k^exp largest indices with k=2
        locs = np.argpartition(np.abs(grad), 2 ** exp)[-2 ** exp:].squeeze()

        for loc in locs:
            for m in range(1, 256):
                new = candidate.copy()
                v = candidate[loc] + m * np.sign(grad[loc])
                if v == np.clip(v, 0, 255):
                    new[loc] = v
                    successors.append(new)
                else:
                    break
        return successors

    # todo: pending
    def _generate_inputs(self) -> List[bytes]:
        # select collection of parent files
        mask = np.random.choice(len(self.train_data), self.n_mutation_candidates, replace=False)
        candidates = self.train_data.X[mask]
        candidate_labels = self.train_data.y[mask]

        inputs = []
        for x, y in zip(tqdm(candidates), candidate_labels):
            # enable additional coverage bits in the output
            # set some "close to zero"-values to one (means we enable some additional coverage markers)
            zero_idxs = np.where(y == 0)[0]
            new_ones = np.random.choice(zero_idxs, self.n_mutation_positions, replace=False)
            y[new_ones] = 1

            # backpropagation to the input and get the gradients [Keras]
            gradient = self.model.gradient(x, y)

            # generate new mutated inputs using the gradient and exhaustive search (Algo 1 from the paper)
            mutations = TransformerFuzzer._gen_mutations(x, gradient, exp=self.exp)
            bytes_mutations = [x.tobytes() for x in mutations]
            inputs.extend(bytes_mutations)

        return inputs
