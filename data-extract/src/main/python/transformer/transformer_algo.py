from sock.rpc_interface import RPCInterface
from transformer.base import BaseFuzzer
from transformer.model import *
from utils import remove_lsb, load_jqf

from transformer.dataset import Dataset
from transformer.model import TransformerModel

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

remote = RPCInterface()


class TransformerFuzzer(BaseFuzzer):
    """
    This class implements a fuzzer using Transformer
    """

    def __init__(self, max_input_len=500, n_sample_candidates=10,
                 n_sample_positions=100, epochs=10, exp=6, vocab_size=100, sequence_length=20, batch_size=64,
                 embed_dim=256, latent_dim=2048, num_heads=8):
        super().__init__()
        self.exp = exp
        self.n_sample_candidates = n_sample_candidates
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_input_len = max_input_len
        self.n_sample_positions = n_sample_positions

        self.model = TransformerModel(vocab_size=vocab_size, sequence_length=sequence_length,
                                      batch_size=batch_size, embed_dim=embed_dim, latent_dim=latent_dim,
                                      num_heads=num_heads)

        # uint8, float32, samples×width
        self.batch = []
        self.events = 100
        self.new_seqs = []
        self.new_files = []
        self.val_data = Dataset()
        self.train_data = Dataset()

        self.event_bitsize = 8  # bytes
        remote.obj = self

    @remote.register("bitsize")
    def event_bitsize(self, d: int):
        self.event_bitsize = d

    @remote.register("totalevents")
    def get_total_events(self, n: int):
        logger.debug(f"total events {n}")
        self.events = n

    @remote.register("pretrain")
    def pretrain(self, seqs: List[bytes], files: List[bytes]):
        """
        :param files: List of input files as bytes, represents output to the model
        :param seqs: represents input to the model as sequences of events
        :return: None
        """
        # todo: use Dataset() to store and split train and val data
        logger.debug("splitting dataset")
        self.train_data, self.val_data = Dataset(X=np.array(seqs), y=np.array(files)).split()
        # prepare training and validation data from the data through the socket
        # Initialize model and update train and val data for training
        if not self.model.is_model_created:
            logger.debug("initializing model")
            self.model.initialize_model()
        # train NN
        logger.info("Begin training on pre-given dataset")
        self.model.train(self.train_data, self.val_data, epochs=self.epochs)
        logger.info("Finished training on pre-given dataset")

    @remote.register("geninput")
    def geninput(self) -> bytes:
        if not self.batch:
            self.batch = self.create_inputs()
        return self.batch[0]

    def create_inputs(self) -> List[bytes]:
        result: List[bytes] = []
        # 1. sample candidates
        if len(self.train_data) == 0:
            raise Exception("No training data available, cannot create inputs. Pre-train first!")
        mask = np.random.choice(len(self.train_data), self.model.batch_size, replace=False)
        for seq in self.train_data.X[mask]:
            mutated_seq = self._mutate(seq, size=10, mode="sub")
            result.append(mutated_seq)

        return result

    def _mutate(self, seqs, alpha=2, beta=1, size=5, mode="sub", seq_type: int = 1) -> bytes:
        """
        :param seqs: an event with a sequence of bytes
        :param alpha and beta: parameters to shift sampling towards left, right, up or down
        :param size: number of samples to extract
        :param mode: substitution, removal or addition of bytes
        :param seq_type:  1 for int, 2 for short and so on
        :return: a mutated sequence of bytes for a particular event
        """
        # todo: remove, add
        if mode == "sub":  # Substitute
            # sample geometric positions
            n_pos = np.random.beta(a=alpha, b=beta, size=size % len(seqs)) * len(seqs)

            # sample values (uniform)
            values = np.random.choice(max(self.events, 1 << (seq_type * self.event_bitsize)), len(n_pos), replace=False)

            # mutate
            res = bytearray(seqs)
            for pos, val in zip(n_pos, values):
                res[int(pos)] = val

            return bytes(res)

        print("incorrect mode, no mutation")
        return seqs

    @remote.register("observe_single")
    def observe_single(self, status: int, seq: bytes):
        if status != 0:
            self.new_files.append(self.batch[0])
            self.new_seqs.append(seq)
        self.batch.pop(0)
        if len(self.new_files) == self.batch_size:
            self.update(Dataset(self.new_files, self.new_seqs))
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
    # get initial zest test data
    seqs, files = load_jqf("/home/ajrox/Programs/pylibfuzzer/examples/transformer_jqf/data/fuzz-results/")
    # test pre-train
    gen.pretrain(seqs, files)
    # test geninput and populate batch

    # test observe_single

    # create
    # remote.run()
