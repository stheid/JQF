from typing import List


class BaseFuzzer:
    supported_extractors = []

    def __init__(self):
        self._initialized = False

    def _check_initialization(self):
        """
        Checks whether the fuzzer implementation is initialized.
        Must be overwritten by fuzzers that need initialization

        :return:
        """
        if not self._initialized:
            raise RuntimeError(
                'please call load_seed() before creating the first input to initialize the internal state')

    def prepare(self):
        """
        prepare runner to be able to create inputs before loading seeds.
        """
        pass

    def load_seed(self, paths: List[str]):
        """
        loads all files in the seed to the model

        :param paths: this is actually a list of strings that refers to a file
        :return:
        """
        pass

    def create_inputs(self) -> List[bytes]:
        """
        create new input from internal model

        in most implementations the first couple of inputs from this functions will be the seed files

        :return: list of files as bytes
        """
        return []

    def observe(self, fuzzing_result: List[bytes]):
        """
        gets execution results of the last input batch passed to the PUT.

        :param fuzzing_result:
        :return:
        """
        pass

    def done(self) -> bool:
        """
        returns true iff the fuzzer wants to terminate execution.

        :return:
        """
        return False

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def obs_type(self):
        # check the input type of the transformer
        return list(self.observe.__annotations__.values())[0]  # noqa
