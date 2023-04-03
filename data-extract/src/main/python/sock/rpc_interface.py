import functools
from socket import socket, AF_UNIX, SOCK_STREAM
from struct import unpack, pack
import struct
from inspect import getfullargspec


class RPCInterface:
    instance = None

    def __init__(self, sock="/tmp/jqf.sock"):
        super().__init__()
        self.addr = sock
        self.funcs = dict()

    def register(self, name):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args):
                if args:
                    return func(*args)
                params = []
                for _ in getfullargspec(func).args:
                    # get length
                    len_ = self.read_int()
                    # get data
                    params.append([self.read() for _ in range(len_)])
                res = func(*params)
                if res is not None:
                    self.write(res)

            self.funcs[name or func.__name__] = wrapper
            return wrapper

        # noinspection PyTypeChecker
        if callable(name) and hasattr(name, "__name__"):
            name, f = None, name
            # noinspection PyTypeChecker
            decorator(f)
        else:
            return decorator

    def run(self):
        with socket(AF_UNIX, SOCK_STREAM).__enter__() as self.socket:
            self.socket.connect(self.addr)
            while True:
                try:
                    func = self.read().decode()
                    print("calling", func)
                    self.funcs[func]()
                except struct.error:
                    break
                except KeyError as e:
                    print("The function", e.args[0], "does not exist")

    def write(self, msg):
        input_ = msg.encode("utf-8")
        self.socket.sendall(pack('I', len(input_)) + input_)

    def read_int(self):
        return unpack('I', self.socket.recv(4))[0]

    def read(self):
        return self.socket.recv(self.read_int())
