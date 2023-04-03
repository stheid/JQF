import functools
from socket import socket, AF_UNIX, SOCK_STREAM
from struct import unpack, pack
import struct
import logging
from inspect import getfullargspec

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RPCInterface:
    instance = None

    def __init__(self, sock="/tmp/jqf.sock"):
        super().__init__()
        self.addr = sock
        self.funcs = dict()
        self.obj = None

    def register(self, name, self_=None):
        def decorator(func):
            logger.debug(f"registering {name}")

            @functools.wraps(func)
            def wrapper(self_, *args):
                logger.debug(f"calling {name}")
                if args:
                    return func(*args)
                params = []
                spec = getfullargspec(func)
                for arg in spec.args:
                    if arg == "self":
                        continue
                    if spec.annotations[arg] == int:
                        params.append(self.read_int())
                    else:
                        # get length
                        len_ = self.read_int()
                        # get data
                        params.append([self.read() for _ in range(len_)])
                res = func(self_, *params)
                if res is not None:
                    self.write(res)
                return res

            self.funcs[name] = wrapper
            return wrapper
        return decorator

    def run(self):
        with socket(AF_UNIX, SOCK_STREAM).__enter__() as self.socket:
            self.socket.connect(self.addr)
            while True:
                try:
                    func = self.read().decode()
                    self.funcs[func](self.obj)
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
