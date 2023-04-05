import functools
import logging
import struct
from inspect import getfullargspec
from struct import unpack, pack
from typing import List

from socket import socket, AF_UNIX, SOCK_STREAM

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
        self.isconnected = False

    def register(self, name, self_=None):
        def decorator(func):
            logger.debug(f"registering {name}")

            @functools.wraps(func)
            def wrapper(self_, *args):
                logger.debug(f"preparing {name}")
                if not self.isconnected:
                    # for debugging purposes, this will execute locally if no connection has been established.
                    logger.warning(f"Calling {name} locally")
                    return func(self_, *args)
                params = []
                spec = getfullargspec(func)
                for arg in spec.args:
                    if arg == "self":
                        continue
                    if spec.annotations[arg] == int:
                        params.append(self.read_int())
                    elif spec.annotations[arg] == bytes:
                        params.append(self.read())
                    elif spec.annotations[arg] == List[bytes]:
                        # get length
                        len_ = self.read_int()
                        # get data
                        params.append([self.read() for _ in range(len_)])
                    else:
                        raise ValueError(f"could not parse parameters of {name}")

                logger.debug(f"calling {name}")
                res = func(self_, *params)
                logger.debug(f"retrieving results of {name}")
                if res is not None:
                    self.write(res)
                logger.debug(f"finished {name}")
                return res

            self.funcs[name] = wrapper
            return wrapper

        return decorator

    def run(self):
        with socket(AF_UNIX, SOCK_STREAM).__enter__() as self.socket:
            self.socket.connect(self.addr)
            self.isconnected = True
            while True:
                try:
                    func = self.read().decode()
                    self.funcs[func](self.obj)
                except struct.error:
                    break
                except KeyError as e:
                    print("The function", e.args[0], "does not exist")
        self.isconnected = False

    def write(self, msg):
        input_ = msg.encode("utf-8")
        self.socket.sendall(pack('I', len(input_)) + input_)

    def read_int(self):
        return unpack('I', self.socket.recv(4))[0]

    def read(self):
        return self.socket.recv(self.read_int())
