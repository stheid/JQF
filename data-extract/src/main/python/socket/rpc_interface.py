import functools
from socket import socket, AF_UNIX, SOCK_STREAM
from struct import unpack, pack
import struct


def get(key):
    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper_repeat(*args, **kwargs):
            value = func(*args, **kwargs)
            return value

        return wrapper_repeat

    return decorator_repeat


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class RPCInterface(metaclass=Singleton):
    version = "1.0"
    instance = None

    def __init__(self, sock="/tmp/jqf.sock"):
        super().__init__()
        self.addr = sock
        self.__enter__()

    def __enter__(self):
        self.socket = socket(AF_UNIX, SOCK_STREAM).__enter__()
        self.socket.connect(self.addr)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.socket.__exit__()


sock = RPCInterface().socket
funcs = dict()


def register(name):
    def decorator(func):
        funcs[name or func.__name__] = func
        return func

    # noinspection PyTypeChecker
    if callable(name) and hasattr(name, "__name__"):
        name, f = None, name
        # noinspection PyTypeChecker
        decorator(f)
    else:
        return decorator


def process():
    func = read()
    funcs[func]()


@register("version")
def send_version():
    write(RPCInterface.version)


@register("pretrain")
def pretrain():
    pass


def write(msg):
    input_ = msg.encode("utf-8")
    sock.sendall(pack('I', len(input_)) + input_)


def read():
    len_ = unpack('I', sock.recv(4))[0]
    return sock.recv(len_).decode()


if __name__ == '__main__':
    print(funcs)
    while True:
        try:
            process()
        except struct.error:
            break
