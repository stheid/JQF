from socket import socket, AF_UNIX, SOCK_STREAM
from struct import unpack, pack

import click


@click.command()
@click.option('--addr', default='/tmp/test.sock', help='socket address')
def main(addr):
    with socket(AF_UNIX, SOCK_STREAM) as sock:
        sock.connect(addr)

        for input_ in ["hello"]:
            input_ = input_.encode("utf-8")
            sock.sendall(pack('I', len(input_)) + input_)
            print('SENT>> ' + input_.decode())
            len_ = unpack('I', sock.recv(4))[0]
            result = sock.recv(len_)
            print('RECV>> ' + str(result))


if __name__ == '__main__':
    main()
