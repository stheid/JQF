package socket.client

import java.net.StandardProtocolFamily
import java.net.UnixDomainSocketAddress
import java.nio.ByteBuffer
import java.nio.channels.SocketChannel
import java.nio.file.Path

fun main() {
    val file = Path.of(System.getProperty("user.home")).resolve("server.socket")
    val address = UnixDomainSocketAddress.of(file)
    val channel = SocketChannel.open(StandardProtocolFamily.UNIX)
    channel.connect(address)
    Thread.sleep(3000)
    writeMessageToSocket(channel, "Hello")
    Thread.sleep(1000)
    writeMessageToSocket(channel, "UNIX domain sockets")
}

private fun writeMessageToSocket(socketChannel: SocketChannel, message: String) {
    val buffer = ByteBuffer.allocate(1024)
    buffer.clear()
    buffer.put(message.toByteArray(Charsets.UTF_8))
    buffer.flip()
    while (buffer.hasRemaining()) {
        socketChannel.write(buffer)
    }
}