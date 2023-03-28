package socket

import java.net.StandardProtocolFamily
import java.net.UnixDomainSocketAddress
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.ServerSocketChannel
import java.nio.channels.SocketChannel
import java.nio.file.Files
import java.nio.file.Path

fun main() {
    val socketFile = Path.of("/tmp/test.sock")
    Files.deleteIfExists(socketFile)
    val address = UnixDomainSocketAddress.of(socketFile)
    val serverChannel = ServerSocketChannel.open(StandardProtocolFamily.UNIX)
    serverChannel.bind(address)
    serverChannel.configureBlocking(true)
    println("[INFO] Waiting for client to connect...")
    val channel = serverChannel.accept()
    println("[INFO] Client connected - waiting for client to send messages")
    while (true)
        println(read(channel))
}

private fun readInt(channel: SocketChannel): Int {
    val buffer = ByteBuffer.allocate(4)
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    channel.read(buffer)
    return buffer.rewind().int
}

private fun read(channel: SocketChannel): String {
    val buffer = ByteBuffer.allocate(readInt(channel))
    val bytes = ByteArray(channel.read(buffer))
    buffer.flip()[bytes]
    return bytes.toString(Charsets.UTF_8)
}
