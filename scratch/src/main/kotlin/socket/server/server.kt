package socket.server

import java.net.StandardProtocolFamily
import java.net.UnixDomainSocketAddress
import java.nio.ByteBuffer
import java.nio.channels.ServerSocketChannel
import java.nio.channels.SocketChannel
import java.nio.file.Files
import java.nio.file.Path
import java.util.*

fun main() {
    val socketFile = Path.of(System.getProperty("user.home")).resolve("server.socket")
    Files.deleteIfExists(socketFile)
    val address = UnixDomainSocketAddress.of(socketFile)
    val serverChannel = ServerSocketChannel.open(StandardProtocolFamily.UNIX)
    serverChannel.bind(address)
    println("[INFO] Waiting for client to connect...")
    val channel = serverChannel.accept()
    println("[INFO] Client connected - waiting for client to send messages")
    readAndPrintMessages(channel)
}

private fun readAndPrintMessages(channel: SocketChannel) {
    while (true) {
        readMessageFromSocket(channel).ifPresent { x: String? -> println(x) }
        Thread.sleep(100)
    }
}

private fun readMessageFromSocket(channel: SocketChannel): Optional<String> {
    val buffer = ByteBuffer.allocate(1024)
    val bytesRead = channel.read(buffer)
    if (bytesRead < 0) return Optional.empty()
    val bytes = ByteArray(bytesRead)
    buffer.flip()
    buffer[bytes]
    val message = bytes.toString(Charsets.UTF_8)
    return Optional.of(message)
}
