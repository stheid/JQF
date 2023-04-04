package socket

import java.net.StandardProtocolFamily
import java.net.UnixDomainSocketAddress
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.ServerSocketChannel
import java.nio.channels.SocketChannel
import java.nio.file.Files
import java.nio.file.Path

class RPCInterface(sock: String = "/tmp/jqf.sock", process: ProcessBuilder) {
    private var channel: SocketChannel

    init {
        val socketFile = Path.of(sock)
        Files.deleteIfExists(socketFile)

        ServerSocketChannel.open(StandardProtocolFamily.UNIX).apply {
            bind(UnixDomainSocketAddress.of(socketFile))
            println("Created socket and accepting connections")
            configureBlocking(true)

            // start python client
            process.start()
            println("started client")

            channel = accept()
            println("accepted connection")
        }
    }

    fun get(key: String): ByteArray {
        write(key)
        val res = readByteArray()
        println("$key -> $res")
        return res
    }

    fun post(key: String, data: ByteArray) {
        write(key)
        write(data)
    }

    fun post(key: String, vararg data: List<ByteArray>) {
        write(key)
        data.forEach { write(it) }
    }

    fun post(key: String, int: Int) {
        write(key)
        write(int)
    }

    private fun write(int: Int) {
        val buffer = ByteBuffer.allocate(4)
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(int)
        buffer.rewind()
        channel.write(buffer)
    }

    private fun write(payload: List<ByteArray>) {
        write(payload.size)
        payload.forEach { write(it)}
    }

    private fun write(payload: ByteArray) {
        write(payload.size)
        val buffer = ByteBuffer.allocate(payload.size)
        buffer.put(payload)
        buffer.rewind()
        channel.write(buffer)
    }

    private fun write(key: String) = write(key.toByteArray(Charsets.UTF_8))

    private fun readInt(): Int {
        val buffer = ByteBuffer.allocate(4)
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        channel.read(buffer)
        return buffer.rewind().int
    }

    private fun readByteArray(): ByteArray {
        val buffer = ByteBuffer.allocate(readInt())
        val bytes = ByteArray(channel.read(buffer))
        buffer.flip()[bytes]
        return bytes
    }
}