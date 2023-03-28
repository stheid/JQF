package socket

import java.net.StandardProtocolFamily
import java.net.UnixDomainSocketAddress
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.ServerSocketChannel
import java.nio.channels.SocketChannel
import java.nio.file.Files
import java.nio.file.Path

class RPCInterface(sock: String = "/tmp/jqf.sock") {
    private val version = "0.2"

    private var channel: SocketChannel

    init {
        val socketFile = Path.of(sock)
        Files.deleteIfExists(socketFile)

        ServerSocketChannel.open(StandardProtocolFamily.UNIX).apply {
            bind(UnixDomainSocketAddress.of(socketFile))
            println("bound socket")
            configureBlocking(true)
            channel = accept()
            println("accepted connection")
        }

        getVersion().also { if (it != version) error("VERSION MISSMATCH: client version ($it) does not match server version ($version)") }
    }

    fun getVersion() = get("version")
    fun prepare(data: List<ByteArray>) = post("pretrain", data)

    private fun get(key: String): String {
        write(key)
        val res = read()
        println("$key -> $res")
        return res
    }

    private fun post(key: String, data: List<ByteArray>) {
        write(key)
        writeInt(data.size)
        data.forEach {
            writeByteArray(it)
        }
    }


    private fun writeInt(int: Int) {
        val buffer = ByteBuffer.allocate(4)
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(int)
        buffer.rewind()
        channel.write(buffer)
    }

    private fun writeByteArray(payload: ByteArray) {
        writeInt(payload.size)
        val buffer = ByteBuffer.allocate(payload.size)
        buffer.put(payload)
        buffer.rewind()
        channel.write(buffer)
    }

    private fun write(key: String) {
        writeByteArray(key.toByteArray(Charsets.UTF_8))
    }


    private fun readInt(): Int {
        val buffer = ByteBuffer.allocate(4)
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        channel.read(buffer)
        return buffer.rewind().int
    }

    private fun read(): String {
        val buffer = ByteBuffer.allocate(readInt())
        val bytes = ByteArray(channel.read(buffer))
        buffer.flip()[bytes]
        return bytes.toString(Charsets.UTF_8)
    }

}

fun main() {
    val remote = RPCInterface()
    remote.prepare(listOf("ich","bin","ein","keks").map { it.toByteArray() })
}