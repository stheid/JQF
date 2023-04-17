import java.io.ByteArrayInputStream
import java.nio.ByteBuffer

class PaddedByteArrayInputStream(array: ByteArray) : ByteArrayInputStream(array) {
    val byteBuffer = ByteBuffer.allocate(4)

    override fun read(bPointer: ByteArray, off: Int, len: Int): Int {
        // Zero out the byte buffer before reading from the source
        byteBuffer.putInt(0, 0)
        super.read(byteBuffer.array(), off, len)
        // copy padded result into out pointer
        System.arraycopy(byteBuffer.array(), 0, bPointer, 0, byteBuffer.array().size)
        return len
    }
}