import edu.berkeley.cs.jqf.fuzz.guidance.Guidance
import edu.berkeley.cs.jqf.fuzz.guidance.Result
import edu.berkeley.cs.jqf.instrument.tracing.events.TraceEvent
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import java.util.function.Consumer

class SeededGuidance(seedDir: File) : Guidance {
    private val files = seedDir.listFiles()?.sorted()?.toMutableList() ?: mutableListOf()
    override fun getInput(): InputStream = FileInputStream(files.removeFirst())

    override fun hasInput(): Boolean = files.isNotEmpty()

    override fun handleResult(result: Result?, error: Throwable?) = Unit

    override fun generateCallBack(thread: Thread?): Consumer<TraceEvent> = Consumer {}
}