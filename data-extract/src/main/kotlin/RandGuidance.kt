import edu.berkeley.cs.jqf.fuzz.guidance.Guidance
import edu.berkeley.cs.jqf.fuzz.guidance.Result
import edu.berkeley.cs.jqf.instrument.tracing.events.TraceEvent
import java.io.ByteArrayInputStream
import java.io.InputStream
import java.util.function.Consumer
import kotlin.random.Random

class RandGuidance() : Guidance {
    override fun getInput(): InputStream = ByteArrayInputStream(Random.Default.nextBytes(100))

    override fun hasInput(): Boolean = true

    override fun handleResult(result: Result?, error: Throwable?) = Unit

    override fun generateCallBack(thread: Thread?): Consumer<TraceEvent> = Consumer {}
}