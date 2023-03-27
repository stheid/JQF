import edu.berkeley.cs.jqf.fuzz.guidance.Guidance
import edu.berkeley.cs.jqf.fuzz.guidance.Result
import edu.berkeley.cs.jqf.instrument.tracing.events.TraceEvent
import java.io.File
import java.io.InputStream
import java.time.Duration
import java.util.function.Consumer

class RPCGuidance(
    testName: String?,
    duration: Duration?,
    outputDirectory: File?,
    val warmupGuidance: Guidance?,
    val warmupInputs: Int = 10_000
) : Guidance {

    var nInputs = 0

    // if there is no warmstart guidance we assume it is already warmed up
    private val warmupRequired: Boolean
        get() = warmupGuidance != null && nInputs < warmupInputs

    init {
        // setup socket as a server
        // start client
    }

    override fun getInput(): InputStream {
        if (warmupRequired) return warmupGuidance!!.input
        TODO()
    }

    override fun hasInput(): Boolean {
        TODO("Not yet implemented")
    }

    override fun handleResult(result: Result?, error: Throwable?) {
        if (warmupGuidance != null) when (nInputs){
            warmupInputs -> TODO("send dataset to prepare method of client")
            else -> TODO("collect data for the client")
        }
        TODO("send only result to clients observe method")
    }

    override fun generateCallBack(thread: Thread?): Consumer<TraceEvent> {
        return Consumer { e -> handleEvent(e) }
    }

    private fun handleEvent(e: TraceEvent?) {
    }
}






fun main(args: Array<String>) {
}
