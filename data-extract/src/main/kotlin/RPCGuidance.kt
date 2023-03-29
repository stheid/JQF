import edu.berkeley.cs.jqf.fuzz.guidance.Guidance
import edu.berkeley.cs.jqf.fuzz.guidance.Result
import edu.berkeley.cs.jqf.fuzz.util.CoverageFactory
import edu.berkeley.cs.jqf.instrument.tracing.events.BranchEvent
import edu.berkeley.cs.jqf.instrument.tracing.events.TraceEvent
import socket.RPCInterface
import java.io.ByteArrayInputStream
import java.io.InputStream
import java.util.function.Consumer

class RPCGuidance(
    testName: String?,
    val warmupGuidance: Guidance?,
    val warmupInputs: Int = 10_000
) : Guidance {

    private var nInputs = 0
    // TODO call python process correctly
    private val socket = RPCInterface(process = ProcessBuilder("python "))
    private var warmupFiles = mutableListOf<ByteArray>()
    private var warmupSeqs = mutableListOf<ByteArray>()
    private var events = mutableListOf<Int>()
    private var totalCoverage = CoverageFactory.newInstance()

    // if there is no warmup guidance we assume it is already warmed up
    private val warmupRequired: Boolean
        get() = warmupGuidance != null && nInputs < warmupInputs

    override fun getInput(): InputStream{
        // pull data from warmup guidance or from the socket
        val arr = if (warmupRequired)
            warmupGuidance!!.input.readAllBytes().also { warmupFiles.add(it) }
        else
            // TODO deal properly with batches
            socket.get("geninput")

        return ByteArrayInputStream(arr)
    }

    override fun hasInput() = true

    override fun handleResult(result: Result?, error: Throwable?) {
        // TODO: calculate eventseq
        val eventseq = byteArrayOf()

        if (warmupGuidance != null) when (nInputs){
            warmupInputs ->{
                // TODO create list of all events
                socket.postInt("totalevents", totalCoverage.counter.size())
                socket.post("pretrain", warmupFiles, warmupSeqs.apply { add(eventseq) })
                warmupFiles.clear()
                warmupSeqs.clear()
            }
            else -> {
                // store event
                warmupSeqs.add(eventseq)
            }
        }
        // send only result to clients observe method
        socket.post("observe", listOf(eventseq))
    }

    override fun generateCallBack(thread: Thread?): Consumer<TraceEvent> {
        return Consumer { e -> handleEvent(e) }
    }

    private fun handleEvent(e: TraceEvent?) {
        if (e is BranchEvent)
            events.add(e.iid)
    }
}