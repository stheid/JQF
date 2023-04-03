import edu.berkeley.cs.jqf.fuzz.ei.ZestDriver
import edu.berkeley.cs.jqf.fuzz.ei.ZestGuidance
import edu.berkeley.cs.jqf.fuzz.guidance.Guidance
import edu.berkeley.cs.jqf.fuzz.guidance.Result
import edu.berkeley.cs.jqf.fuzz.junit.GuidedFuzzing
import edu.berkeley.cs.jqf.fuzz.util.CoverageFactory
import edu.berkeley.cs.jqf.instrument.tracing.events.BranchEvent
import edu.berkeley.cs.jqf.instrument.tracing.events.TraceEvent
import socket.RPCInterface
import java.io.ByteArrayInputStream
import java.io.File
import java.io.InputStream
import java.util.function.Consumer
import kotlin.system.exitProcess

class RPCGuidance(
    process: ProcessBuilder,
    private val warmupGuidance: Guidance?,
    private val warmupInputs: Long = 1_00L
) : Guidance {

    private var nInputs = 0L

    // TODO call python process correctly
    private val socket = RPCInterface(process = process)
    private var warmupFiles = mutableListOf<ByteArray>()
    private var warmupSeqs = mutableListOf<ByteArray>()
    private var events = mutableListOf<Int>()
    private var totalCoverage = CoverageFactory.newInstance()

    // if there is no warmup guidance we assume it is already warmed up
    private val warmupRequired: Boolean
        get() = warmupGuidance != null && nInputs < warmupInputs

    override fun getInput(): InputStream {
        // pull data from warmup guidance or from the socket
        val arr = if (warmupRequired)
            (warmupGuidance!!.input).run {
                val bytes = mutableListOf<Int>()
                while (true) {
                    try {
                        bytes.add(this.read())
                    } catch (_: IllegalStateException) {
                        break
                    }
                }

                bytes.map { it.toByte() }.toByteArray()
            }.also { warmupFiles.add(it) }
        else
        // TODO deal properly with batches
            socket.get("geninput")

        return ByteArrayInputStream(arr)
    }

    override fun hasInput() = true

    override fun handleResult(result: Result?, error: Throwable?) {
        nInputs += 1
        when {
            (nInputs % 10000 == 0L) -> print("${nInputs / 1000}k")
            (nInputs % 5000 == 0L) -> print(",")
            (nInputs % 1000 == 0L) -> print('.')
        }

        // TODO: calculate eventseq
        val eventseq = byteArrayOf()

        if (warmupGuidance != null) when (nInputs) {
            warmupInputs -> {
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
        if (!warmupRequired)
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

fun main(args: Array<String>) {
    // wait for the JVM to be completely ready with javaagent and so on
    Thread.sleep(1_000)

    if (args.size < 2) {
        System.err.println("Usage: java " + ZestDriver::class.java + "TEST_CLASS TEST_METHOD")
        exitProcess(1)
    }
    val pythoninter = System.getenv("PYTHONINTER") ?: "python"
    val testClassName = args[0]
    val testMethodName = args[1]
    val outputDirectory = File("fuzz-results")

    try {
        // Load the guidance
        val title = "$testClassName#$testMethodName"
        val guidance = ZestGuidance(title, null, outputDirectory)
        val dir = File("data-extract/src/main/python")
        val rpcGuidance = RPCGuidance(
            ProcessBuilder(pythoninter, "transformer/transformer_algo.py")
                .directory(dir).inheritIO().apply { environment()["PYTHONPATH"] = dir.absolutePath}, guidance
        )

        // Run the Junit test
        val res = GuidedFuzzing.run(testClassName, testMethodName, rpcGuidance, System.out)
    } catch (e: Exception) {
        e.printStackTrace()
        exitProcess(2)
    }
}