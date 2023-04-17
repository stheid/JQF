import edu.berkeley.cs.jqf.fuzz.ei.ZestDriver
import edu.berkeley.cs.jqf.fuzz.ei.ZestGuidance
import edu.berkeley.cs.jqf.fuzz.guidance.Guidance
import edu.berkeley.cs.jqf.fuzz.guidance.Result
import edu.berkeley.cs.jqf.fuzz.junit.GuidedFuzzing
import edu.berkeley.cs.jqf.fuzz.util.Coverage
import edu.berkeley.cs.jqf.fuzz.util.CoverageFactory
import edu.berkeley.cs.jqf.instrument.tracing.events.BranchEvent
import edu.berkeley.cs.jqf.instrument.tracing.events.TraceEvent
import socket.RPCInterface
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.net.SocketException
import java.util.function.Consumer
import kotlin.system.exitProcess

class RPCGuidance(
    process: ProcessBuilder, private val warmupGuidance: Guidance?, private val warmupInputs: Long = 10_000L,
    outputDirectory: File = File("fuzz-results").apply { mkdir() }
) : Guidance {

    private var nInputs = 0L
    private val socket = RPCInterface(process = process)
    private var currwarmupFile: ByteArray? = null
    private var warmupFiles = mutableListOf<ByteArray>()
    private var warmupSeqs = mutableListOf<ByteArray>()
    private var events = mutableListOf<Int>()
    private var totalCoverage = CoverageFactory.newInstance()
    private val totalCovFile =  File("$outputDirectory/total_coverage.csv").apply { bufferedWriter().write("") }

    // if there is no warmup guidance we assume it is already warmed up
    private val warmupRequired: Boolean
        get() = warmupGuidance != null && nInputs < warmupInputs

    override fun getInput(): InputStream {
        // pull data from warmup guidance or from the socket
        val arr = if (warmupRequired) (warmupGuidance!!.input).run {
            val bytes = mutableListOf<Int>()
            while (true) {
                try {
                    bytes.add(this.read())
                } catch (_: IllegalStateException) {
                    break
                }
            }

            bytes.map { it.toByte() }.toByteArray()
        }.also {
            // @FuzzStatement.evaluate() might sometimes eat away inputs because of AssumptionViolationExceptions
            //     which will not trigger the handleResult.
            // Therefore, we need to override the current file in that case and only store it inside the handleResult
            currwarmupFile = it
        }
        else
            socket.get("geninput")
        return PaddedByteArrayInputStream(arr)
        //ByteArrayInputStream(arr)
    }

    override fun hasInput() = true

    override fun handleResult(result: Result?, error: Throwable?) {
        nInputs += 1
        when {
            (nInputs % 10000 == 0L) -> print("${nInputs / 1000}k")
            (nInputs % 5000 == 0L) -> print(",")
            (nInputs % 1000 == 0L) -> print('.')
        }

        // TODO convert it to bytes properly.
        val eventseq = events.map { it.toByte() }.toByteArray()
        events.clear()

        if (warmupGuidance != null && nInputs <= warmupInputs) {
            if (nInputs<warmupInputs){
                // store event
                warmupFiles.add(currwarmupFile!!)
                warmupSeqs.add(eventseq)
            }
            else {
                socket.post("bitsize", 8)
                socket.post("totalevents", totalCoverage.counter.size())
                socket.post("pretrain", warmupSeqs.apply { add(eventseq) }, warmupFiles.apply { add(currwarmupFile!!) })
                warmupFiles.clear()
                warmupSeqs.clear()
            }
        }else
        // send only result to clients observe method
            socket.observe((result != Result.SUCCESS).toInt(), eventseq)

        FileOutputStream(totalCovFile, true).bufferedWriter().use{ out ->
            out.write((totalCoverage.nonZeroCount * 100.0 / totalCoverage.size()).toString())
            out.newLine()
        }
    }

    override fun generateCallBack(thread: Thread?): Consumer<TraceEvent> {
        return Consumer { e -> handleEvent(e) }
    }

    private fun handleEvent(e: TraceEvent?) {
        if (e is BranchEvent) events.add(e.iid)
        // Collect totalCoverage
        (totalCoverage as Coverage).handleEvent(e)
    }
}

private fun Boolean.toInt() = if (this) 1 else 0


private fun RPCInterface.observe(int: Int, eventseq: ByteArray) {
    try {
        write("observe")
        write(int)
        write(eventseq)
    } catch (e: SocketException) {
        error("Socket connection failed while calling observe")
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
    val warmUpOutputDirectory = File("fuzz-results-warmup")

    try {
        // Load the guidance
        val title = "$testClassName#$testMethodName"
        val warmupGuidance = ZestGuidance(title, null, warmUpOutputDirectory)
        val dir = File("data-extract/src/main/python")
        val rpcGuidance = RPCGuidance(
            ProcessBuilder(pythoninter, "transformer/transformer_algo.py").directory(dir).inheritIO()
                .apply { environment()["PYTHONPATH"] = dir.absolutePath }, warmupGuidance=warmupGuidance, warmupInputs = 10_000L
        )

        // Run the Junit test
        val res = GuidedFuzzing.run(testClassName, testMethodName, rpcGuidance, System.out)
    } catch (e: Exception) {
        e.printStackTrace()
        exitProcess(2)
    }
}