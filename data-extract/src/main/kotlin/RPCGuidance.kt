import edu.berkeley.cs.jqf.fuzz.ei.ZestDriver
import edu.berkeley.cs.jqf.fuzz.ei.ZestGuidance
import edu.berkeley.cs.jqf.fuzz.guidance.Guidance
import edu.berkeley.cs.jqf.fuzz.guidance.Result
import edu.berkeley.cs.jqf.fuzz.junit.GuidedFuzzing
import edu.berkeley.cs.jqf.fuzz.util.Coverage
import edu.berkeley.cs.jqf.fuzz.util.CoverageFactory
import edu.berkeley.cs.jqf.instrument.tracing.FastCoverageSnoop
import edu.berkeley.cs.jqf.instrument.tracing.events.BranchEvent
import edu.berkeley.cs.jqf.instrument.tracing.events.TraceEvent
import janala.instrument.FastCoverageListener
import socket.RPCInterface
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.net.SocketException
import java.util.function.Consumer
import kotlin.system.exitProcess

class RPCGuidance(
    process: ProcessBuilder, private val warmupGuidance: Guidance?, private val warmupInputs: Int = 10_000,
    outputDirectory: File = File("fuzz-results").apply { mkdir() }
) : Guidance {

    private var nInputs = 0
    private val socket = RPCInterface(process = process)
    private var currwarmupFile: ByteArray? = null
    private var warmupFiles = mutableListOf<ByteArray>()
    private var warmupSeqs = mutableListOf<ByteArray>()
    private var events = mutableListOf<Int>()
    private var totalCoverage = CoverageFactory.newInstance()
    private val totalCovFile = File(outputDirectory, "total_coverage.csv").apply { bufferedWriter().write("") }

    init {
        if (totalCoverage is FastCoverageListener)
            FastCoverageSnoop.setFastCoverageListener(totalCoverage as FastCoverageListener?)
    }

    override fun getInput(): InputStream {
        // pull data from warmup guidance or from the socket
        val arr =
            if (warmupGuidance != null && nInputs < warmupInputs && warmupGuidance.hasInput())
                warmupGuidance.input.run {
                    val bytes = mutableListOf<Int>()
                    if (this is FileInputStream)
                        readAllBytes()
                    else {
                        while (true) {
                            try {
                                bytes.add(read())
                            } catch (_: IllegalStateException) {
                                break
                            }
                        }
                        bytes.map { it.toByte() }.toByteArray()
                    }
                }.also {
                    // @FuzzStatement.evaluate() might sometimes eat away inputs because of AssumptionViolationExceptions
                    //     which will not trigger the handleResult.
                    // Therefore, we need to override the current file in that case and only store it inside the handleResult
                    currwarmupFile = it
                }
            else
                socket.get("geninput")
        return PaddedByteArrayInputStream(arr)
    }

    override fun hasInput() = true

    override fun handleResult(result: Result?, error: Throwable?) {
        nInputs += 1
        when {
            (nInputs % 10000 == 0) -> print("${nInputs / 1000}k")
            (nInputs % 5000 == 0) -> print(",")
            (nInputs % 1000 == 0) -> print('.')
        }

        // TODO convert it to bytes properly.
        val eventseq = events.map { it.toByte() }.toByteArray()
        events.clear()

        if (warmupGuidance != null && nInputs <= warmupInputs) {
            warmupGuidance.handleResult(result, error)

            // store event and file
            warmupFiles.add(currwarmupFile!!)
            warmupSeqs.add(eventseq)

            if (nInputs == warmupInputs || !warmupGuidance.hasInput()) {
                // if warmup finished
                socket.post("bitsize", 8)
                socket.post("totalevents", totalCoverage.counter.size())
                socket.post("pretrain", warmupSeqs, warmupFiles)
                warmupFiles.clear()
                warmupSeqs.clear()
            }
        } else
        // send only result to clients observe method
            socket.observe((result != Result.SUCCESS).toInt(), eventseq)

        FileOutputStream(totalCovFile, true).bufferedWriter().use { out ->
            out.write((totalCoverage.nonZeroCount * 100.0 / totalCoverage.size()).toString())
            out.newLine()
        }
    }

    override fun generateCallBack(thread: Thread?): Consumer<TraceEvent> {
        // todo: disconnect callback after warmup is done. (fix this later: creates memoryleak)
        val warmupCallback = warmupGuidance?.generateCallBack(thread)
        val callback = Consumer<TraceEvent> { e -> handleEvent(e) }
        return warmupCallback?.andThen(callback) ?: callback
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

    try {
        // create dataset
        // TODO: doing two consecutive GuidedFuzzing.run executions will cause the the coverage events to be triggered only on the first run. hence measurement breaks if we execute this guidance before
        // TODO: This is probably related to instrumentation of some sort and i have currently no easy way to prevent it.
        //val guidance =
        //    MeasureZest("$testClassName#$testMethodName", null, File("fuzz-results-runner"), 20_000, dumpAll = true)
        //GuidedFuzzing.run(testClassName, testMethodName, guidance, System.out)

        // create Warmup Guidance that reads the generated dataset
        val warmupGuidance = SeededGuidance(File("fuzz-results-runner/corpus"))

        val dir = File("data-extract/src/main/python")
        val rpcGuidance = RPCGuidance(
            ProcessBuilder(pythoninter, "transformer/transformer_algo.py").directory(dir).inheritIO()
                .apply { environment()["PYTHONPATH"] = dir.absolutePath },
            warmupGuidance = warmupGuidance,
            warmupInputs = 10_000
        )

        // Run the Junit test
        val res = GuidedFuzzing.run(testClassName, testMethodName, rpcGuidance, System.out)
    } catch (e: Exception) {
        e.printStackTrace()
        exitProcess(2)
    }
}