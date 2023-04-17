import edu.berkeley.cs.jqf.fuzz.ei.ZestDriver
import edu.berkeley.cs.jqf.fuzz.ei.ZestGuidance
import edu.berkeley.cs.jqf.fuzz.guidance.Result
import edu.berkeley.cs.jqf.fuzz.junit.GuidedFuzzing
import edu.berkeley.cs.jqf.instrument.tracing.events.BranchEvent
import edu.berkeley.cs.jqf.instrument.tracing.events.TraceEvent
import org.eclipse.collections.impl.set.mutable.primitive.IntHashSet
import java.io.*
import java.time.Duration
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.system.exitProcess


class MeasureZest(testName: String?, duration: Duration?, outputDirectory: File?) :
    ZestGuidance(testName, duration, outputDirectory) {
    private var events = mutableListOf<Int>()
    private val eventsFile = File(outputDirectory, "events.bin").apply { bufferedWriter().write("") }
    private val idsFile = File(outputDirectory, "ids.csv").apply { bufferedWriter().write("") }
    private val totalCovFile = File(outputDirectory, "total_coverage.csv").apply { bufferedWriter().write("") }
    private val executor = Executors.newSingleThreadExecutor()
    private val iidMap = mutableMapOf<Int, Int>()

    override fun displayStats(force: kotlin.Boolean) {
//        super.displayStats(force)
    }

    override fun handleEvent(e: TraceEvent?) {
        if (e is BranchEvent) {
            events.add(iidMap.getOrPut(e.iid) {
                let {
                    val newID = iidMap.size

                    // for each new ID print the whole mapping to a file
                    FileOutputStream(idsFile, true).bufferedWriter().apply {
                        write(
                            listOf(
                                newID,
                                e.containingClass,
                                e.containingMethodName,
                                e.containingMethodDesc,
                                e.lineNumber
                            ).joinToString("\t")
                        )
                        newLine()
                        flush()
                    }

                    newID
                }
            })
        }
        super.handleEvent(e)
    }

    override fun handleResult(result: Result?, error: Throwable?) {
        val maxAllowedInputs = 500000
        val eventList = events.toList()
        executor.execute {
            DataOutputStream(BufferedOutputStream(FileOutputStream(eventsFile, true))).also { dst ->
                dst.writeInt(eventList.size)
                eventList.forEach(dst::writeShort)
                dst.flush()
            }
            FileOutputStream(totalCovFile, true).bufferedWriter().use{ out ->
                val tc = getTotalCoverage()
                out.write((tc.nonZeroCount * 100.0 / tc.size()).toString())
                out.newLine()
            }
        }

        events = mutableListOf()
        super.handleResult(result, error)

        when {
            (numTrials % 10000 == 0L) -> print(numTrials)
            (numTrials % 5000 == 0L) -> print(",")
            (numTrials % 1000 == 0L) -> print('.')
        }

        if (numTrials >= maxAllowedInputs) {
            executor.shutdown()
            executor.awaitTermination(1000, TimeUnit.MINUTES)
            println()
            println("Total number of IDs ${iidMap.size}")
            val duration = (Date().time - startTime.time) / 1000.0
            println("Time consumed ${"%.2f".format(duration)}s, ${"%.2f".format(numTrials/duration)} inputs/s")
            if (iidMap.size.toShort().toInt() != iidMap.size)
                error("To many IDs, change code back to Integer ids")
            exitProcess(0)
        }
    }

//    override fun checkSavingCriteriaSatisfied(result: Result?): List<String> {
//        return super.checkSavingCriteriaSatisfied(result).apply { add("dumpall") }
//    }

    @Throws(IOException::class)
    override fun saveCurrentInput(responsibilities: IntHashSet, why: String?) {
        super.saveCurrentInput(responsibilities, why)
        if (why == "dumpall") {
            // revert saving the input if it has only been added because of dumpall
            savedInputs.remove(currentInput)
            savedInputs[currentParentInputIdx].dec_offspring()
        }
    }
}


fun main(args: Array<String>) {
    // wait for the JVM to be completely ready with javaagent and so on
    Thread.sleep(1_000)

    if (args.size < 2) {
        System.err.println("Usage: java " + ZestDriver::class.java + "TEST_CLASS TEST_METHOD")
        exitProcess(1)
    }

    val testClassName = args[0]
    val testMethodName = args[1]
    val outputDirectory = File("fuzz-results-runner")

    try {
        // Load the guidance
        val title = "$testClassName#$testMethodName"
        val guidance = MeasureZest(title, null, outputDirectory)

        // Run the Junit test
        val res = GuidedFuzzing.run(testClassName, testMethodName, guidance, System.out)
    } catch (e: Exception) {
        e.printStackTrace()
        exitProcess(2)
    }
}