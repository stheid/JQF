import edu.berkeley.cs.jqf.fuzz.ei.ZestDriver
import edu.berkeley.cs.jqf.fuzz.ei.ZestGuidance
import edu.berkeley.cs.jqf.fuzz.junit.GuidedFuzzing
import edu.berkeley.cs.jqf.instrument.tracing.events.TraceEvent
import java.io.File
import java.lang.Boolean
import java.time.Duration
import kotlin.Array
import kotlin.String
import kotlin.system.exitProcess


class MeasureZest(testName: String?, duration: Duration?, outputDirectory: File?) :
    ZestGuidance(testName, duration, outputDirectory) {

    override fun handleEvent(e: TraceEvent?) {
        super.handleEvent(e)
    }

//    override fun checkSavingCriteriaSatisfied(result: Result?): List<String> {
//        return listOf("dumpall")
//    }
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
    val outputDirectory = File("fuzz-results")

    try {
        // Load the guidance
        val title = "$testClassName#$testMethodName"
        val guidance = MeasureZest(title, null, outputDirectory)

        // Run the Junit test
        val res = GuidedFuzzing.run(testClassName, testMethodName, guidance, System.out)
        if (Boolean.getBoolean("jqf.logCoverage")) {
            println(
                String.format(
                    "Covered %d edges.",
                    guidance.totalCoverage.nonZeroCount
                )
            )
        }
        if (Boolean.getBoolean("jqf.ei.EXIT_ON_CRASH") && !res.wasSuccessful()) {
            exitProcess(3)
        }
    } catch (e: Exception) {
        e.printStackTrace()
        exitProcess(2)
    }
}