import edu.berkeley.cs.jqf.fuzz.ei.ZestDriver
import edu.berkeley.cs.jqf.fuzz.ei.ZestGuidance
import edu.berkeley.cs.jqf.fuzz.guidance.Result
import edu.berkeley.cs.jqf.fuzz.junit.GuidedFuzzing
import edu.berkeley.cs.jqf.instrument.tracing.events.TraceEvent
import java.io.BufferedReader
import java.io.File
import java.io.InputStream
import java.lang.Boolean
import java.time.Duration
import kotlin.Array
import kotlin.Exception
import kotlin.String
import kotlin.arrayOfNulls


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
    if (args.size < 2) {
        System.err.println("Usage: java " + ZestDriver::class.java + " TEST_CLASS TEST_METHOD [OUTPUT_DIR [SEED_DIR | SEED_FILES...]]")
        System.exit(1)
    }
    val testClassName = args[0]
    val testMethodName = args[1]
    val outputDirectoryName = if (args.size > 2) args[2] else "fuzz-results"
    val outputDirectory = File(outputDirectoryName)

    try {
        // Load the guidance
        val title = "$testClassName#$testMethodName"
        var guidance: ZestGuidance? = null
        guidance = MeasureZest(title, null, outputDirectory)


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
            System.exit(3)
        }
    } catch (e: Exception) {
        e.printStackTrace()
        System.exit(2)
    }
}