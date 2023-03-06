import edu.berkeley.cs.jqf.fuzz.ei.ZestDriver
import edu.berkeley.cs.jqf.fuzz.ei.ZestGuidance
import edu.berkeley.cs.jqf.fuzz.guidance.Result
import edu.berkeley.cs.jqf.fuzz.junit.GuidedFuzzing
import edu.berkeley.cs.jqf.fuzz.util.ICoverage
import edu.berkeley.cs.jqf.instrument.tracing.events.BranchEvent
import edu.berkeley.cs.jqf.instrument.tracing.events.CallEvent
import edu.berkeley.cs.jqf.instrument.tracing.events.ReturnEvent
import edu.berkeley.cs.jqf.instrument.tracing.events.TraceEvent
import java.io.File
import java.io.InputStream
import java.security.MessageDigest
import java.time.Duration
import kotlin.Array
import kotlin.String
import kotlin.system.exitProcess


class MeasureZest(testName: String?, duration: Duration?, outputDirectory: File?) :
    ZestGuidance(testName, duration, outputDirectory) {
    // call event -> 0, return event -> 1, branch event -> 2, total events -> 3
    var events = mutableListOf<TraceEvent?>()

    override fun displayStats(force: kotlin.Boolean) {
//        super.displayStats(force)
    }

    override fun getTotalCoverage(): ICoverage<*> {
        return super.getTotalCoverage()
    }

    override fun getInput(): InputStream {
        return super.getInput()
    }

    override fun handleEvent(e: TraceEvent?) {
        events.add(e)
        super.handleEvent(e)
    }

    private fun hash(str:String): String {
            val bytes = str.toByteArray()
            val md = MessageDigest.getInstance("SHA-256")
            val digest = md.digest(bytes)
            return digest.fold("") { s, it -> s + "%02x".format(it) }.substring(0,6)

    }

    override fun handleResult(result: Result?, error: Throwable?) {
        println("total events for input ${this.numTrials}: ${events.size}")
        events.toList().filterIsInstance<BranchEvent>().take(40).forEach { print(hash(it.iid.toString())+" ") }
        println()
        events = mutableListOf()
        super.handleResult(result, error)
        val maxAllowedInputs = 11
        if (this.numTrials > maxAllowedInputs){
            println("Finished $maxAllowedInputs trials")
            exitProcess(0)
        }
    }

    override fun checkSavingCriteriaSatisfied(result: Result?): List<String> {
        return listOf("dumpall")
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
    val outputDirectory = File("fuzz-results")

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