import edu.berkeley.cs.jqf.fuzz.afl.AFLDriver
import edu.berkeley.cs.jqf.fuzz.afl.AFLGuidance
import edu.berkeley.cs.jqf.fuzz.junit.GuidedFuzzing
import java.io.File
import java.time.Duration
import kotlin.system.exitProcess


class MeasureAFL(inputFileName: String, inPipeName: String, outPipeName: String) :
    AFLGuidance(inputFileName, inPipeName, outPipeName) {

}


fun main(args: Array<String>) {
    // wait for the JVM to be completely ready with javaagent and so on
    Thread.sleep(1_000)

    if (args.size != 5) {
        System.err.println("Usage: java " + AFLDriver::class.java + " TEST_CLASS TEST_METHOD TEST_INPUT_FILE AFL_TO_JAVA_PIPE JAVA_TO_AFL_PIPE")
        exitProcess(1)
    }

    val testClassName = args[0]
    val testMethodName = args[1]
    val testInputFile = args[2]
    val a2jPipe = args[3]
    val j2aPipe = args[4]

    try {
        val guidance = MeasureAFL(testInputFile, a2jPipe, j2aPipe)

        // Run the Junit test
        GuidedFuzzing.run(testClassName, testMethodName, guidance, System.out)
    } catch (e: Exception) {
        e.printStackTrace()
        exitProcess(2)
    }
}