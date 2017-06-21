package app;

import classifier.TestSoftmax;
import core.Options;
import data.ClassLabels;
import data.Dataset;
import preprocess.Preprocessor;
import classifier.Log;
import util.Msg;
import util.Rand;
import util.Reader;

import java.io.File;

/**
 * Created by jain on 24/03/2017.
 */
public class AppMNIST {

    //### OPTIONS ##############################################################

    //--- options
    String optPP = "-zc 0 -zr 0 -p0 0 -p1 0 -a 0 -b -0.1";
    String optSM = "-e 4 -A 4 -l 0.0003 -r 0 -m 0.5 -r1 0.9 -r2 0.9999 -W -20000 -w 1 -s 1 -T 20000 -S 100 -o 3";

    //--- flag for logging results: 0 - no logging #  1 - logs results
    private int LOG = 0;

    //--- seed for random numbers
    private int seed = 123456789;

    //--- path to log directory
    private String logpath = "./results/";

    //--- path to directory containing time series datasets
    private String path = "/Users/jain/_Data/mnist/";

    public static void main(String[] args) {
        runMNIST();
    }

    static void runMNIST() {
        AppMNIST sm = new AppMNIST();
        sm.apply();
    }


    public void apply() {
        Rand.setSeed(seed);

        // get data
        System.out.print("Load training examples ...");
        ClassLabels cl = new ClassLabels();
        Dataset train = load(path + "mnist_train.csv", cl);
        System.out.println(" done.");
        System.out.print("Load runMNIST examples ...");
        Dataset test = load(path + "mnist_test.csv", cl);
        System.out.println(" done.");

        // check existence
        if(train == null) {
            Msg.error("Error. Training set is missing.");
        }
        if(test == null) {
            Msg.error("Error. Test set is missing.");
        }

        // preprocess
        Preprocessor pre = new Preprocessor(optPP);
        pre.fit(train);
        train = pre.apply(train);
        test = pre.apply(test);



        // evaluate
        TestSoftmax model = holdout(train, test);
        double accTrain = model.score(train);
        double accTest = model.score(test);

        // report
        info(accTrain, accTest);

        // log
        if(0 < LOG) {
            Options opts = new Options(optSM);
            Log logger = new Log(model, logpath, "mnist", opts.getInt("-e"));
            logger.log(train, 0);
        }

    }

    private Dataset load(String filename, ClassLabels cl) {
        File f = new File(filename);
        System.out.println(filename);
        if(f.isFile()) {
            return Reader.load(filename, cl, ",", true);
        }
        return null;
    }


    private TestSoftmax holdout(Dataset train, Dataset test) {
        // train and runMNIST
        TestSoftmax sm = new TestSoftmax(optSM);
        sm.fit(train, test);
        return sm;
    }

    private void info(double accTrain, double accTest) {
        System.out.println();
        System.out.println(result2str(accTrain, accTest));
    }

    private String result2str(double accTrain, double accTest) {
        StringBuilder s = new StringBuilder();
        s.append("Result:\n");
        s.append(String.format("\t accTrain = %5.2f%n", 100*accTrain));
        s.append(String.format("\t accTest  = %5.2f%n", 100*accTest));
        s.append(String.format("\t errTrain = %5.2f%n", 100*(1-accTrain)));
        s.append(String.format("\t errTest  = %5.2f%n", 100*(1-accTest)));
        return s.toString();
    }
}
