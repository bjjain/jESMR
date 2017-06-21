package app;

import classifier.TestSoftmax;
import core.Options;
import data.ClassLabels;
import data.Dataset;
import preprocess.Preprocessor;
import classifier.Log;
import util.*;

import java.io.File;

/**
 * Created by jain on 24/03/2017.
 */
public class AppUCI {

    //### OPTIONS ##############################################################

    //--- dataset in UCR format
    private String uci = "yeast";

    //--- options
    String optPP = "-zc 1 -zr 0 -p0 0 -p1 0 -a 1 -b -0.1";
    String optSM = "-e 3 -A 4 -l 0.03 -r 0 -m 0.9 -r1 0.9 -r2 0.9999 -W -20000 -w 1 -s 1 -T 20000 -S 100 -o 3";

    //--- flag for logging results: 0 - no logging #  1 - logs results
    private int LOG = 0;

    //--- seed for random numbers
    private int seed = 123456789;

    //--- path to log directory
    private String logpath = "./results/";

    //--- path to directory containing time series datasets
    private String path = "/Users/jain/_Data/uci/";

    public static void main(String[] args) {
        testUCI();
    }

    static void testUCI() {
        AppUCI sm = new AppUCI();
        sm.apply();
    }

    static void test() {
        //int[] ela = {1, 3, 5, 7, 10, 15, 20, 25, 30, 50, 75, 100};
        int[] ela = {200, 300, 400, 500};
        //int[] ela = {2500, 5000};
        for(int i = 0; i < ela.length; i++) {
            AppUCI sm = new AppUCI();
            sm.optSM = "-e " + ela[i]
                    + " -A 4 -l 0.0003 -r 0.0 -m 0.9 -r1 0.9 -r2 0.999 -W -100 -w 1 -s 1 -T 1000 -S 100 -o 3";
            sm.apply();
        }
    }


    public void apply() {
        Rand.setSeed(seed);

        // get data
        ClassLabels cl = new ClassLabels();
        Dataset X = load(path + uci + ".csv", cl);

        // check existence
        if(X == null) {
            Msg.error("Error. Training set is missing.");
        }

        // split
        Dataset[] folds = X.split(2.0/3.0);
        Dataset train = folds[0];
        Dataset test = folds[1];

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
            Log logger = new Log(model, logpath, uci, opts.getInt("-e"));
            logger.log(train, 0);
            logger.log(test, 1);
        }

    }

    private Dataset load(String filename, ClassLabels cl) {
        File f = new File(filename);
        System.out.println(filename);
        if(f.isFile()) {
            return Reader.load(filename, cl, ",", false);
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
