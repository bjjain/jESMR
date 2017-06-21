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
import util.Writer;

import java.io.File;
import java.util.Date;

/**
 * Created by jain on 24/03/2017.
 */
public class AppUCR {

    //### OPTIONS ##############################################################

    //--- dataset in UCR format
    private String ucr = "Lighting2";

    //--- options
    String optPP = "-zc 0 -zr 1 -p0 0 -p1 0 -a 1 -b -0.1";
    String optSM = "-e 5 -A 4 -l 0.3 -r 0.0 -m 0.9 -r1 0.9 -r2 0.999 -W -100 -w 1 -s 1 -T 20000 -S 100 -o 3";


    //--- flag for logging results: 0 - no logging #  1 - logs results
    private int LOG = 1;

    //--- seed for random numbers
    private int seed = 123456789;

    //--- path to result directory
    private String logpath = "./results/";

    //--- path to directory containing time series datasets
    private String path = "/Users/jain/_Data/timeseries/ucr/";


    //### ATTRIBUTES ############################################################

    //--- file of training set
    private String trainfile;

    //--- file of runMNIST set
    private String testfile;



    public static void main(String[] args) {
        test();
    }

    static void testUCR() {
        AppUCR sm = new AppUCR();
        sm.apply();
    }

    static void test() {
        //int[] ela = {1, 3, 5, 7, 10, 15, 20, 25, 30, 50, 75, 100};
        int[] ela = {2500, 5000};
        for(int i = 0; i < ela.length; i++) {
            AppUCR sm = new AppUCR();
            sm.optSM = "-e " + ela[i]
                    + " -A 4 -l 0.01 -r 0.0 -m 0.9 -r1 0.9 -r2 0.999 -W -100 -w 1 -s 1 -T 20000 -S 100 -o 3";
            sm.apply();
        }
    }


    public void apply() {
        Rand.setSeed(seed);

        // get data
        setFileNames();
        ClassLabels cl = new ClassLabels();
        Dataset train = load(trainfile, cl);
        Dataset test = load(testfile, cl);
        check(train, test);

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
            Log logger = new Log(model, logpath, ucr, opts.getInt("-e"));
            logger.log(train, 0);
            logger.log(test, 1);
        }
    }

    private Dataset load(String filename, ClassLabels cl) {
        File f = new File(filename);
        if(f.isFile()) {
            return Reader.load(filename, cl);
        }
        return null;
    }

    private void check(Dataset train, Dataset test) {
        if(train == null) {
            Msg.error("Error. Training set is missing.");
        }
        if(test == null) {
            Msg.error("Error. Test set is missing.");
        }

    }

    private void setFileNames() {
        boolean isUCR = !ucr.equals("0");
        if(isUCR) {
            String f = path + ucr + "/" + ucr;
            trainfile = f + "_TRAIN";
            testfile =  f + "_TEST";
        } else {
            trainfile = path + trainfile;
            testfile = path + testfile;
        }
    }

    private TestSoftmax holdout(Dataset train, Dataset test) {
        TestSoftmax sm = new TestSoftmax(optSM);
        sm.fit(train, test);
        return sm;
    }

    private void log(double accTrain, double accTest) {
        Date date = new Date();
        String file = logpath + ucr + ".txt";
        StringBuilder s = new StringBuilder();
        s.append(String.format("***** TEST ACCURACY = %5.2f *****%n", 100*accTest));
        s.append(date.toString());
        s.append(result2str(accTrain, accTest)).append("\n\n");
        Writer.append(s.toString(), file);
    }

    private void info(double accTrain, double accTest) {
        System.out.println();
        System.out.println(result2str(accTrain, accTest));
        if(0 < LOG) {
            log(accTrain, accTest);
        }
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
