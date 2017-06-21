package app;

import classifier.Classifier;
import classifier.Log;
import classifier.PolySoftmax;
import classifier.Softmax;
import core.Options;
import data.Dataset;
import preprocess.Preprocessor;
import util.Msg;
import util.Rand;
import util.Writer;

import java.util.Date;

/**
 * Created by jain on 24/03/2017.
 */
public abstract class HoldOut {

    //### OPTIONS ##############################################################

    String dataset = "sonar";

    //--- type of classifier
    //  0 : elastic softmax
    //  1 : polyhedral max softmax
    int typeClf = 1;

    //--- options
    String optPP = "-p0 0 -p1 0 -a 1 -b 0.1";
    String optSM = "-e 20 -A 4 -l 0.1 -r 0.0 -m 0.9 -r1 0.9 -r2 0.999 -T 1000 -S 100 -o 3";

    //--- flag for logging results: 0 - no logging #  1 - logs results
    int LOG = 0;

    //--- seed for random numbers
    int seed = 123456789;

    //--- path to result directory
    String logpath = "./results/";


    public void apply() {
        Rand.setSeed(seed);

        // set options


        // get data
        Dataset[] folds = getData();
        Dataset train = folds[0];
        Dataset test = folds[1];

        // preprocess
        Preprocessor pre = new Preprocessor(optPP);
        pre.fit(train);
        train = pre.apply(train);
        test = pre.apply(test);

        // evaluate
        Classifier clf = getClassifier();
        clf.fit(train);
        double accTrain = clf.score(train);
        double accTest = clf.score(test);

        // report
        info(accTrain, accTest);

        // log
        if (0 < LOG && typeClf == 0) {
            Options opts = new Options(optSM);
            Log logger = new Log((Softmax) clf, logpath, dataset, opts.getInt("-e"));
            logger.log(train, 0);
            logger.log(test, 1);
        }
    }

    abstract Dataset[] getData();

    private Classifier getClassifier() {
        if (typeClf == 0) {
            return new Softmax(optSM);
        } else if (typeClf == 1) {
            return new PolySoftmax(optSM);
        } else {
            Msg.error("Error! Unknown type of classifier: %d.", typeClf);
        }
        return null;
    }

    private void log(double accTrain, double accTest) {
        Date date = new Date();
        String file = logpath + dataset + ".txt";
        StringBuilder s = new StringBuilder();
        s.append(String.format("***** TEST ACCURACY = %5.2f *****%n", 100 * accTest));
        s.append(date.toString());
        s.append(result2str(accTrain, accTest)).append("\n\n");
        Writer.append(s.toString(), file);
    }

    private void info(double accTrain, double accTest) {
        System.out.println();
        System.out.println(result2str(accTrain, accTest));
        if (0 < LOG) {
            log(accTrain, accTest);
        }
    }

    private String result2str(double accTrain, double accTest) {
        StringBuilder s = new StringBuilder();
        s.append("Result:\n");
        s.append(String.format("\t accTrain = %5.2f%n", 100 * accTrain));
        s.append(String.format("\t accTest  = %5.2f%n", 100 * accTest));
        s.append(String.format("\t errTrain = %5.2f%n", 100 * (1 - accTrain)));
        s.append(String.format("\t errTest  = %5.2f%n", 100 * (1 - accTest)));
        return s.toString();
    }
}
