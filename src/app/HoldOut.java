package app;

import app.utils.Util;
import classifier.Classifier;
import classifier.ElasticSoftmax;
import classifier.Log_ElasticSoftmax;
import classifier.Status;
import core.Options;
import data.Dataset;
import preprocess.Preprocessor;
import util.Array;
import util.Rand;
import util.Writer;

import java.util.Date;

/**
 * Created by jain on 24/03/2017.
 */
public abstract class HoldOut {

    //### OPTIONS ##############################################################

    static String dataset = "Beef";

    //--- type of classifier
    //  0 : elastic-product
    //  1 : max-linear
    //  2 : min elastic-product
    //  3 : warped-product
    //  4 : semi-elastic-product
    int typeClf = 3;
    //--- options
    String optPP = " -p0 0 -p1 0 -a 1 -b -0.1";
    String optSM = "-p 5 -e 3 -A 4 -l 0.003 -r 0 -m 0.9 -r1 0.9 -r2 0.999 -T 5000 -S 100 -o 3";

    // Time series
    //String optSM = "-p 20 -e 3 -A 4 -l 0.03 -r 0 -m 0.9 -r1 0.9 -r2 0.999 -T 1000 -S 100 -o 3";
    // MNIST
    //String optSM = "-p 20 -e 10 -A 4 -l 0.001 -r 0.0 -m 0.9 -r1 0.9 -r2 0.999 -T 500 -S 100 -o 3";


    //--- flag for logging results: 0 - no logging #  1 - logs results
    int LOG = 0;

    //--- seed for random numbers
    int seed = 123456789;

    //--- path to result directory
    String logpath = "./results/";


    static void runWPSM(String[] args) {
        double[] ela = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
        int numEla = ela.length;
        double[] acc = new double[numEla];
        int[] complexity = new int[numEla];
        for (int i = 0; i < numEla; i++) {
            double lr = 0.3;
            int status = -1;
            while (status < 0) {
                lr /= 3.0;
                HoldOutUCR sm = new HoldOutUCR();
                sm.optSM = "-e " + ela[i] + " -l " + lr
                        + " -A 4 -r 0.0 -m 0.9 -r1 0.9 -r 0 -m 0.9 -r1 0.9 -r2 0.999 -T 5000 -S 5000 -o 3";
                System.out.println("options: -e " + ela[i] + " -l " + lr);
                double[] result = sm.apply();
                status = (int) result[0];
                acc[i] = 100.0 * result[2];
                complexity[i] = (int) (result[3] * result[3] * result[4] * ela[i]);
            }
        }
        System.out.println();
        System.out.println("Results: ");
        System.out.println("ACC " + dataset + " warped-product-0x0 " + Array.toString(acc, "%5.2f "));
        System.out.println("COM " + dataset + " warped-product-0x0 " + Array.toString(complexity));
    }


    static void test() {
        int[] ela = {3, 5, 7, 10, 15};
        int numEla = ela.length;
        double[][] acc = new double[2][numEla];
        for (int typeclf = 0; typeclf < 2; typeclf++) {
            for (int i = 0; i < numEla; i++) {
                double lr = 0.3;
                int status = -1;
                while (status < 0) {
                    lr /= 3.0;
                    HoldOutUCR sm = new HoldOutUCR();
                    sm.typeClf = typeclf;
                    sm.optSM = "-e " + ela[i] + " -l " + lr
                            + " -A 4 -r 0.0 -m 0.9 -r1 0.9 -r 0 -m 0.9 -r1 0.9 -r2 0.999 -T 5000 -S 5000 -o 3";
                    double[] result = sm.apply();
                    status = (int) result[0];
                    acc[typeclf][i] = 100.0 * result[2];
                }
            }
        }
        System.out.println();
        System.out.println("Results: ");
        System.out.println(dataset + " Polyhedral " + Array.toString(acc[1], "%5.2f "));
        System.out.println(dataset + " Elastic    " + Array.toString(acc[0], "%5.2f "));
    }

    public double[] apply() {
        Rand.setSeed(seed);

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
        Classifier clf = Util.getClassifier(typeClf, optSM);
        Status status = clf.fit(train);

        double accTrain = clf.score(train);
        double accTest = clf.score(test);

        // report
        info(accTrain, accTest);

        // log
        if (0 < LOG && typeClf == 0) {
            Options opts = new Options(optSM);
            Log_ElasticSoftmax logger = new Log_ElasticSoftmax((ElasticSoftmax) clf, logpath, dataset, opts.getInt("-e"));
            logger.log(train, 0);
            logger.log(test, 1);
        }

        return new double[]{status.state, accTrain, accTest, train.maxLength(), train.numLabels()};
    }

    abstract Dataset[] getData();

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
