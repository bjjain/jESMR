package app;

import app.utils.LogResults;
import app.utils.Util;
import classifier.Classifier;
import core.Options;
import data.Dataset;
import data.Pattern;
import preprocess.Preprocessor;
import util.Msg;
import util.Rand;

/**
 * Created by jain on 24/03/2017.
 */
public class Synthetic {

    //### OPTIONS ##############################################################

    //--- flag for logging results: 0 - no logging #  1 - logs results
    private static final int LOG = 0;
    //--- options
    String optPP = "-zc 0 -zr 0 -p0 0 -p1 0 -a 1 -b 0.1";
    String optSM = "-p 20 -e 3 -A 4 -l 0.01 -r 0.0 -m 0.5 -r1 0.9 -r2 0.9999 -T 1000 -S 100 -o 3";
    //--- data:
    //      0 = triangle up vs down
    int data = 0;
    //--- type of classifier
    //  0 : elastic softmax
    //  1 : polyhedral max softmax
    //  2 : min elastic softmax
    //  3 : max-min elastic softmax
    //  4 : min-max elastic softmax
    //  5 : DTW softmax
    //  6 : semi-elastic
    int typeClf = 1;

    //--- number of elements per class
    int N = 500;

    //--- length of time series
    int L = 500;

    //--- name of datasets
    String[] dataname = {"triangle"};
    //--- seed for random numbers
    private int seed = 123456789;
    //--- path to log directory
    private String logpath = "./results/";

    public static void main(String[] args) {
        test();
    }

    static void test() {
        Synthetic sm = new Synthetic();
        sm.apply();
    }


    public void apply() {

        Rand.setSeed(seed);

        // get data
        Dataset train = load();
        Dataset test = load();

        // preprocess
        Preprocessor pre = new Preprocessor(optPP);
        pre.fit(train);
        train = pre.apply(train);
        test = pre.apply(test);

        // evaluate
        Classifier clf = Util.getClassifier(typeClf, optSM);
        clf.fit(train);
        double accTrain = clf.score(train);
        double accTest = clf.score(test);

        // report
        info(accTrain, accTest);

        // log
        if (0 < LOG) {
            Options opts = new Options(optSM);
            int p = opts.getInt("-p");
            int e = opts.getInt("-e");
            LogResults logger = new LogResults(clf, logpath, dataname[data], p, e);
            logger.log(train, 0);
        }

    }

    private Dataset load() {
        if (data == 0) {
            return loadTriangleUpDown();
        } else {
            Msg.error("Error! Invalid dataset: %d.", data);
        }
        return null;
    }

    private Dataset loadTriangleUpDown() {
        int K = 2;
        Rand RND = Rand.getInstance();
        Dataset X = new Dataset(K);
        for (int k = 0; k < K; k++) {

            for (int i = 0; i < N; i++) {
                double[] x = new double[L];
                for (int j = 0; j < L; j++) {
                    x[j] = 0.1 * RND.nextGaussian();
                }
                int w = RND.nextInt((int) Math.ceil(0.1 * L));
                w = Math.max(1, w);
                double peak = k == 0 ? +1 : -1;
                peak *= 2.5 * RND.nextDouble();
                double delta = peak / (w + 1);
                int ic = RND.nextInt(L);
                int i0 = Math.max(0, ic - w);
                int i1 = Math.min(L - 1, ic + w);
                x[ic] = peak + 0.1 * RND.nextGaussian();
                int count = 1;
                for (int l = i0; l < ic; l++) {
                    x[l] += count * delta;
                    count++;
                }
                count = 1;
                for (int l = i1; l > ic; l--) {
                    x[l] += count * delta;
                    count++;
                }
                X.add(new Pattern(x, k));
            }
        }
        return X;
    }


    private void info(double accTrain, double accTest) {
        System.out.println();
        System.out.println(result2str(accTrain, accTest));
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
