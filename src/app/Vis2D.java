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
public class Vis2D {

    //### OPTIONS ##############################################################

    //--- flag for logging results: 0 - no logging #  1 - logs results
    private static final int LOG = 1;
    //--- options
    String optPP = "-zc 0 -zr 0 -p0 0 -p1 0 -a 1 -b 0.1";
    String optSM = "-p 20 -e 10 -A 4 -l 0.01 -r 0.0 -m 0.5 -r1 0.9 -r2 0.9999 -T 5000 -S 1000 -o 3";
    //--- data:
    //      0 = rings
    //      1 = swiss-roll
    //      2 = parallel
    //      3 = xor
    //      4 = grid
    //      5 = circle-ring
    int data = 1;
    //--- type of classifier
    //  0 : elastic softmax
    //  1 : polyhedral max softmax
    //  2 : min elastic softmax
    //  3 : max-min elastic softmax
    //  4 : min-max elastic softmax
    int typeClf = 4;
    //--- number of elements per class
    int N = 500;
    //*** data-specific parameters
    //--- 0: parameter for rings
    int K = 3;                      // number of rings
    //--- 5: parameter for circle-ring
    int LT = 1;                     // label of circle
    //--- 0, 5: parameter for rings & circle-ring
    double sigma = 1.0;
    double gap = 0.0;
    //--- 4: parameter for grid
    int rows = 3;
    int cols = 3;
    //--- name of datasets
    String[] dataname = {"rings", "swiss-roll", "parallel", "xor", "grid", "circle-ring"};
    //--- seed for random numbers
    private int seed = 123456789;
    //--- path to log directory
    private String logpath = "./results/";

    public static void main(String[] args) {
        test();
    }

    static void test() {
        Vis2D sm = new Vis2D();
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
            return loadRings();
        } else if (data == 1) {
            return loadSwissRoll();
        } else if (data == 2) {
            return loadParallel();
        } else if (data == 3) {
            return loadXor();
        } else if (data == 4) {
            return loadGrid();
        } else if (data == 5) {
            return loadCircleRing();
        } else {
            Msg.error("Error! Invalid dataset: %d.", data);
        }
        return null;
    }

    private Dataset loadRings() {
        if (K < 2) {
            Msg.error("Invalid number of classes: %d.", K);
        }
        Rand RND = Rand.getInstance();
        Dataset X = new Dataset(K);
        for (int k = 0; k < K; k++) {
            double g = k == 0 ? 0 : gap;
            double sigma0 = k * sigma + g;
            double sigma1 = (k + 1) * sigma;
            int count = 0;
            while (count < N) {
                double x1 = sigma1 * RND.nextGaussian();
                double x2 = sigma1 * RND.nextGaussian();
                double len = Math.sqrt(x1 * x1 + x2 * x2);
                if (sigma0 <= len && len < sigma1) {
                    double[] x = {x1, x2};
                    X.add(new Pattern(x, k));
                    count++;
                }
            }
        }
        return X;
    }

    private Dataset loadSwissRoll() {

        Rand RND = Rand.getInstance();

        double z0 = 0.5;
        double z1 = 10.0;

        int n = 2 * N;
        Dataset X = new Dataset(2);
        for (int i = 0; i < n; i++) {
            double r = z0 + z1 * RND.nextDouble();
            double x1 = r * Math.cos(r);
            double x2 = r * Math.sin(r);

            if (i % 2 == 0) {
                double[] x = {x1, x2};
                X.add(new Pattern(x, 0));
            } else {
                double[] x = {-x1, -x2};
                X.add(new Pattern(x, 1));
            }
        }
        return X;
    }

    private Dataset loadParallel() {

        int K = 6;
        double gap = 0.2;
        Rand RND = Rand.getInstance();
        Dataset X = new Dataset(2);

        double x2 = gap;
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < N; i++) {
                double x1 = 10.0 * RND.nextDouble();
                double[] x = {x1, x2};
                if (k % 2 == 0) {
                    X.add(new Pattern(x, 0));
                } else {
                    X.add(new Pattern(x, 1));
                }
            }
            x2 += gap;
        }
        return X;
    }

    private Dataset loadXor() {
        Rand RND = Rand.getInstance();
        Dataset X = new Dataset(2);
        int n = 4 * N;
        for (int i = 0; i < n; i++) {
            double x1 = RND.nextDouble();
            double x2 = RND.nextDouble();
            double[] x = {x1, x2};
            if (0 <= (x1 - 0.5) * (x2 - 0.5)) {
                X.add(new Pattern(x, 0));
            } else {
                X.add(new Pattern(x, 1));
            }
        }
        return X;
    }

    private Dataset loadGrid() {
        Rand RND = Rand.getInstance();
        int numLabels = rows * cols;
        Dataset X = new Dataset(numLabels);

        int y = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < N; k++) {
                    double x1 = i + RND.nextDouble();
                    double x2 = j + RND.nextDouble();
                    double[] x = {x1, x2};
                    X.add(new Pattern(x, y));
                }
                y++;
            }
        }
        return X;
    }

    private Dataset loadCircleRing() {

        Rand RND = Rand.getInstance();
        int K = 2;
        Dataset X = new Dataset(K);
        for (int k = 0; k < K; k++) {
            int label = k == 0 ? LT : 1 - LT;
            double g = k == 0 ? 0 : gap;
            double sigma0 = k * sigma + g;
            double sigma1 = sigma0 + sigma;
            int count = 0;
            while (count < N) {
                double x1 = sigma1 * RND.nextGaussian();
                double x2 = sigma1 * RND.nextGaussian();
                double len = Math.sqrt(x1 * x1 + x2 * x2);
                double[] x = {x1, x2};
                if (sigma0 <= len && len < sigma1) {
                    X.add(new Pattern(x, label));
                    count++;
                }
            }
        }
        return X;
    }


    private Dataset loadDiag4DTrain() {

        Rand RND = Rand.getInstance();
        Dataset X = new Dataset(2);
        int[] count = {0, 0};
        while (count[0] < N || count[1] < N) {
            double x1 = RND.nextDouble();
            double x2 = RND.nextDouble();
            double x3 = RND.nextDouble();
            if (count[0] < N && x2 < 0.5) {
                double[] x = {x1, x2, x2, x3};
                X.add(new Pattern(x, 0));
                count[0]++;
            }
            if (count[1] < N && x2 > 0.5) {
                double[] x = {x1, x2, x2, x3};
                X.add(new Pattern(x, 1));
                count[1]++;
            }
        }
        return X;
    }

    private Dataset loadDiag4DTest() {

        Rand RND = Rand.getInstance();
        int K = 2;
        Dataset X = new Dataset(K);
        int[] count = {0, 0};
        while (count[0] < N || count[1] < N) {
            double x1 = RND.nextDouble();
            double x2 = RND.nextDouble();
            double x3 = RND.nextDouble();
            if (count[0] < N && x2 < 0.5) {
                boolean left = RND.nextBoolean();
                if (left) {
                    double[] x = {x1, x2, x3, x3};
                    X.add(new Pattern(x, 0));
                } else {
                    double[] x = {x1, x1, x2, x3};
                    X.add(new Pattern(x, 0));
                }
                count[0]++;
            }
            if (count[1] < N && x2 > 0.5) {
                boolean left = RND.nextBoolean();
                if (left) {
                    double[] x = {x1, x2, x3, x3};
                    X.add(new Pattern(x, 1));
                } else {
                    double[] x = {x1, x1, x2, x3};
                    X.add(new Pattern(x, 1));
                }
                count[1]++;
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
