package app;

import data.ClassLabels;
import data.Dataset;
import data.Pattern;
import data.Standardization;
import classifier.softmax.Softmax;
import util.*;

import java.util.Arrays;
import java.util.Date;

/**
 * Created by jain on 03.05.17.
 */
public class App2D {

    //### OPTIONS ##############################################################

    //--- options
    String opts = "-A 4 -b 0.1 -e 30 -l 0.1 -r 0 -m 0.9 -r1 0.9 -r2 0.9999 -T 20000 -S 100 -o 3";

    //--- data
    String data = "circle-ring";

    //--- number of elements per class
    int N = 100;

    //--- z-normalization feature-wise
    private int Z0 = 0;

    //--- z-normalization vector-wise
    private int Z1 = 0;

    //--- seed for random numbers
    private int seed = 123456789;

    //--- flag for logging results: 0 - no logging #  1 - logs results
    private static final int LOG = 0;

    //--- path to result directory
    private String log = "./results/";

    public static void main(String[] args) {
        testUCR();
    }

    static void testUCR() {
        App2D sm = new App2D();
        sm.apply();
    }


    public void apply() {
        Rand.setSeed(seed);

        // get data
        ClassLabels cl = new ClassLabels();
        Dataset train = load(N);
        Dataset test = load(N);

        // augment patterns
        train.augment();
        train.pad(0, 0);

        // normalize
        if (Z0 > 0) {
            Standardization std = new Standardization();
            std.fit(train);
            train = std.normalize(train);
            test = std.normalize(test);
        }
        if (Z1 > 0) {
            train.normalize();
            test.normalize();
        }
        holdout(train, test);
    }

    private Dataset load(int N) {

        Rand RND = Rand.getInstance();

        Dataset X = new Dataset(2);
        for(int i = 0; i < N; i++) {
            double x1 = RND.nextGaussian();
            double x2 = RND.nextGaussian();
            double len = x1*x1 + x2*x2;
            if(1 < len) {
                x1 /= len;
                x2 /= len;
            }
            double[] x = {x1, x2};
            X.add(new Pattern(x, 0));
        }

        for(int i = 0; i < N; i++) {
            double x1 = RND.nextGaussian();
            double x2 = RND.nextGaussian();
            double len = x1*x1 + x2*x2;
            if(len < 1) {
                x1 *= 2.0-len;
                x2 *= 2.0-len;
            }
            double[] x = {x1, x2};
            X.add(new Pattern(x, 1));
        }
        return X;
    }


    private void holdout(Dataset train, Dataset test) {

        // train and test
        Softmax sm = new Softmax(opts);
        sm.fit(train, test);
        int[][][] C = sm.updateCounter();
        double accTrain = sm.score(train);
        double accTest = sm.score(test);
        info(accTrain, accTest);
        log(C);
        log(sm, train);

    }

    private void log(Softmax sm, Dataset X) {
        String file = log + data + "_D.txt";
        int n = X.maxLength();
        for(Pattern x : X) {
            double[] z = Arrays.copyOf(x.sequence(), n+2);
            z[n] = x.label();
            z[n+1] = sm.predict(x);;
            String str = Array.toString(z, "%7.4f ");
            Writer.append(str + "\n", file);
        }

    }

    private void log(int[][][] C) {
        String file = log + data + "_C.txt";
        int outUnits = C.length;
        int elasticity = C[0][0].length;
        int[] header = new int[elasticity];
        header[0] = C[0].length;
        String text = Array.toString(header) + "\n";
        Writer.append(text, file);
        for(int i = 0; i < outUnits; i++) {
            Writer.append(Array.toString(C[i]), file);
        }
    }

    private void log(double accTrain, double accTest) {
        Date date = new Date();
        String file = log + data + ".txt";
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
