package app.experiments;

import classifier.Classifier;
import classifier.Status;
import classifier.WarpedSoftmax;
import core.Options;
import data.ClassLabels;
import data.Dataset;
import preprocess.Preprocessor;
import util.Array;
import util.Msg;
import util.Rand;
import util.Reader;

import java.io.File;

/**
 * Created by jain on 24/03/2017.
 * Gun_Point
 * Beef
 * ECG5000
 * Lighting7
 * SwedishLeaf
 */
public class WarpedUCR {


    //### OPTIONS ##############################################################

    //--- name of ucr dataset
    static String ucr = "Lighting7";

    //--- path to directory
    String path = "/Users/jain/_Data/timeseries/ucr/";

    //--- options
    String optPP = " -zc 0 -zr 1 -p0 1 -p1 1 -a 1 -b -0.1 ";
    String optSM1 = " -e 5 -A 4 -l 0.00001 -r 0.0 ";
    String optSM2 = " -m 0.9 -r1 0.9 -r2 0.999 -T 5000 -S 500 -o 3 ";

    //--- seed for random numbers
    int seed = 123456789;


    public static void main(String[] args) {
        experiment(args);
    }

    // simple test run
    static void test() {
        WarpedUCR sm = new WarpedUCR();
        sm.apply();
    }

    static void experiment(String[] args) {
        //double[] ela = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
        double[] ela = {3};
        int numEla = ela.length;
        double[] acc = new double[numEla];
        int[] com = new int[numEla];
        for (int i = 0; i < numEla; i++) {
            int t = 1;
            int status = -1;
            while (status < 0) {
                int e = 1 + t / 2;
                double lr = t % 2 == 1 ? Math.pow(10, -e) : 3.0 * Math.pow(10, -e);
                WarpedUCR sm = new WarpedUCR();
                sm.optSM1 = " -e " + ela[i] + " -l " + lr + " -r 0.0 ";
                System.out.println(sm.optSM1);

                double[] result = sm.apply();
                status = (int) result[0];
                acc[i] = 100.0 * result[2];
                com[i] = (int) (result[3] * result[3] * result[4] * ela[i]);
                t++;
            }
        }
        System.out.println();
        System.out.println("Results: ");
        System.out.println("ACCURACY   " + ucr + " warped-product-0x0 " + Array.toString(acc, "%5.2f "));
        System.out.println("COMPLEXITY " + ucr + " warped-product-0x0 " + Array.toString(com));
        System.out.println("ELASTICITY " + ucr + " warped-product-0x0 " + Array.toString(ela, "%5.2f "));
    }


    Options parse(String[] args) {
        String s = Array.toString(args);
        Options o = new Options("");
        o.add(s);

        String flag = "-dir";
        if (o.containsKey(flag)) {
            path = o.getString(flag);
            o.remove(flag);
        }
        flag = "-ucr";
        if (o.containsKey(flag)) {
            ucr = o.getString(flag);
            o.remove(flag);
        }
        return o;
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
        Classifier clf = new WarpedSoftmax(optSM1 + optSM2);
        Status status = clf.fit(train);

        double accTrain = clf.score(train);
        double accTest = clf.score(test);

        // report
        info(accTrain, accTest);

        return new double[]{status.state, accTrain, accTest, train.maxLength(), train.numLabels()};
    }


    Dataset[] getData() {
        String f = path + ucr + "/" + ucr;
        String trainfile = f + "_TRAIN";
        String testfile = f + "_TEST";
        ClassLabels cl = new ClassLabels();
        Dataset[] folds = new Dataset[2];
        folds[0] = load(trainfile, cl);
        folds[1] = load(testfile, cl);
        check(folds[0], folds[1]);
        return folds;
    }

    private Dataset load(String filename, ClassLabels cl) {
        File f = new File(filename);
        if (f.isFile()) {
            return Reader.load(filename, cl);
        }
        return null;
    }

    private void check(Dataset train, Dataset test) {
        if (train == null) {
            Msg.error("Error. Training set is missing.");
        }
        if (test == null) {
            Msg.error("Error. Test set is missing.");
        }
    }

    private void info(double accTrain, double accTest) {
        System.out.println();
        StringBuilder s = new StringBuilder();
        s.append("Result:\n");
        s.append(String.format("\t accTrain = %5.2f%n", 100 * accTrain));
        s.append(String.format("\t accTest  = %5.2f%n", 100 * accTest));
        s.append(String.format("\t errTrain = %5.2f%n", 100 * (1 - accTrain)));
        s.append(String.format("\t errTest  = %5.2f%n", 100 * (1 - accTest)));
        System.out.println(s.toString());
    }
}
