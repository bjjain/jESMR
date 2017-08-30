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


/*
iris
eye
sonar
whitewine
occupancy
letter
pima
ionosphere
glass
ecoli
 */
public class WarpedUCI {


    //### OPTIONS ##############################################################

    //--- name of ucr dataset
    String uci = "ionosphere";

    //--- path to directory
    String path = "/Users/jain/_Data/uci/";

    //--- options
    String optPP = " -zc 1 -zr 0 -p0 1 -p1 1 -a 1 -b -0.1 ";
    String optSM1 = " -e 5 -A 4 -l 0.00001 -r 0.0 ";
    String optSM2 = " -m 0.9 -r1 0.9 -r2 0.999 -T 5000 -S 250 -o 3 ";

    //--- seed for random numbers
    int seed = 123456789;


    public static void main(String[] args) {
        experiment(args);
    }

    // simple test run
    static void test() {
        WarpedUCI sm = new WarpedUCI();
        sm.apply();
    }

    static void experiment(String[] args) {

        //double[] ela = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
        double[] ela = {1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15, 20};
        int numEla = ela.length;
        Log log = new Log("warped-product", numEla);
        for (int i = 0; i < numEla; i++) {
            int t = 1;
            boolean decrLR = true;
            while (decrLR) {
                int e = 1 + t / 2;
                double lr = t % 2 == 1 ? Math.pow(10, -e) : 3.0 * Math.pow(10, -e);
                WarpedUCI sm = new WarpedUCI();
                sm.optSM1 = " -e " + ela[i] + " -l " + lr + " -r 0.0 ";
                System.out.println(sm.optSM1);
                log.setPointer(i);
                decrLR = sm.apply(log, args);
                t++;
            }
        }
        log.info();
    }

    private boolean apply(Log log, String[] args) {

        parse(args);
        Rand.setSeed(seed);

        // get data
        Dataset[] folds = getData();
        Dataset train = folds[0];
        Dataset test = folds[1];
        log.set(uci, train.maxLength(), train.numLabels());

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
        log.set(status, accTrain, accTest, getElasticity(clf));

        return status.decrLearningRate();
    }

    private double getElasticity(Classifier clf) {
        Options opts = clf.getOptions();
        String flag = "-e";
        if (opts.containsKey(flag)) {
            return opts.getDouble(flag);
        } else {
            return -1;
        }
    }

    public void apply() {
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
        clf.fit(train);

        double accTrain = clf.score(train);
        double accTest = clf.score(test);

        // report
        info(accTrain, accTest);

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
        flag = "-uci";
        if (o.containsKey(flag)) {
            uci = o.getString(flag);
            o.remove(flag);
        }
        return o;
    }

    Dataset[] getData() {
        String filename = path + uci + ".csv";
        File f = new File(filename);
        Dataset X = Reader.load(filename, new ClassLabels(), ",", false);
        if (X == null) {
            Msg.error("Error. Training set is missing.");
        }
        return X.split(1.0 / 3.0);
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
        s.append(String.format("\t m_accTrain = %5.2f%n", 100 * accTrain));
        s.append(String.format("\t m_accTest  = %5.2f%n", 100 * accTest));
        s.append(String.format("\t errTrain = %5.2f%n", 100 * (1 - accTrain)));
        s.append(String.format("\t errTest  = %5.2f%n", 100 * (1 - accTest)));
        System.out.println(s.toString());
    }
}
