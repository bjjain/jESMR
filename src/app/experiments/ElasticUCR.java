package app.experiments;

import classifier.Classifier;
import classifier.ElasticSoftmax;
import classifier.Status;
import core.Options;
import data.ClassLabels;
import data.Dataset;
import preprocess.Preprocessor;
import util.Array;
import util.Msg;
import util.Rand;
import util.Reader;

import java.io.File;

public class ElasticUCR {


    //### OPTIONS ##############################################################

    //--- name of ucr dataset
    String ucr = "DistalPhalanxOutlineAgeGroup";

    //--- path to directory
    String path = "/Users/jain/_Data/timeseries/ucr/";

    //--- options
    String optPP = " -zc 0 -zr 1 -p0 0 -p1 0 -a 1 -b -0.1 ";
    String optSM1 = " -e 5 -A 4 -l 0.00001 -r 0.0 ";
    String optSM2 = " -m 0.9 -r1 0.9 -r2 0.999 -T 5000 -S 250 -o 3 ";

    //--- seed for random numbers
    int seed = 123456789;


    public static void main(String[] args) {
        experiment(args);
    }

    // simple test run
    static void test() {
        ElasticUCR sm = new ElasticUCR();
        sm.apply();
    }

    static void experiment(String[] args) {

        double[] ela = {1, 2, 3, 4, 5, 7, 10, 15, 20};
        int numEla = ela.length;
        Log log = new Log("elastic-product", numEla);
        for (int i = 0; i < numEla; i++) {
            int t = 1;
            boolean decrLR = true;
            while (decrLR) {
                int e = 1 + t / 2;
                double lr = t % 2 == 1 ? Math.pow(10, -e) : 3.0 * Math.pow(10, -e);
                ElasticUCR sm = new ElasticUCR();
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
        log.set(ucr, train.maxLength(), train.numLabels());

        // preprocess
        Preprocessor pre = new Preprocessor(optPP);
        pre.fit(train);
        train = pre.apply(train);
        test = pre.apply(test);

        // evaluate
        Classifier clf = new ElasticSoftmax(optSM1 + optSM2);
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
        Classifier clf = new ElasticSoftmax(optSM1 + optSM2);
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
        flag = "-ucr";
        if (o.containsKey(flag)) {
            ucr = o.getString(flag);
            o.remove(flag);
        }
        return o;
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
        s.append(String.format("\t m_accTrain = %5.2f%n", 100 * accTrain));
        s.append(String.format("\t m_accTest  = %5.2f%n", 100 * accTest));
        s.append(String.format("\t errTrain = %5.2f%n", 100 * (1 - accTrain)));
        s.append(String.format("\t errTest  = %5.2f%n", 100 * (1 - accTest)));
        System.out.println(s.toString());
    }
}
