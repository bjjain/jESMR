package app;
import core.Options;
import data.ClassLabels;
import data.Dataset;
import preprocess.Preprocessor;
import classifier.TestSoftmax;
import util.*;

import java.io.File;

/**
 * Created by jain on 24/03/2017.
 */
public class AppCV_UCR {

    //### OPTIONS ##############################################################

    //--- dataset in UCR format
    private String ucr = "Lighting7";

    //--- number of folds
    private int K = 10;

    //--- elasticity
    private int e = 7;

    //--- options
    String optPP = "-zc 0 -zr 1 -p0 0 -p1 0 -a 1 -b -0.1";
    String optSM = "-e 5 -A 4 -r 0.0 -m 0.9 -r1 0.9 -r2 0.999 -T 1000 -S 100 -o 0";


    //--- seed for random numbers
    private int seed = 123456789;

    //--- path to directory containing time series datasets
    private String path = "/Users/jain/_Data/timeseries/ucr/";


    //### ATTRIBUTES ############################################################

    //--- file of training set
    private String trainfile;

    //--- file of runMNIST set
    private String testfile;



    public static void main(String[] args) {
        AppCV_UCR sm = new AppCV_UCR();
        sm.apply(args);
    }

    public void apply(String[] args) {

        parse(args);
        Rand.setSeed(seed);
        optSM = " -e " + e + " " + optSM;


        // get data
        setFileNames();
        ClassLabels cl = new ClassLabels();
        Dataset X = load(trainfile, cl);
        X.addAll(load(testfile, cl));
        if(X == null) {
            Msg.error("Error. Training set is missing.");
        }

        double eta0 = 0.8;
        double[] acc = new double[K];
        Dataset[] folds = X.splitFolds(K);
        for(int i = 0; i < K; i++) {
            Dataset train = merge(folds, i, X.numLabels()).cp();
            Dataset test = folds[i].cp();

            // preprocess
            Preprocessor pre = new Preprocessor(optPP);
            pre.fit(train);
            train = pre.apply(train);
            test = pre.apply(test);

            // train and evaluate
            System.out.println("Select learning rate.");
            TestSoftmax sm = fit(train, eta0);
            eta0 = Math.min(0.8, 8.0*sm.learningRate());

            System.out.println("Evaluate model.");
            sm.fit(train);
            acc[i] = sm.score(test);

            // report
            System.out.format("Fold %d / %d: %5.3f%n", (i+1), K, acc[i]);
            System.out.println();
        }

        // report
        info(acc);
    }

    private void parse(String[] args) {
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
        flag = "-e";
        if (o.containsKey(flag)) {
            e = o.getInt(flag);
            o.remove(flag);
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

    private Dataset load(String filename, ClassLabels cl) {
        File f = new File(filename);
        if(f.isFile()) {
            return Reader.load(filename, cl);
        }
        return null;
    }

    private Dataset merge(Dataset[] folds, int i0, int numLabels) {
        Dataset X = new Dataset(numLabels);
        for(int i = 0; i < K; i++) {
            if(i != i0) {
                X.addAll(folds[i]);
            }
        }
        return X;
    }

    private TestSoftmax fit(Dataset train, double eta0) {
        TestSoftmax model = null;
        Options opts = new Options(optSM);
        double eta = eta0;
        int state = -1;
        while(state < 0) {
            eta /= 2.0;
            System.out.format("\t eta = %f\n", eta);
            opts.remove("-l");
            opts.add("-l " + eta);
            model = new TestSoftmax(opts.toString());
            model.fit(train);
            state = model.state();
        }
        return model;
    }

    private void info(double[] acc) {

        Options opts = new Options(optSM);
        String e = opts.get("-e");
        double avg = Array.mean(acc);
        double std = Array.std(acc, avg);

        System.out.println();
        System.out.format("111 %s %s %n", ucr, optSM);
        System.out.format("222 %s %s   ", ucr, e);
        System.out.format("%7.4f %7.4f  ", avg, std);
        for(int i = 0; i < K; i++) {
            System.out.format("%7.4f ", acc[i]);
        }
        System.out.println();
    }
}
