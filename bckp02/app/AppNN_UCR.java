package app;
import core.Options;
import data.ClassLabels;
import data.Dataset;
import mean.NearestNeighbor;
import preprocess.Preprocessor;
import util.*;

import java.io.File;

/**
 * Created by jain on 24/03/2017.
 */
public class AppNN_UCR {

    //### OPTIONS ##############################################################

    //--- dataset in UCR format
    private String ucr = "Wine";

    //--- number of folds
    private int K = 10;


    //--- options
    String optPP = "-zc 0 -zr 1 -p0 0 -p1 0 -a 0";

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
        AppNN_UCR sm = new AppNN_UCR();
        sm.apply(args);
    }

    public void apply(String[] args) {

        parse(args);
        Rand.setSeed(seed);

        // get data
        setFileNames();
        ClassLabels cl = new ClassLabels();
        Dataset X = load(trainfile, cl);
        X.addAll(load(testfile, cl));
        if(X == null) {
            Msg.error("Error. Training set is missing.");
        }

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
            NearestNeighbor nn = new NearestNeighbor();
            nn.fit(train);
            acc[i] = nn.score(test);

            // report
            System.out.format("Fold %d / %d: %5.3f%n", (i+1), K, acc[i]);
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

    private void info(double[] acc) {

        double avg = Array.mean(acc);
        double std = Array.std(acc, avg);

        System.out.println();
        System.out.format("222 %s   ", ucr);
        System.out.format("%7.4f %7.4f  ", avg, std);
        for(int i = 0; i < K; i++) {
            System.out.format("%7.4f ", acc[i]);
        }
        System.out.println();
    }

}
