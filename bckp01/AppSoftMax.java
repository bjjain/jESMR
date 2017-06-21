package app;

import data.Augmentor;
import data.ClassLabels;
import data.Dataset;
import dimred.DimRed;
import mean.Normalization;
import classifier.softmax.Softmax;
import util.Msg;
import util.Rand;
import util.Reader;
import util.Writer;

import java.io.File;
import java.util.Date;

/**
 * Created by jain on 24/03/2017.
 */
public class AppSoftMax {

    //### OPTIONS ##############################################################

    //--- dataset in UCR format
    private String ucr = "StarLightCurves";

    //--- options
    String opts = "-A 4 -e 200 -l 0.0001 -r 0 -m 0.5 -r1 0.9 -r2 0.99 -T 1000 -S 100 -o 3";
    String meanOpts = "-A DBA -L 300 -T 600 -t 2 -o 1";

    //--- point-wise normalization
    int N = 0;

    //--- length reduction
    int R = 0;

    //--- augmentation factor
    int A = 0;

    //--- toggles z-normalization of time series
    private int Z = 1;

    //--- seed for random numbers
    private int seed = 123456789;

    //--- flag for logging results: 0 - no logging #  1 - logs results
    private static final int LOG = 0;

    //--- path to result directory
    private String log = "./results/";

    //--- path to directory containing time series datasets
    private String path = "/Users/jain/_Data/timeseries/ucr/";

    //### ATTRIBUTES ############################################################

    //--- file of training set
    private String trainfile;

    //--- file of test set
    private String testfile;



    public static void main(String[] args) {
        testUCR();
    }

    static void testUCR() {
        AppSoftMax sm = new AppSoftMax();
        sm.apply();
    }


    public void apply() {
        Rand.setSeed(seed);

        // get data
        setFileNames();
        ClassLabels cl = new ClassLabels();
        Dataset train = load(trainfile, cl);
        Dataset test = load(testfile, cl);

        // check existence
        if(train == null) {
            Msg.error("Error. Training set is missing.");
        }
        if(test == null) {
            Msg.error("Error. Test set is missing.");
        }

        // augment patterns
        train.augment();
        test.augment();

        // normalize
        if (Z > 0) {
            train.normalize();
            test.normalize();
        }
        holdout(train, test);
    }

    private Dataset load(String filename, ClassLabels cl) {
        File f = new File(filename);
        if(f.isFile()) {
            return Reader.load(filename, cl);
        }
        return null;
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

    private void holdout(Dataset train, Dataset test) {

        // augment data
        Augmentor data = new Augmentor(train);
        train = data.augment(A);

        if(0 < R) {
            double delta = 0.001;
            DimRed dr = new DimRed();
            train = dr.reduce(train, delta);
            test = dr.reduce(test, delta);
            train.normalize();
            test.normalize();
        }

        // point-wise normalization
        if(0 < N) {
            Normalization norm = new Normalization(meanOpts);
            norm.fit(train);
            train = norm.normalize(train);
            test = norm.normalize(test);
        }


        // train and test
        Softmax sm = new Softmax(opts);
        sm.fit(train, test);
        double accTrain = sm.score(train);
        double accTest = sm.score(test);
        info(accTrain, accTest);
    }

    private void log(double accTrain, double accTest) {
        Date date = new Date();
        String file = log + ucr + ".txt";
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
