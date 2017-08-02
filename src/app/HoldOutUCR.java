package app;

import data.ClassLabels;
import data.Dataset;
import util.Msg;
import util.Reader;

import java.io.File;

/**
 * Created by jain on 24/03/2017.
 */
public class HoldOutUCR extends HoldOut {

    //--- path to directory containing time series datasets
    String path = "/Users/jain/_Data/timeseries/ucr/";

    public HoldOutUCR() {
        optPP = "-zc 0 -zr 1 " + optPP;
    }

    public static void main(String[] args) {
        test();
    }

    static void test01() {
        HoldOutUCR sm = new HoldOutUCR();
        sm.apply();
    }

    Dataset[] getData() {
        String f = path + dataset + "/" + dataset;
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

}
