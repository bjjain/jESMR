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
        test01();
    }

    static void test01() {
        HoldOutUCR sm = new HoldOutUCR();
        sm.apply();
    }

    static void test02() {
        //int[] ela = {1, 3, 5, 7, 10, 15, 20, 25, 30, 50, 75, 100};
        int[] ela = {2500, 5000};
        for (int i = 0; i < ela.length; i++) {
            HoldOutUCR sm = new HoldOutUCR();
            sm.optSM = "-e " + ela[i]
                    + " -A 4 -l 0.01 -r 0.0 -m 0.9 -r1 0.9 -r2 0.999 -W -100 -w 1 -s 1 -T 20000 -S 100 -o 3";
            sm.apply();
        }
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
