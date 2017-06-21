package app;

import data.ClassLabels;
import data.Dataset;
import util.Msg;
import util.Reader;

import java.io.File;

/**
 * Created by jain on 24/03/2017.
 */
public class HoldOutUCI extends HoldOut {

    //--- path to directory containing time series datasets
    String path = "/Users/jain/_Data/uci/";

    public HoldOutUCI() {
        optPP = "-zc 1 -zr 0 " + optPP;
    }

    public static void main(String[] args) {
        test01();
    }

    static void test01() {
        HoldOutUCI sm = new HoldOutUCI();
        sm.apply();
    }

    static void test02() {
        //int[] ela = {1, 3, 5, 7, 10, 15, 20, 25, 30, 50, 75, 100};
        int[] ela = {200, 300, 400, 500};
        //int[] ela = {2500, 5000};
        for (int i = 0; i < ela.length; i++) {
            HoldOutUCI sm = new HoldOutUCI();
            sm.optSM = "-e " + ela[i]
                    + " -A 4 -l 0.0003 -r 0.0 -m 0.9 -r1 0.9 -r2 0.999 -W -100 -w 1 -s 1 -T 1000 -S 100 -o 3";
            sm.apply();
        }
    }

    Dataset[] getData() {
        String filename = path + dataset + ".csv";
        File f = new File(filename);
        Dataset X = Reader.load(filename, new ClassLabels(), ",", false);
        if (X == null) {
            Msg.error("Error. Training set is missing.");
        }
        return X.split(0.5);
    }
}
