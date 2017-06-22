package app;

import data.ClassLabels;
import data.Dataset;
import util.Msg;
import util.Reader;

import java.io.File;

/**
 * Created by jain on 24/03/2017.
 */
public class HoldOutMNIST extends HoldOut {

    //--- path to directory containing time series datasets
    private String path = "/Users/jain/_Data/mnist/";

    public static void main(String[] args) {
        runMNIST();
    }

    static void runMNIST() {
        HoldOutMNIST sm = new HoldOutMNIST();
        sm.apply();
    }


    Dataset[] getData() {
        ClassLabels cl = new ClassLabels();
        Dataset[] folds = new Dataset[2];
        System.out.print("Load training examples ...");
        folds[0] = load(path + "mnist_train.csv", cl);
        System.out.println(" done.");
        System.out.print("Load test examples ...");
        folds[1] = load(path + "mnist_test.csv", cl);
        System.out.println(" done.");
        check(folds[0], folds[1]);
        return folds;
    }

    private Dataset load(String filename, ClassLabels cl) {
        File f = new File(filename);
        if (f.isFile()) {
            return Reader.load(filename, cl, ",", true);
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
