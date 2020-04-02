package app;

import esmr.Classifier;
import esmr.MLSR;
import esmr.WSR;
import data.Dataset;
import transform.Augment;
import transform.Transform;
import transform.ZNormalize;
import util.Msg;
import util.Rand;
import util.Reader;

/**
 * This class is the entry point for running warped and max-linear softmax regression for UCR datasets of the following
 * format:
 *      - one time series per row
 *      - first value of a row is the class label
 *      - second to last value of a row are the elements of the time series
 *      - values are separated by a comma
 * The directory ./datasets/ includes two example datasets from the UCR repository (Beef, Coffee).
 *
 * NOTE:
 *      1. Time series need to be augmented for including the bias term. For this, method apply() calls the method
 *          augment().
 *      2. Learning rate is selected automatically.
 *
 * The following options need to be set:
 *      - dir           directory of UCR repository
 *      - data          name of UCR dataset
 *      - typeClf       type of classifier: 0 = warped softmax regression, 1 = max-linear softmax regression
 *      - opts          options for classifiers, details are given below
 *      - flagZNORM     toggles z-normalization: 0 = off, 1 = on
 *      - bias          sets value of bias
 *      - seed          seed for random number generator
 *
 * Parameters of opts
 *      -e [int]        elasticity; e > 0; "-e 1" corresponds to standard softmax regression
 *      -l [real]       learning rate; 0 < l < 1 (is selected automatically)
 *      -R [0|1|2]      type of regularization: 0 = no regularization, 1 = L1-regularization, 2 = L2-regularization
 *      -r [real]       regularization parameter; 0 <= r <= 1
 *      -b1 [real]      first momentum of ADAM optimizer
 *      -b2 [real]      second momentum of ADAM optimizer
 *      -T [int]        maximum number of epochs; T > 0
 *      -S [int]        maximum number of epochs without improvement; S > 0
 *      -o [0|1|2]      verbositiy: 0 = quiet, 1 = reports current number of epoch, 2 = reports progress in each epoch
 *
 * Output o = 2
 *      number of epoch, current loss, current error rate, minimum loss, minimum error rate
 */
public class Test {

    String dir = "./datasets/";
    String data = "Coffee";
    int typeClf = 0;
    String opts = " -e 5 -l 0.4 -R 0 -r 0.01 -b1 0.9 -b2 0.99 -T 5000 -S 250 -o 2 ";
    int flagZNORM = 0;
    double bias = -0.1;
    int seed = 10;

    public static void main(String[] args) {
        Test test = new Test();
        test.apply();
    }

    public Test() {
        Rand.setSeed(seed);
    }

    public void apply() {

        // prepare data
        Dataset[] X = getData();
        augment(X);                     // augments time series for bias
        Dataset train = X[0];
        Dataset test = X[1];

        // train and test
        Classifier clf = getClassifier();
        clf.fit(train);
        double errTr = 100.0 * clf.eval(train);
        double errTe = 100.0 * clf.eval(test);

        // print result
        System.out.println();
        System.out.format("data        : %s %n", data);
        System.out.format("classifier  : %s %n", getClassifier().getName());
        System.out.format("options     : %s %n", clf.getOptions());
        System.out.format("train error : %7.4f%n", errTr);
        System.out.format("test error  : %7.4f%n", errTe);
    }

    public Classifier getClassifier() {
        switch (typeClf) {
            case 0:
                return new WSR(opts);
            case 1:
                return new MLSR(opts);
            default:
                Msg.error("Error! Unknown type of classifier: %d.", typeClf);
        }
        return null;
    }

    public Dataset[] getData() {
        String f = dir;
        Dataset[] X;
        f += data + "/" + data;
        X = new Dataset[2];
        X[0] = new Dataset(Reader.loadCSV(f + "_TRAIN.txt"));
        X[1] = new Dataset(Reader.loadCSV(f + "_TEST.txt"));
        if (flagZNORM > 0) {
            Transform t = new ZNormalize();
            X[0] = X[0] != null ? t.transform(X[0]) : null;
            X[1] = X[1] != null ? t.transform(X[1]) : null;
        }
        return X;
    }

    void augment(Dataset[] X) {
        if (X == null || X.length != 2) {
            Msg.error("Error! Incompatible number of m_data sets (2) to be processed.");
        }
        Transform augm = new Augment(bias);
        X[0] = X[0] == null ? null : augm.transform(X[0]);
        X[1] = X[1] == null ? null : augm.transform(X[1]);
    }
}
