package app;

import data.ClassLabels;
import data.Dataset;
import dimred.DimRed;
import util.Array;
import util.Rand;
import util.Reader;
import util.Writer;

import java.io.File;
/**
 * Created by jain on 24/03/2017.
 */
public class AppDimRed {

    //### OPTIONS ##############################################################

    // max difference
    double delta = 0.001;

    //--- seed for random numbers
    int seed = 123456789;

    //--- flag for logs: 0 - no logs #  1 - logs
    int LOG = 0;

    //--- path to result directory
    String log = "./results/";

    //--- path to directory containing time series datasets
    String path = "/Users/jain/_Data/timeseries/ucr/";

    //--- dataset in UCR format
    String ucr = "Phoneme";

    //--- toggles z-normalization of time series
    int Z = 1;

    //-- training set
    Dataset train;

    public static void main(String[] args) {
        testUCR();
    }

    static void testUCR() {
        AppDimRed norm = new AppDimRed();
        norm.apply();
    }


    public void apply() {
        Rand.setSeed(seed);
        train = load();
        process();
    }

    private void process() {
        Dataset X = train.cp();
        DimRed dr = new DimRed();
        X = dr.reduce(X, delta);

        int n = X.size();
        double[] len = new double[n];
        for(int i = 0; i < n; i++) {
            len[i] = X.pattern(i).length;
        }

        double[] stats = stats(len);
        info(stats);
    }

    private double[] stats(double[] x) {
        double avg = Array.mean(x);
        double std = Array.std(x, avg);
        double max = Array.max(x);
        double min = Array.min(x);
        return new double[] {avg, std, min, max};
    }

    private Dataset load() {
        Dataset X = null;
        String file = path + ucr + "/" + ucr + "_TRAIN";
        File f = new File(file);
        if(f.isFile()) {
            ClassLabels cl = new ClassLabels();
            X = Reader.load(file, cl);
            if(Z > 0) {
                X.normalize();
            }
        }
        return X;
    }

    private void info(double[] results) {
        String text = results2str(results);
        System.out.println(text);
        if(0 < LOG) {
            String file = log + ucr + ".txt";
            Writer.append(text + "\n", file);
        }
    }

    private String results2str(double[] results) {
        StringBuilder s = new StringBuilder();
        for(int i = 0; i < results.length; i++) {
            s.append(String.format("%7.4f  ", results[i]));
        }
        return s.toString();
    }

}
