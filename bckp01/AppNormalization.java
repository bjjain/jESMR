package app;

import data.ClassLabels;
import data.Dataset;
import mean.Normalization;
import util.Array;
import util.Rand;
import util.Reader;
import util.Writer;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static java.util.stream.Collectors.toList;

/**
 * Created by jain on 24/03/2017.
 */
public class AppNormalization {

    //### OPTIONS ##############################################################

    //--- options
    String meanOpts = "-A DBA -T 50 -t 2 -o 0";

    //--- seed for random numbers
    int seed = 123456789;

    // type of score: 0 - variance # 1 - difference
    int SCORE = 0;

    //--- flag for logs: 0 - no logs #  1 - logs
    int LOG = 0;

    //--- path to result directory
    String log = "./results/";

    //--- path to directory containing time series datasets
    String path = "/Users/jain/_Data/timeseries/ucr/";

    //--- dataset in UCR format
    String ucr = "ElectricDevices";

    //--- toggles z-normalization of time series
    int Z = 1;

    //-- training set
    Dataset train;

    public static void main(String[] args) {
        testUCR();
    }

    static void testUCR() {
        AppNormalization norm = new AppNormalization();
        norm.apply();
    }


    public void apply() {
        Rand.setSeed(seed);
        train = load();
        info(train.maxLength(), new double[4]);
        List<Integer> list = configure(train);
        if(SCORE <= 0) {
            list.parallelStream().map(this::processVar).collect(toList());
        } else {
            list.parallelStream().map(this::processDiff).collect(toList());
        }
    }

    private double processVar(Integer len) {
        Dataset X = train.cp();
        String opts = "-L " + len + " " + meanOpts;
        Normalization norm = new Normalization(opts);
        norm.fit(X);

        double[] var = norm.score(X);
        double[] stats = stats(var);
        info(len, stats);
        return stats[0];
    }

    private double processDiff(Integer len) {
        Dataset X = train.cp();
        String opts = "-L " + len + " " + meanOpts;
        Normalization norm = new Normalization(opts);
        norm.fit(X);

        X = norm.normalize(X);
        mean.DTW dtw = new mean.DTW();
        int n = X.size();
        double[] diff = new double[n];
        for(int i = 0; i < n; i++) {
            diff[i] = Math.sqrt(dtw.d(train.pattern(i), X.pattern(i)));
        }
        double[] stats = stats(diff);
        info(len, stats);
        return stats[0];
    }

    private double[] stats(double[] x) {
        double avg = Array.mean(x);
        double std = Array.std(x, avg);
        double max = Array.max(x);
        double min = Array.min(x);
        return new double[] {avg, std, min, max};
    }

    private List<Integer> configure(Dataset X) {
        int n = 2*X.maxLength();
        List<Integer> list = new ArrayList<>();
        list.add(1);
        for(int i = 5; i <= n; i += 5) {
            list.add(i);
        }
        return list;
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

    private void info(int len, double[] results) {
        String text = results2str(len, results);
        System.out.println(text);
        if(0 < LOG) {
            String file = log + ucr + ".txt";
            Writer.append(text + "\n", file);
        }
    }

    private String results2str(int len, double[] results) {
        StringBuilder s = new StringBuilder();
        s.append(String.format("%4d  ", len));
        for(int i = 0; i < results.length; i++) {
            s.append(String.format("%7.4f  ", Math.sqrt(results[i])));
        }
        return s.toString();
    }

}
