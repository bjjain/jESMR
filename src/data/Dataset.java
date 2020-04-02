package data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

@SuppressWarnings("serial")
public class Dataset extends ArrayList<Pattern> {

    Integer numLabels;

    public Dataset() {
        super();
    }

    /*
     * Converts data-matrix into data set of labeled time series. Every row of the matrix starts with a class label
     * followed by a time series.
     */
    public Dataset(double[][] data) {
        for (double[] x : data) {
            int y = (int) x[0];
            add(new Pattern(Arrays.copyOfRange(x, 1, x.length), y));
        }
    }

    public double[][] patterns() {
        int n = size();
        double[][] X = new double[n][];
        for (int i = 0; i < n; i++) {
            X[i] = get(i).x;
        }
        return X;
    }

    public int[] labels() {
        int n = size();
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            y[i] = label(i);
        }
        return y;
    }

    public int label(int i) {
        return get(i).y;
    }

    public int numLabels() {
        if (numLabels != null) {
            return numLabels;
        }
        HashSet<Integer> labels = new HashSet<>();
        for (Pattern p : this) {
            labels.add(p.y);
        }
        numLabels = labels.size();
        return numLabels;
    }

    public int maxlength() {
        int max = 0;
        for (Pattern p : this) {
            if (max < p.length()) {
                max = p.length();
            }
        }
        return max;
    }
}
