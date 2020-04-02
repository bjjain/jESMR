package util;

import java.util.Arrays;

public final class Array {

    private Array() { }

    public static double[] cp(double[] x) {
        if (x == null) {
            return null;
        }
        return Arrays.copyOf(x, x.length);
    }

    public static double[][] cp(double[][] x) {
        if (x == null) {
            return null;
        }
        int n = x.length;
        double[][] y = new double[n][];
        for (int i = 0; i < n; i++) {
            y[i] = cp(x[i]);
        }
        return y;
    }

    public static double[][][] cp(double[][][] x) {
        if (x == null) {
            return null;
        }
        int n = x.length;
        double[][][] y = new double[n][][];
        for (int i = 0; i < n; i++) {
            y[i] = cp(x[i]);
        }
        return y;
    }

    public static double max(double[] x) {
        int n = x.length;
        double val = x[0];
        for (int i = 1; i < n; i++) {
            if (val < x[i]) {
                val = x[i];
            }
        }
        return val;
    }
}
