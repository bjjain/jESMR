package util;

import java.util.Arrays;

public final class Array {

    private static final String NULL = "<null>";

    private Array() {
    }

    public static int sum(int[] x) {
        int n = x.length;
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += x[i];
        }
        return sum;
    }

    public static double[][] fill(double[][] x, double value) {
        int n = x.length;
        for (int i = 0; i < n; i++) {
            Arrays.fill(x[i], value);
        }
        return x;
    }

    public static double[] normalize(double[] x) {
        int n = x.length;
        double mean = mean(x);
        double std = std(x, mean);
        double[] z = new double[n];
        if (std == 0) {
            return z;
        }
        for (int i = 0; i < n; i++) {
            z[i] = (x[i] - mean) / std;
        }
        return z;
    }


    public static int[] cp(int[] x) {
        if (x == null) {
            return null;
        }
        return Arrays.copyOf(x, x.length);
    }

    public static double[] cp(double[] x) {
        if (x == null) {
            return null;
        }
        return Arrays.copyOf(x, x.length);
    }

    public static int[][] cp(int[][] x) {
        if (x == null) {
            return null;
        }
        int n = x.length;
        int[][] y = new int[n][];
        for (int i = 0; i < n; i++) {
            y[i] = cp(x[i]);
        }
        return y;
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

    public static double[][][][] cp(double[][][][] x) {
        if (x == null) {
            return null;
        }
        int n = x.length;
        double[][][][] y = new double[n][][][];
        for (int i = 0; i < n; i++) {
            y[i] = cp(x[i]);
        }
        return y;
    }

    public static double min(double[] x) {
        int n = x.length;
        double val = x[0];
        for (int i = 1; i < n; i++) {
            if (x[i] < val) {
                val = x[i];
            }
        }
        return val;
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

    public static int indexOfMin(double[] x) {
        int n = x.length;
        int idx = 0;
        for (int i = 1; i < n; i++) {
            if (x[idx] > x[i]) {
                idx = i;
            }
        }
        return idx;
    }

    public static int indexOfMax(double[] x) {
        int n = x.length;
        int idx = 0;
        for (int i = 1; i < n; i++) {
            if (x[idx] < x[i]) {
                idx = i;
            }
        }
        return idx;
    }

    public static double sum(double[] x) {
        int n = x.length;
        double sum = x[0];
        for (int i = 1; i < n; i++) {
            sum += x[i];
        }
        return sum;
    }


    /*** Statistics ***********************************************************/
    public static double mean(double[] x) {
        if (x == null || x.length == 0) {
            return 0;
        }
        return Array.sum(x) / ((double) x.length);
    }


    public static double var(double[] x, double mean) {
        if (x == null || x.length == 0) {
            return 0;
        }
        double v = 0;
        int n = x.length;
        for (double aX : x) {
            v += Math.pow(aX - mean, 2);
        }
        return v / ((double) n);
    }

    public static double std(double[] x, double mean) {
        return Math.sqrt(var(x, mean));
    }


    /*** Converters ***********************************************************/
    public static String toString(int[] x) {
        if (x == null) {
            return NULL;
        }
        StringBuilder s = new StringBuilder();
        for (double val : x) {
            s.append(val).append(" ");
        }
        return s.toString();
    }

    public static String toString(int[][] x) {
        if (x == null) {
            return NULL;
        }
        StringBuilder s = new StringBuilder();
        for (int[] val : x) {
            s.append(toString(val)).append("\n");
        }
        return s.toString();
    }

    public static String toString(double[] x, String format) {
        if (x == null) {
            return NULL;
        }
        StringBuilder s = new StringBuilder();
        for (double val : x) {
            s.append(String.format(format, val));
        }
        return s.toString();
    }

    public static String toString(double[][] x, String format) {
        if (x == null) {
            return NULL;
        }
        StringBuilder s = new StringBuilder();
        for (double[] val : x) {
            s.append(toString(val, format)).append("\n");
        }
        return s.toString();
    }

    public static String toString(String[] x) {
        if (x == null) {
            return "";
        }
        StringBuilder s = new StringBuilder();
        for (String str : x) {
            s.append(str).append(" ");
        }
        return s.toString();
    }
}
