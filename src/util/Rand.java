package util;

import java.util.Random;

public class Rand extends Random {

    private static int SEED = -1;
    private static Rand RND = new Rand();

    private Rand() {
        super();
    }

    private Rand(int seed) {
        super(seed);
    }

    public static Rand getInstance() {
        return RND;
    }

    public static void setSeed(int seed) {
        SEED = seed;
        RND = new Rand(SEED);
    }

    public int[] shuffle(int n) {
        int[] x = new int[n];
        for (int i = 0; i < n; i++) {
            x[i] = i;
        }
        for (int i = 0; i < n; i++) {
            int pos = nextInt(n);
            int tmp = x[i];
            x[i] = x[pos];
            x[pos] = tmp;
        }
        return x;
    }

    public int[] choose(int k, int n) {
        if (k < 0) {
            return null;
        }
        if (k == 0) {
            return new int[0];
        }
        if (n == 1) {
            return new int[1];
        }
        if (k == n) {
            return arr(n);
        }

        int[] set = arr(n);

        if (k >= n) {
            return set;
        }

        int[] result = new int[k];
        int index;
        for (int i = 0; i < k; i++) {
            index = nextInt(n - i);
            result[i] = set[index];
            set[index] = set[n - i - 1];
        }
        return result;
    }

    private int[] arr(int n) {
        int[] x = new int[n];
        for (int i = 0; i < n; i++) {
            x[i] = i;
        }
        return x;
    }

    public double[] nextArray(int n, double beta) {
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = nextGaussian() / beta;
        }
        return x;
    }

    public double[][] nextArray(int p, int q, double beta) {
        double[][] x = new double[p][q];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < q; j++) {
                x[i][j] = nextGaussian() / beta;
            }
        }
        return x;
    }

    public double[][][] nextArray(int p, int q, int r, double beta) {
        double[][][] x = new double[p][q][r];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < q; j++) {
                for (int k = 0; k < r; k++) {
                    x[i][j][k] = nextGaussian() / beta;
                }
            }
        }
        return x;
    }

    public double[][][][] nextArray(int p, int q, int r, int s, double beta) {
        double[][][][] x = new double[p][q][r][s];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < q; j++) {
                for (int k = 0; k < r; k++) {
                    for (int l = 0; l < s; l++) {
                        x[i][j][k][l] = nextGaussian() / beta;
                    }
                }
            }
        }
        return x;
    }

    public double[] perturbe(double[] x) {
        int n = x.length;
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            z[i] = x[i] + 0.01 * nextGaussian();
        }
        return z;
    }

}
