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
}
