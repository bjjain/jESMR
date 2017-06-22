package core;

import util.Array;

import java.util.Arrays;

public final class MinMaxAlignment {

    int m_H;
    double[][] m_scores;
    int[][] m_direction;
    private int[][] m_path;
    private double m_sim;


    public MinMaxAlignment(double[] x, double[][][] w) {
        m_sim = computeSimilarity(x, w);
        m_path = computePath();
    }

    public static double sim(double[] x, double[][][] w) {

        int height = w.length;          // number of partitions
        int depth = w[0].length;        // dimension of input
        int width = w[0][0].length;     // elasticity

        if (depth == 0 || width == 0 || height == 0) {
            return 0;
        }

        if (x.length > depth) {
            w = stretch(w, x.length);
        }

        double sim = Double.MAX_VALUE;
        double[][] S = new double[depth][width];
        for (int h = 0; h < height; h++) {
            double[][] W = w[h];
            S[0][0] = x[0] * W[0][0];
            for (int i = 1; i < depth; i++) {
                S[i][0] = S[i - 1][0] + x[i] * W[i][0];
            }
            for (int j = 1; j < width; j++) {
                S[0][j] = S[0][j - 1] + x[0] * W[0][j];
            }

            double max;
            for (int i = 1; i < depth; i++) {
                for (int j = 1; j < width; j++) {
                    max = S[i - 1][j];
                    if (S[i][j - 1] > max) {
                        max = S[i][j - 1];
                    }
                    if (S[i - 1][j - 1] > max) {
                        max = S[i - 1][j - 1];
                    }
                    S[i][j] = max + x[i] * W[i][j];
                }
            }
            sim = Math.min(sim, S[depth - 1][width - 1]);
        }
        return sim;
    }

    private static double[][][] stretch(double[][][] z, int n) {
        int height = z.length;
        double[][][] cpz = new double[height][][];
        for (int h = 0; h < height; h++) {
            cpz[h] = Arrays.copyOf(z[h], n);
        }
        return cpz;
    }

    public double sim() {
        return m_sim;
    }

    public int indexOfMin() {
        return m_H;
    }

    public int[][] path() {
        return m_path;
    }

    private double computeSimilarity(double[] x, double[][][] w) {

        int height = w.length;          // number of partitions
        int depth = w[0].length;        // dimension of input
        int width = w[0][0].length;     // elasticity

        if (depth == 0 || width == 0 || height == 0) {
            return 0;
        }

        if (x.length > depth) {
            w = stretch(w, x.length);
        }

        m_H = -1;
        m_direction = new int[depth][width];
        m_scores = new double[depth][width];

        double sim = Double.MAX_VALUE;
        double[][] S = new double[depth][width];
        int[][] D = new int[depth][width];
        for (int h = 0; h < height; h++) {
            double[][] W = w[h];
            S[0][0] = x[0] * W[0][0];
            for (int i = 1; i < depth; i++) {
                S[i][0] = S[i - 1][0] + x[i] * W[i][0];
                D[i][0] = 0;
            }
            for (int j = 1; j < width; j++) {
                S[0][j] = S[0][j - 1] + x[0] * W[0][j];
                D[0][j] = 1;
            }

            double max;
            int i0;
            for (int i = 1; i < depth; i++) {
                for (int j = 1; j < width; j++) {
                    max = S[i - 1][j];
                    i0 = 0;
                    if (S[i][j - 1] > max) {
                        max = S[i][j - 1];
                        i0 = 1;
                    }
                    if (S[i - 1][j - 1] > max) {
                        max = S[i - 1][j - 1];
                        i0 = 2;
                    }
                    S[i][j] = max + x[i] * W[i][j];
                    D[i][j] = i0;
                }
            }
            if (S[depth - 1][width - 1] < sim) {
                sim = S[depth - 1][width - 1];
                m_H = h;
                m_scores = Array.cp(S);
                m_direction = Array.cp(D);
            }
        }
        return sim;
    }

    private int[][] computePath() {
        final int LEFT = 0;
        final int UP = 1;

        int xlen = m_scores.length;
        int wlen = m_scores[0].length;
        int ix = xlen - 1;
        int iw = wlen - 1;

        int[][] path = new int[xlen + wlen][2];
        path[0][0] = ix;
        path[0][1] = iw;

        int n = 1;
        int direction;
        while (ix != 0 || iw != 0) {

            if (ix == 0) {
                iw--;
            } else if (iw == 0) {
                ix--;
            } else {
                direction = m_direction[ix][iw];
                if (direction == LEFT) {
                    ix--;
                } else if (direction == UP) {
                    iw--;
                } else {
                    ix--;
                    iw--;
                }
            }
            path[n][0] = ix;
            path[n][1] = iw;
            n++;
        }

        // reverse order and trim
        int[][] p = new int[n][2];
        int m = n - 1;
        for (int i = 0; i < n; i++) {
            p[i] = path[m - i];
        }
        return p;
    }

}
