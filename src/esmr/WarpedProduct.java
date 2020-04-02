package esmr;

public final class WarpedProduct {

    private int[][] m_path;
    private double m_prod;

    public WarpedProduct(double[][] w, double[] x) {
        warp(w, x);
    }

    public double score() {
        return m_prod;
    }

    public int[][] path() {
        return m_path;
    }


    public static double score(double[][] w, double[] x) {

        int m = w.length;
        int n = x.length;

        if (w[0].length < n) {
            w = expand(w, n);
        }

        int i, j;
        double[][] scores = new double[m][n];
        scores[0][0] = w[0][0] * x[0];
        for (i = 1; i < m; i++) {
            scores[i][0] = scores[i - 1][0] + (w[i][0] * x[0]);
        }
        for (j = 1; j < n; j++) {
            scores[0][j] = scores[0][j - 1] + (w[0][j] * x[j]);
        }

        double max;
        for (i = 1; i < m; i++) {
            for (j = 1; j < n; j++) {
                max = scores[i - 1][j - 1];
                if (scores[i - 1][j] > max) {
                    max = scores[i - 1][j];
                }
                if (scores[i][j - 1] > max) {
                    max = scores[i][j - 1];
                }
                scores[i][j] = max + (w[i][j] * x[j]);
            }
        }
        return scores[m - 1][n - 1];
    }

    private void warp(double[][] w, double[] x) {

        final int U = 0;     // up
        final int R = 1;     // right
        final int D = 2;     // diagonal

        int m = w.length;
        int n = x.length;

        if (w[0].length < n) {
            w = expand(w, n);
        }

        int i, j;
        int[][] direction = new int[m][n];
        double[][] scores = new double[m][n];

        scores[0][0] = w[0][0] * x[0];
        for (i = 1; i < m; i++) {
            scores[i][0] = scores[i - 1][0] + (w[i][0] * x[0]);
            direction[i][0] = U;
        }
        for (j = 1; j < n; j++) {
            scores[0][j] = scores[0][j - 1] + (w[0][j] * x[j]);
            direction[0][j] = R;
        }

        double max;
        int dir;
        for (i = 1; i < m; i++) {
            for (j = 1; j < n; j++) {
                max = scores[i - 1][j - 1];
                dir = D;
                if (scores[i - 1][j] > max) {
                    max = scores[i - 1][j];
                    dir = U;
                }
                if (scores[i][j - 1] > max) {
                    max = scores[i][j - 1];
                    dir = R;
                }
                scores[i][j] = max + (w[i][j] * x[j]);
                direction[i][j] = dir;
            }
        }
        m_prod = scores[m - 1][n - 1];

        // compute path
        int wi = m - 1;
        int xj = n - 1;

        int[][] path = new int[m + n][2];
        path[0][0] = wi;
        path[0][1] = xj;

        int l = 1;
        dir = -1;
        while (wi != 0 || xj != 0) {
            if (wi == 0) {
                xj--;
            } else if (xj == 0) {
                wi--;
            } else {
                dir = direction[wi][xj];
                if (dir == U) {
                    wi--;
                } else if (dir == R) {
                    xj--;
                } else {
                    wi--;
                    xj--;
                }
            }
            path[l][0] = wi;
            path[l][1] = xj;
            l++;
        }

        // reverse order and trim
        m_path = new int[l][2];
        int k = l - 1;
        for (i = 0; i < l; i++) {
            m_path[i] = path[k - i];
        }
    }

    private static double[][] expand(double[][] w, int nx) {
        int m = w.length;
        int n = w[0].length;
        double[][] v = new double[m][nx];
        for (int i = 0; i < m; i++) {
            System.arraycopy(w[i], 0, v[i], 0, n);
        }
        return v;
    }
}
