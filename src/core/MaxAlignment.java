package core;

public final class MaxAlignment {

    double[][] m_scores;
    int[][] m_direction;
    private int[][] m_path;
    private double m_sim;


    public MaxAlignment(double[] x, double[][] w) {
        m_sim = computeSimilarity(x, w);
        m_path = computePath();
    }

    public static double sim(double[] x, double[][] w) {
        int xlen = x.length;
        int wlen = w[0].length;

        if(xlen == 0 || wlen == 0) {
            return 0;
        }

        if(xlen > w.length) {
            w = stretch(w, xlen);
        }

        int i, j;
        double[][] scores = new double[xlen][wlen];
        scores[0][0] = x[0]*w[0][0];
        for(i = 1; i < xlen; i++) {
            scores[i][0] = scores[i-1][0] + x[i]*w[i][0];
        }
        for(j = 1; j < wlen; j++) {
            scores[0][j] = scores[0][j-1] + x[0]*w[0][j];
        }

        double max;
        for (i = 1; i < xlen; i++) {
            for (j = 1; j < wlen; j++) {
                max = scores[i-1][j];
                if(scores[i][j-1] > max) {
                    max = scores[i][j-1];
                }
                if(scores[i-1][j-1] > max) {
                    max = scores[i-1][j-1];
                }
                scores[i][j] = max + x[i]*w[i][j];
            }
        }
        return scores[xlen-1][wlen -1];
    }

    private static double[][] stretch(double[][] y, int nx) {
        int n = y.length;
        int m = y[0].length;
        double[][] _y = new double[nx][m];
        for (int i = 0; i < n; i++) {
            System.arraycopy(y[i], 0, _y[i], 0, m);
        }
        return _y;
    }

    public double sim() {
        return m_sim;
    }

    public int[][] path() { return m_path; }

    private double computeSimilarity(double[] x, double[][] w) {

        int xlen = x.length;
        int wlen = w[0].length;

        if(xlen == 0 || wlen == 0) {
            return 0;
        }

        if(xlen > w.length) {
            w = stretch(w, xlen);
        }

        int i, j;
        m_direction = new int[xlen][wlen];
        m_scores = new double[xlen][wlen];
        m_scores[0][0] = x[0]*w[0][0];
        for(i = 1; i < xlen; i++) {
            m_scores[i][0] = m_scores[i-1][0] + x[i]*w[i][0];
            m_direction[i][0] = 0;
        }
        for(j = 1; j < wlen; j++) {
            m_scores[0][j] = m_scores[0][j-1] + x[0]*w[0][j];
            m_direction[0][j] = 1;
        }

        double max;
        int i0;
        for (i = 1; i < xlen; i++) {
            for (j = 1; j < wlen; j++) {
                max = m_scores[i-1][j];
                i0 = 0;
                if(m_scores[i][j-1] > max) {
                    max = m_scores[i][j-1];
                    i0 = 1;
                }
                if(m_scores[i-1][j-1] > max) {
                    max = m_scores[i-1][j-1];
                    i0 = 2;
                }
                m_scores[i][j] = max + x[i]*w[i][j];
                m_direction[i][j] = i0;
            }
        }
        return m_scores[xlen-1][wlen -1];
    }

    private int[][] computePath() {
        final int LEFT 	= 0;
        final int UP 	= 1;

        int xlen = m_scores.length;
        int wlen = m_scores[0].length;
        int ix = xlen - 1;
        int iw = wlen - 1;

        int[][] path = new int[xlen + wlen][2];
        path[0][0] = ix;
        path[0][1] = iw;

        int n = 1;
        int direction;
        while(ix != 0 || iw != 0) {

            if(ix == 0) {
                iw--;
            } else if(iw == 0) {
                ix--;
            } else {
                direction = m_direction[ix][iw];
                if(direction == LEFT) {
                    ix--;
                } else if(direction == UP) {
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
        int m = n-1;
        for(int i = 0; i < n; i++) {
            p[i] = path[m-i];
        }
        return p;
    }

}
