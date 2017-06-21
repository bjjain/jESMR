package mean;

import data.Dataset;
import data.Pattern;

/**
 * Created by jain on 12.04.17.
 */
public class Normalization {

    // distance function
    private final DTW dtw = new DTW();

    Mean m_mean;
    double[] m_mu;
    double[] m_std;

    public Normalization(String opts) {
        m_mean = new Mean(opts);
    }

    public void fit(Dataset X) {
        int n = X.maxLength();
        m_mu = new double[n];
        m_std = new double[n];
        double[][] x = getSequences(X);
        m_mu = m_mean.compute(x);
        m_std = std(x);
    }

    private double[][] getSequences(Dataset X) {
        int n = X.size();
        double[][] x = new double[n][];
        for (int i = 0; i < n; i++) {
            x[i] = X.get(i).sequence();
        }
        return x;
    }

    private double[] var(double[] x) {
        int n = m_mu.length;
        int[] v = new int[n];
        int[][] p = dtw.align(m_mu, x);
        int l = p.length;
        for (int j = 0; j < l; j++) {
            v[p[j][0]]++;
        }
        // align
        double[] z = new double[n];
        for (int j = 0; j < l; j++) {
            z[p[j][0]] += x[p[j][1]] / v[p[j][0]];
        }
        // compute variance
        double[] var = new double[n];
        for (int i = 0; i < n; i++) {
            var[i] = Math.pow(m_mu[i] - z[i], 2);
        }
        return var;
    }

    private double[] std(double[][] x) {
        int N = x.length;
        int n = m_mu.length;
        double[] std = new double[n];
        for (int i = 0; i < N; i++) {
            double[] var = var(x[i]);
            for (int j = 0; j < n; j++) {
                std[j] += var[j];
            }
        }
        for (int j = 0; j < n; j++) {
            std[j] = Math.sqrt(std[j] / ((double) N));
        }
        return std;
    }

    private double[] normalize(double[] x) {
        int n = m_mu.length;
        // aligned x
        double[] z = new double[n];
        // valence
        int[] v = new int[n];
        // path
        int[][] p = dtw.align(m_mu, x);
        int l = p.length;
        // get valences
        for (int j = 0; j < l; j++) {
            v[p[j][0]]++;
        }
        // align
        for (int j = 0; j < l; j++) {
            z[p[j][0]] += x[p[j][1]] / v[p[j][0]];
        }
        // normalize
        for (int i = 0; i < n; i++) {
            z[i] = (z[i] - m_mu[i]) / m_std[i];
        }
        return z;
    }

    public Dataset normalize(Dataset X) {
        if (X == null) {
            return null;
        }
        Dataset Y = new Dataset(X.numLabels());
        for (Pattern x : X) {
            double[] y = normalize(x.sequence());
            Y.add(new Pattern(y, x.label()));
        }
        return Y;
    }


    public double[] score(Dataset X) {
        if (X == null) {
            return null;
        }
        int n = X.size();
        m_mean.compute(getSequences(X));

        double[] var = new double[n];
        for (int i = 0; i < n; i++) {
            var[i] = dtw.d(m_mu, X.pattern(i));
        }
        return var;
    }
}
