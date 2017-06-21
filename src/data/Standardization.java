package data;

/**
 * Created by jain on 12.04.17.
 */
public class Standardization {

    double[] m_mu;
    double[] m_std;

    public void fit(Dataset X) {
        mean(X);
        std(X);
    }

    private void mean(Dataset X) {
        int m = X.size();
        int n = X.maxLength();
        m_mu = new double[n];
        for (int i = 0; i < m; i++) {
            double[] x = X.pattern(i);
            for (int j = 0; j < n; j++) {
                m_mu[j] += x[j];
            }
        }
        for (int j = 0; j < n; j++) {
            m_mu[j] /= (double) m;
        }
    }

    private void std(Dataset X) {
        int m = X.size();
        int n = X.maxLength();
        m_std = new double[n];
        for (int i = 0; i < m; i++) {
            double[] x = X.pattern(i);
            for (int j = 0; j < n; j++) {
                m_std[j] += Math.pow(m_mu[j] - x[j], 2);
            }
        }
        for (int j = 0; j < n; j++) {
            m_std[j] /= (double) m;
            m_std[j] = Math.sqrt(m_std[j]);
        }
    }

    private double[] normalize(double[] x) {
        int n = x.length;
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            z[i] = (x[i] - m_mu[i]) / (m_std[i] + 10E-10);
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

}
