package transform;

import data.Pattern;

/**
 * Z-Standardization.
 */
public class ZNormalize extends Transform {

    @Override
    Pattern transform(Pattern p) {
        int n = p.length();
        double[] x = p.x;
        double avg = mean(x);
        double std = std(x, avg);

        if (std < EPS) {
            std = EPS;
        }
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            z[i] = (x[i] - avg) / std;
        }
        return new Pattern(z, p.y);
    }

    private double mean(double[] x) {
        int n = x.length;
        double avg = x[0];
        for (int i = 1; i < n; i++) {
            avg += x[i];
        }
        return avg / ((double) n);
    }

    private double std(double[] x, double mean) {
        if (x == null || x.length == 0) {
            return 0;
        }
        double v = 0;
        int n = x.length;
        for (double aX : x) {
            v += Math.pow(aX - mean, 2);
        }
        return Math.sqrt(v / ((double) n));
    }

}