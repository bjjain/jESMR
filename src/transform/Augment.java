package transform;

import data.Pattern;

/**
 * Augments time series with bias at first position.
 */
public class Augment extends Transform {

    double bias;

    public Augment(double b) {
        bias = b;
    }

    @Override
    Pattern transform(Pattern p) {
        int n = p.length();
        double[] x = new double[n + 1];
        System.arraycopy(p.x, 0, x, 1, n);
        x[0] = bias;
        return new Pattern(x, p.y);
    }
}

