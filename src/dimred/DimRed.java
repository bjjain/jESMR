package dimred;

import data.Dataset;
import data.Pattern;
import util.Array;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jain on 18.04.17.
 */
public class DimRed {

    public static void main(String[] args) {
        double[] x = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        DimRed dr = new DimRed();
        double[] y = dr.reduce(x, 0.11);
        System.out.println(Array.toString(y, "%5.3f "));
    }

    public Dataset reduce(Dataset X, double delta) {
        Dataset Y = new Dataset(X.numLabels());
        for (Pattern x : X) {
            double[] z = reduce(x.sequence(), delta);
            Y.add(new Pattern(z, x.label()));
        }
        return Y;
    }

    public double[] reduce(double[] x, double delta) {
        int m = 0;
        int n = x.length;
        double[] z = x;
        while (m != n) {
            z = diff(z, delta);
            m = n;
            n = z.length;
        }
        return z;
    }

    double[] diff(double[] x, double delta) {
        int n = x.length - 1;
        List<Double> y = new ArrayList<>();
        for (int i = 0; i < n; i += 2) {
            if (Math.abs(x[i] - x[i + 1]) < delta) {
                y.add((x[i] + x[i + 1]) / 2.0);
            } else {
                y.add(x[i]);
                y.add(x[i + 1]);
            }
        }
        if (n % 2 == 0) {
            if (0 < n && Math.abs(x[n] - x[n - 1]) < delta) {
                y.add((x[n] + x[n - 1]) / 2.0);
            } else {
                y.add(x[n]);
            }
        }
        n = y.size();
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            z[i] = y.get(i);
        }
        return z;
    }

}
