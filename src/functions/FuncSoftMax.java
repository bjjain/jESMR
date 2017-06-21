package functions;

import util.Array;

/**
 * Created by jain on 28/03/2017.
 */
public class FuncSoftMax implements Func {

    @Override
    public double[] apply(double[] a) {
        int n = a.length;
        double[] z = new double[n];
        int i0 = Array.indexOfMax(a);
        double max = a[i0];
        double sum = 0;
        for (int i = 0; i < n; i++) {
            z[i] = Math.exp(a[i] - max);
            sum += z[i];
        }
        for (int i = 0; i < n; i++) {
            z[i] /= sum;
        }
        return z;
    }

    @Override
    public double[] derivative(double[] z, int y) {
        int n = z.length;
        double[] d = new double[n];
        for (int i = 0; i < n; i++) {
            d[i] = y == i ? -1 : 0;
            d[i] += z[i];
        }
        return d;
    }

    @Override
    public int predict(double[] a) {
        return Array.indexOfMax(a);
    }

    @Override
    public double loss(double[] a, int y) {
        int i0 = Array.indexOfMax(a);
        double max = a[i0];
        double sum = 0;
        int n = a.length;
        for (int i = 0; i < n; i++) {
            sum += Math.exp(a[i] - max);
        }
        return -Math.log(Math.exp(a[y] - max) / sum);
    }

    public String toString() {
        return "classifier/softmax";
    }
}
