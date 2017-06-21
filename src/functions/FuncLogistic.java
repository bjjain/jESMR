package functions;

/**
 * Created by jain on 28/03/2017.
 */
public class FuncLogistic implements Func {


    @Override
    public double[] apply(double[] a) {
        return new double[]{1.0 / (1.0 + Math.exp(-a[0]))};
    }

    @Override
    public double[] derivative(double[] z, int y) {
        return new double[]{z[0] - y};
    }

    @Override
    public double loss(double[] a, int y) {
        double z = 1.0 / (1.0 + Math.exp(-a[0]));
        z = y == 1 ? z : 1.0 - z;
        return -Math.log(z);
    }

    @Override
    public int predict(double[] a) {
        return 0 <= a[0] ? 1 : 0;
    }

    public String toString() {
        return "logistic regression";
    }
}
