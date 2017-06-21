package functions;

/**
 * Created by jain on 24/03/2017.
 */

public interface Func {

    /**************************************
     * double[] a       input a = W * x
     * double[] z       output z = f(x)
     * int y            class label
     **************************************/

    double[] apply(double[] a);

    double[] derivative(double[] z, int y);

    int predict(double[] a);

    double loss(double[] a, int y);
}

