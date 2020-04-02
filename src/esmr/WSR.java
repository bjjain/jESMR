package esmr;

import data.Dataset;
import data.Pattern;
import esmr.regularizer.Regularizer;
import util.Array;
import util.Options;
import util.Rand;

/**
 * Warped Softmax Regression.
 */
public class WSR extends Classifier {

    Rand rand;                                      // random number generator
    Parameter params;                               // parameters
    double[][][] W;                                 // weights
    double[][] Z;                                   // training examples
    int[] Y;                                        // class labels

    private int n;                                  // number of training examples
    private int d0;                                 // number of labels
    private int d1;                                 // elasticity
    private int d2;                                 // max length of time series


    public WSR(String opts) {
        rand = Rand.getInstance();
        params = new Parameter(opts);
    }

    @Override
    public Options getOptions() {
        return params.getOptions();
    }

    @Override
    public String getName() {
        return "Warped Softmax Regression";
    }

    @Override
    public int predict(Pattern p) {
        int i0 = 0;
        double z0 = WarpedProduct.score(W[0], p.x);
        for (int i = 1; i < d0; i++) {
            double z = WarpedProduct.score(W[i], p.x);
            if (z0 < z) {
                i0 = i;
                z0 = z;
            }
        }
        return i0;
    }

    @Override
    public void fit(Dataset X) {

        // initialize
        check(X);
        Z = X.patterns();
        Y = X.labels();
        d0 = X.numLabels();

        // dimensions
        n = X.size();                               // number of training examples
        d0 = X.numLabels();                         // number of classes
        d1 = Math.max(1, params.e);                 // elasticity
        d2 = X.maxlength();                         // max length of time series

        // hyper-parameters
        double lambda = params.r;                   // regularization parameter
        double eta = params.lr;                     // initial learning rate
        double b1 = params.b1;                      // first momentum
        double b2 = params.b2;                      // second momentum
        int T = params.T;                           // max number of epochs

        // auxiliary variables
        double[][][] M = new double[d0][d1][d2];    // first moment
        double[][][] V = new double[d0][d1][d2];    // second moment
        WarpedProduct[] P = new WarpedProduct[d0];  // warped products per class
        double[] out = new double[d0];              // output per class

        // initialize weights
        W = rand.nextArray(d0, d1, d2, Math.sqrt(d2));
        double[][][] optW = Array.cp(W);

        // logger
        Log log = new Log(params.T, params.S, params.o);

        // learn
        for (int t = 1; t <= T && log.proceed(); t++) {

            int[] f = rand.shuffle(n);
            for (int next = 0; next < n; next++) {

                // get next example
                double[] x = Z[f[next]];
                int y = Y[f[next]];

                // compute output
                int i0 = 0;
                for (int i = 0; i < d0; i++) {
                    P[i] = new WarpedProduct(W[i], x);
                    out[i] = P[i].score();
                    if (out[i0] < out[i]) {
                        i0 = i;
                    }
                }
                out = softmax(out, out[i0]);

                // update
                double[] delta = derivative(out, y);
                for (int i = 0; i < d0; i++) {
                    double[][] w = W[i];
                    int[][] path = P[i].path();
                    int len = path.length;
                    Regularizer reg = params.getRegularizer(w);
                    int r, s;
                    for (int l = 0; l < len; l++) {
                        r = path[l][0]; // index of w
                        s = path[l][1]; // index of x
                        double grad = delta[i] * x[s] + lambda * reg.derivative(r, s);
                        M[i][r][s] = (b1 * M[i][r][s] + (1 - b1) * grad);
                        V[i][r][s] = (b2 * V[i][r][s] + (1 - b2) * grad * grad);
                        double mean = M[i][r][s] / (1.0 - b1);
                        double var = V[i][r][s] / (1.0 - b2);
                        w[r][s] -= eta * (mean / (Math.sqrt(var) + 10E-8));
                    }
                }
            }

            // check convergence
            log.log(new double[]{loss(out), 100 * eval(X)}, t);
            if (log.hasImproved[1]) {
                optW = Array.cp(W);
            }

            // decrease learning rate if necessary
            if (log.decreaseLearningRate(t)) {
                t = 0;
                eta /= 2.0;
                W = rand.nextArray(d0, d1, d2, Math.sqrt(d2));
            }
        }
        W = optW;
    }

    private double[] softmax(double[] p, double max_p) {
        double sum = 0;
        int n = p.length;
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            z[i] = Math.exp(p[i] - max_p);
            sum += z[i];
        }
        for (int i = 0; i < n; i++) {
            z[i] /= sum;
        }
        return z;
    }

    private double[] derivative(double[] z, int y) {
        int n = z.length;
        double[] d = new double[n];
        for (int i = 0; i < n; i++) {
            d[i] = y == i ? -1 : 0;
            d[i] += z[i];
        }
        return d;
    }

    private double loss(double[] p) {
        int R = params.R;
        double lambda = params.r;
        double loss = 0;
        for (int next = 0; next < n; next++) {
            for (int i = 0; i < d0; i++) {
                WarpedProduct P = new WarpedProduct(W[i], Z[next]);
                p[i] = P.score();
                if (0 < R) {
                    int[][] path = P.path();
                    int len = path.length;
                    Regularizer reg = params.getRegularizer(W[i]);
                    for (int l = 0; l < len; l++) {
                        p[i] += lambda * reg.loss(path[l][0], path[l][1]);
                    }
                }
            }
            double sum = 0;
            double max_p = Array.max(p);
            for (int i = 0; i < d0; i++) {
                sum += Math.exp(p[i] - max_p);
            }
            double z = Math.exp(p[Y[next]] - max_p) / sum;
            if (z < EPS) {
                z = EPS;
            }
            loss -= Math.log(z);
        }
        return loss / n;
    }
}
