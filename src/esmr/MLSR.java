package esmr;

import data.Dataset;
import data.Pattern;
import util.Array;
import util.Options;
import util.Rand;

/**
 * Max-Linear Softmax Regression
 */
public class MLSR extends Classifier {

    Rand rand;                              // random number generator
    Parameter params;                       // parameters
    double[][][] W;                         // weights
    int numLabels;                          // number of class labels
    double[][] XX;                          // training set
    int[] yy;                               // class labels

    public MLSR(String opts) {
        rand = Rand.getInstance();
        params = new Parameter(opts);
    }

    @Override
    public Options getOptions() {
        return params.getOptions();
    }

    @Override
    public String getName() {
        return "Max-Linear Softmax Regression";
    }

    @Override
    public int predict(Pattern p) {
        int k0 = 0;
        double[] out = new double[numLabels];
        for (int k = 0; k < numLabels; k++) {
            double[] result = mult(W[k], p.x);
            out[k] = result[0];
            if (out[k0] < out[k]) {
                k0 = k;
            }
        }
        return k0;
    }

    @Override
    public void fit(Dataset X) {

        // initialize
        check(X);
        XX = X.patterns();
        yy = X.labels();
        numLabels = X.numLabels();

        // hyper-parameters
        int m = Math.max(1, params.e);      // elasticity
        double lambda = params.r;                 // weight decay
        double eta = params.lr;                   // initial learning rate
        double b1 = params.b1;                    // first momentum
        double b2 = params.b2;                    // second momentum
        int T = params.T;                         // max number of epochs

        // sizes
        int n = X.maxlength();                  // input dimension
        int c = numLabels;                        // number of classes
        int N = X.size();                       // number of training examples

        // auxiliary variables
        double[][][] M = new double[c][m][n];       // first moment
        double[][][] V = new double[c][m][n];       // second moment
        double[] h = new double[c];                 // values of optimal inner products
        int[] I = new int[c];                       // indices of optimal inner score
        double[] z = new double[c];                 // output

        // weights
        W = rand.nextArray(c, m, n, Math.sqrt(n));    // initialize weights
        double[][][] optW = Array.cp(W);                  // optimal weights

        // logger
        Log log = new Log(params.T, params.S, params.o);

        for (int t = 1; t <= T && log.proceed(); t++) {
            int[] f = rand.shuffle(N);
            for (int i = 0; i < N; i++) {

                // get next example
                double[] x = XX[f[i]];

                // compute output
                int jmax = 0;
                double[] res;
                for (int j = 0; j < c; j++) {
                    res = mult(W[j], x);
                    h[j] = res[0];
                    I[j] = (int) res[1];
                    if (h[jmax] < h[j]) {
                        jmax = j;
                    }
                }
                z = softmax(h, h[jmax]);

                // update
                double[] delta = derivative(z, yy[f[i]]);
                for (int j = 0; j < c; j++) {
                    int j0 = I[j];
                    double[] w = W[j][j0];
                    for (int k = 0; k < n; k++) {
                        double grad = delta[j] * x[k] + lambda * w[k];
                        M[j][j0][k] = (b1 * M[j][j0][k] + (1 - b1) * grad);
                        V[j][j0][k] = (b2 * V[j][j0][k] + (1 - b2) * grad * grad);
                        double mean = M[j][j0][k] / (1.0 - b1);
                        double var = V[j][j0][k] / (1.0 - b2);
                        w[k] -= eta * (mean / (Math.sqrt(var) + 10E-8));
                    }
                }
            }

            // check convergence
            log.log(new double[]{loss(z), 100 * eval(X)}, t);
            if (log.hasImproved[1]) {
                optW = Array.cp(W);
            }

            // decrease learning rate if necessary
            if (log.decreaseLearningRate(t)) {
                t = 0;
                eta /= 2.0;
                W = rand.nextArray(c, m, n, Math.sqrt(n));
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
        int n = p.length;
        int N = XX.length;
        double loss = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < numLabels; j++) {
                double[] prod = mult(W[j], XX[i]);
                p[j] = prod[0];
            }
            double sum = 0;
            double max_p = Array.max(p);
            for (int j = 0; j < n; j++) {
                sum += Math.exp(p[j] - max_p);
            }
            double z = Math.exp(p[yy[i]] - max_p) / sum;
            if (z < EPS) {
                z = EPS;
            }
            loss -= Math.log(z);
        }
        return loss / N;
    }

    // returns max-value and index of active function
    private double[] mult(double[][] W, double[] x) {
        int m = W.length;
        int n = x.length;
        double[] z = new double[m];
        int i0 = 0;
        for (int i = 0; i < m; i++) {
            z[i] = 0;
            for (int j = 0; j < n; j++) {
                z[i] += W[i][j] * x[j];
            }
            if (z[i0] < z[i]) {
                i0 = i;
            }
        }
        return new double[]{z[i0], i0};
    }

}
