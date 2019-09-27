package classifier.lvq;

import classifier.Classifier;
import data.Dataset;
import data.Pattern;
import distance.Distance;
import distance.ED;
import util.Array;
import util.Options;
import util.Rand;

/**
 * Generalized LVQ for feature vectors.
 */
public class GLVQ extends Classifier {

    private Rand rand;
    private Distance euclid;                // Euclidean distance
    private Parameter params;               // parameters
    private double[][] Y;                   // protoytpes
    private int[] ylabel;                   // class labels of prototypes

    public GLVQ(String opts) {
        rand = Rand.getInstance();
        params = new Parameter(opts);
        euclid = new ED();
    }

    @Override
    public Options getOptions() {
        return params.getOptions();
    }

    @Override
    public void setOptions(String opts) {
        params.setOptions(opts);
    }

    @Override
    public String getName() {
        return "GLVQ";
    }

    @Override
    public int predict(Pattern p) {
        int k = Y.length;
        int i0 = 0;
        double minDist = POSINF;
        for(int i = 0; i < k; i++) {
            double d = euclid.d(p.x, Y[i]);
            if(d < minDist) {
                minDist = d;
                i0 = i;
            }
        }
        return ylabel[i0];
    }

    public void fit(Dataset X) {

        // initialization
        initPrototypes(X);

        double[][] Yopt = Array.cp(Y);
        int numX = X.size();
        int numY = Y.length;

        // hyperparameters
        double g = params.g;
        int T = params.T;
        double eta = params.l;

        // monitoring
        int log = params.o;
        int maxStable = params.t;
        int numStableLoss = 0;
        int numStableErr = 0;
        double minLoss = POSINF;
        double minErr = POSINF;
        int t_minLoss = 0;
        int t_minErr = 0;
        boolean proceed = true;

        for (int t = 1; t <= T && proceed; t++) {

            int[] f = rand.shuffle(numX);
            for (int i = 0; i < numX; i++) {
                double[] x = X.pattern(f[i]);
                int xlabel = X.label(f[i]);

                double d1 = POSINF;
                double d2 = POSINF;
                int j1 = 0;
                int j2 = 0;
                for (int j = 0; j < numY; j++) {
                    double d = Math.pow(euclid.d(x, Y[j]), 2);
                    if (xlabel == ylabel[j]) {
                        if (d < d1) {
                            d1 = d;
                            j1 = j;
                        }
                    } else {
                        if (d < d2) {
                            d2 = d;
                            j2 = j;
                        }
                    }
                }
                eta = Math.log(t+1);
                double sum_d = d1 + d2;
                double mu = (d1 - d2) / sum_d;
                double sgn = 1.0 / (1.0 + Math.exp(-mu*g*eta));
                double delta = sgn * (1.0 - sgn) / (sum_d * sum_d);
                double delta1 = d2 * delta;
                double delta2 = -d1 * delta;
                update(x, Y[j1], delta1);
                update(x, Y[j2], delta2);
            }

            // check convergence
            numStableLoss++;
            numStableErr++;
            double loss = loss(X, eta);
            if(loss < minLoss) {
                if(loss < 0.999*minLoss) {
                    numStableLoss = 0;
                }
                minLoss = loss;
                t_minLoss = t;
                //numStableLoss = 0;
            }
            double err = eval(X);
            if (err < minErr) {
                minErr = err;
                t_minErr = t;
                numStableErr = 0;
                Yopt = Array.cp(Y);
            }
            proceed = 0.0 < minErr && numStableLoss <= maxStable;

            // print progress
            if(log == 1) {
                System.out.printf("\rEpoch %d", t);
            } else {
                System.out.printf("%5d loss = %7.5f (%6.5f @%d)  train = %5.2f (%3.2f @%d)%n",
                        t, loss, minLoss, t_minLoss, 100* err, 100* minErr, t_minErr);
            }

            // decrease learning rate if necessary
            if(t < 10 && 0 < numStableLoss && 0.0001 < eta) {
                t = 0;
                minLoss = POSINF;
                minErr = POSINF;
                numStableLoss = 0;
                numStableErr = 0;
                eta /= 2.0;
                initPrototypes(X);
            }

        }
        Y = Yopt;
    }

    public double[] update(double[] x, double[] y, double delta) {
        int n = x.length;
        for (int i = 0; i < n; i++) {
            y[i] = delta * x[i] + (1-delta) * y[i];
        }
        return y;
    }


    private double loss(Dataset X, double eta) {
        int numX = X.size();
        int numY = Y.length;
        double g = params.g;
        double loss = 0;
        for (int i = 0; i < numX; i++) {
            double[] x = X.pattern(i);
            int xlabel = X.label(i);
            double d1 = POSINF;
            double d2 = POSINF;
            for (int j = 0; j < numY; j++) {
                double d = Math.pow(euclid.d(x, Y[j]), 2);
                if (xlabel == ylabel[j]) {
                    if (d < d1) {
                        d1 = d;
                    }
                } else {
                    if (d < d2) {
                        d2 = d;
                    }
                }
            }
            loss += 1.0/(1.0+Math.exp(-g * eta * (d1 - d2) / (d1 + d2)));
        }
        return loss;
    }

    private void initPrototypes(Dataset X) {
        CodeBook cb = new CodeBook();
        Dataset Z = cb.codebook(X, params.k, true);
        Y = Z.patterns();
        ylabel = Z.labels();
    }
}
