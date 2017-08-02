package classifier;

import core.MaxAlignment;
import core.Options;
import data.Dataset;
import data.Pattern;
import functions.Func;
import functions.FuncSoftMax;
import util.Array;
import util.Msg;
import util.Rand;

/**
 * Created by jain on 24/03/2017.
 */
public class SemiMaxSoftmax extends Classifier {

    int m_numLabels;        // number of labels
    double[][][] m_W;       // weights
    double[] m_b;           // bias
    Func m_F;               // output function

    Rand m_random;
    Parameter m_params;

    public SemiMaxSoftmax(String opts) {
        m_random = Rand.getInstance();
        m_params = new Parameter(opts);
    }

    public Options getOptions() {
        return m_params.getOptions();
    }

    /* Returns state of convergence:
     *      -2 : loss diverges
     *      -1 : loss oscillates
     *       0 : loss decreases
     *       1 : loss converged
     */
    public int fit(Dataset train) {
        m_numLabels = train.numLabels();
        m_F = new FuncSoftMax();
        SSG ssg = new SSG(train);
        int S = ssg.fit();
        if (S < 0) {
            Msg.warn("Warning! Decrease learning rate.");
        }
        return S;
    }

    public int predict(Pattern x) {
        return m_F.predict(activate(x));
    }

    private double[] activate(Pattern x) {
        double[] a = new double[m_numLabels];
        for (int j = 0; j < m_numLabels; j++) {
            a[j] = MaxAlignment.sim(x.sequence(), m_W[j]) + m_b[j];
        }
        return a;
    }

    @FunctionalInterface
    interface Update {
        double apply(double grad, int j, int r, int s);
    }


    class SSG {

        // data
        double[][] X;       // data set
        int[] y;            // labels

        // hyper-parameters
        int type;           // type of solver
        int elasticity;     // elasticity
        double eta;         // current learning rate
        double mu;          // momentum
        double lambda;      // weight decay
        double rho1;        // decay rate 1
        double rho2;        // decay rate 2

        // sizes
        int inUnits;        // input size
        int outUnits;       // output size
        int numX;           // sample size

        // auxiliary variables
        double[][][] M;     // first moment
        double[][][] V;     // second moment
        double[] a;         // activate
        MaxAlignment[] A;      // alignment

        // update method
        Update ssg;

        // monitor class
        Monitor monitor;


        SSG(Dataset data) {

            // set data
            X = data.patterns();
            y = data.labels();

            // set monitor
            monitor = new Monitor(data);

            // set hyper-parameters
            type = m_params.A;
            elasticity = m_params.e;
            eta = m_params.l;
            mu = m_params.m;
            lambda = m_params.r;
            rho1 = m_params.r1;
            rho2 = m_params.r2;

            // set sizes
            inUnits = data.maxLength();
            outUnits = m_numLabels;
            numX = data.size();

            // set auxiliary variables
            if (type != Parameter.A_SGD) {
                M = new double[outUnits][inUnits][elasticity];
            }
            if (type == Parameter.A_ADAM) {
                V = new double[outUnits][inUnits][elasticity];
            }
            a = new double[outUnits];
            A = new MaxAlignment[outUnits];

            // initialize weights
            double sigma = Math.sqrt(inUnits);
            m_W = m_random.nextArray(outUnits, inUnits, elasticity, sigma);
            m_b = m_random.nextArray(outUnits, sigma);

            // set optimization technique
            setSSG();
        }

        int fit() {

            int T = m_params.T;
            int S = 0;
            for (int t = 1; t <= T && S == 0; t++) {
                int[] f = m_random.shuffle(numX);
                for (int i = 0; i < numX; i++) {

                    // get next example
                    double[] xi = X[f[i]];
                    int yi = y[f[i]];

                    // compute activate
                    for (int j = 0; j < outUnits; j++) {
                        A[j] = new MaxAlignment(xi, m_W[j]);
                        a[j] = A[j].sim() + m_b[j];
                    }

                    // update
                    double[] z = m_F.apply(a);
                    double[] delta = m_F.derivative(z, yi);
                    for (int j = 0; j < outUnits; j++) {
                        m_b[j] -= eta * delta[j];
                        double[][] wj = m_W[j];
                        int[][] path = A[j].path();
                        int len = path.length;
                        int r, s;
                        for (int l = 0; l < len; l++) {
                            r = path[l][0]; // max_idx of input x
                            s = path[l][1]; // max_idx of elasticity
                            double grad = delta[j] * xi[r] + lambda * wj[r][s];
                            wj[r][s] -= ssg.apply(grad, j, r, s);
                        }
                    }
                }
                S = monitor.check(t);
                monitor.info(t);
            }
            m_W = monitor.optWeights();
            return S;
        }


        private Update setSSG() {
            ssg = null;
            if (type == Parameter.A_SGD) {
                ssg = (grad, j, r, s) -> eta * grad;
            } else if (type == Parameter.A_MOMENTUM) {
                ssg = (grad, j, r, s) -> {
                    M[j][r][s] = mu * M[j][r][s] - eta * grad;
                    return -M[j][r][s];
                };
            } else if (type == Parameter.A_ADAGRAD) {
                ssg = (grad, j, r, s) -> {
                    M[j][r][s] += grad * grad;
                    return eta * grad / (Math.sqrt(M[j][r][s]) + 10E-8);
                };
            } else if (type == Parameter.A_ADADELTA) {
                ssg = (grad, j, r, s) -> {
                    M[j][r][s] = rho1 * M[j][r][s] + (1 - rho1) * grad * grad;
                    return eta * grad / (Math.sqrt(M[j][r][s]) + 10E-8);
                };
            } else if (type == Parameter.A_ADAM) {
                ssg = (grad, j, r, s) -> {
                    M[j][r][s] = (rho1 * M[j][r][s] + (1 - rho1) * grad);
                    V[j][r][s] = (rho2 * V[j][r][s] + (1 - rho2) * grad * grad);
                    double m = M[j][r][s] / (1.0 - rho1);
                    double v = V[j][r][s] / (1.0 - rho2);
                    return eta * m / (Math.sqrt(v) + 10E-8);
                };
            }
            return ssg;
        }
    }


    class Monitor {

        Dataset X;              // training set
        boolean loggable;       // output mode
        double curAcc;          // current accuracy
        double maxAcc;          // maximum accuracy
        double curLoss;         // current loss
        double minLoss;         // minimum loss
        int numStable;          // epochs without improvement
        int maxStable;          // maximum number of stable epochs
        double[][][] optW;      // optimal model


        Monitor(Dataset train) {
            X = train;
            loggable = 0 < m_params.o;
            curAcc = -1;
            maxAcc = 0;
            curLoss = -1;
            minLoss = Double.POSITIVE_INFINITY;
            numStable = 0;
            maxStable = m_params.S;
            optW = Array.cp(m_W);
        }

        int check(int t) {
            curAcc = 0;
            curLoss = 0;
            for (Pattern x : X) {
                int y = x.label();
                double[] a = activate(x);
                if (y == m_F.predict(a)) {
                    curAcc++;
                }
                curLoss += m_F.loss(a, y);
            }
            curAcc /= (double) X.size();
            curLoss /= (double) X.size();

            // check convergence
            if (!Double.isFinite(curLoss)) {
                return -2;

            }
            if (maxAcc <= curAcc) {
                maxAcc = curAcc;
            }
            if (curLoss < minLoss) {
                minLoss = curLoss;
                numStable = 0;
                optW = Array.cp(m_W);
            } else {
                numStable++;
            }
            double ratio = numStable / ((double) t);
            if (20 <= t && t <= 100 && 0.2 < ratio) {
                return -1;
            }
            return maxAcc < 1.0 && numStable < maxStable ? 0 : 1;
        }

        double[][][] optWeights() {
            return optW;
        }

        private void info(int t) {
            if (loggable) {
                String s = "[ESMR] %5d  loss = %7.5f (%7.5f)  train = %1.3f (%1.3f)%n";
                System.out.printf(s, t, curLoss, minLoss, curAcc, maxAcc);
            }
        }
    }
}
