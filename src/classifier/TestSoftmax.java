package classifier;

import core.Alignment;
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
public class TestSoftmax {

    int m_numLabels;        // number of labels
    int m_outUnits;         // number of output units
    double[][][] m_W;       // weights
    Func m_F;               // output function
    int[][][] m_C;          // update counter

    Rand m_random;
    Parameter m_params;

    public TestSoftmax(String opts) {
        m_random = Rand.getInstance();
        m_params = new Parameter(opts);
    }

    private TestSoftmax(double[][][] W) {
        m_W = Array.cp(W);
    }

    public void fit(Dataset X) {
        fit(X, null);
    }

    public void fit(Dataset train, Dataset val) {

        initialize(train, val);

        SGD sgd = new SGD(train);
        Monitor monitor = new Monitor(train, val);
        int T = m_params.T;
        boolean proceed = true;
        for (int t = 1; t <= T && proceed; t++) {
            sgd.fit(t);
            proceed = monitor.check(t);
        }
        m_W = monitor.optWeights();
    }

    public int predict(Pattern x) {
        return m_F.predict(otimes(x));
    }

    public double score(Dataset X) {
        int acc = 0;
        for(Pattern x : X) {
            if(x.label() == predict(x)) {
                acc++;
            }
        }
        return ((double)acc)/X.size();
    }

    public double loss(Dataset X) {
        double loss = 0;
        for(Pattern x : X) {
            loss += m_F.loss(otimes(x), x.label());
        }
        loss /= (double) X.size();
        return loss;
    }

    public int[][][] updateCounter() {
        return m_C;
    }

    private void initialize(Dataset train, Dataset val) {
        m_numLabels = train.numLabels();
        m_outUnits = m_numLabels;
        m_F = new FuncSoftMax();
    }

    private double[] otimes(Pattern x) {
        double[] a = new double[m_outUnits];
        for(int j = 0; j < m_outUnits; j++) {
            a[j] = Alignment.sim(x.sequence(), m_W[j]);
        }
        return a;
    }

    @FunctionalInterface
    interface Update {
        double apply(double grad, int j, int r, int s);
    }


    class SGD {

        // data
        double[][] X;       // data set
        int[] y;            // labels

        // hyper-parameters
        int type;           // type of solver
        int nh_T;           // epochs neighborhood is active
        int nh_w;           // neighborhood width
        double nh_s;        // neighborhood sigma
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
        double[] a;         // otimes
        Alignment[] A;      // alignment

        // update method
        Update sgd;


        SGD(Dataset data) {

            // set data
            X = data.patterns();
            y = data.labels();

            // set hyper-parameters
            type = m_params.A;
            nh_T = m_params.W;
            nh_w = m_params.w;
            nh_s = m_params.s;
            elasticity = m_params.e;
            eta = m_params.l;
            mu = m_params.m;
            lambda = m_params.r;
            rho1 = m_params.r1;
            rho2 = m_params.r2;

            // set sizes
            inUnits = data.maxLength();
            outUnits = m_outUnits;
            numX = data.size();

            // set auxiliary variables
            m_C = new int[outUnits][inUnits][elasticity];
            if(type != Parameter.A_SGD) {
                M = new double[outUnits][inUnits][elasticity];
            }
            if(type == Parameter.A_ADAM) {
                V = new double[outUnits][inUnits][elasticity];
            }
            a = new double[outUnits];
            A = new Alignment[outUnits];

            // initialize weights
            double sigma = Math.sqrt(inUnits);
            m_W = m_random.nextArray(outUnits, inUnits, elasticity, sigma);

            // set optimization technique
            setSGD();
        }

        void fit(int t) {

            m_C = new int[outUnits][inUnits][elasticity];

            int[] f = m_random.shuffle(numX);
            for (int i = 0; i < numX; i++) {

                // get next example
                double[] xi = X[f[i]];
                int yi = y[f[i]];

                // compute otimes
                for(int j = 0; j < outUnits; j++) {
                    A[j] = new Alignment(xi, m_W[j]);
                    a[j] = A[j].sim();
                }

                // update
                if(t < nh_T) {
                    update(xi, yi, nh_s*t, nh_w);
                } else {
                    update(xi, yi);
                }
            }
        }

        private void update(double[] x, int y) {
            double[] z = m_F.apply(a);
            double[] delta = m_F.derivative(z, y);
            for (int j = 0; j < outUnits; j++) {
                double[][] wj = m_W[j];
                int[][] path = A[j].path();
                int len = path.length;
                int r, s;
                for (int l = 0; l < len; l++) {
                    r = path[l][0]; // max_idx of input x
                    s = path[l][1]; // max_idx of elasticity
                    double grad = delta[j] * x[r] + lambda * wj[r][s];
                    wj[r][s] -= sgd.apply(grad, j, r, s);
                    m_C[j][r][s]++;
                }
            }
        }

        private void update(double[] x, int y, double sigma, int width) {

            double[] z = m_F.apply(a);
            double[] delta = m_F.derivative(z, y);

            int r, r_prev , s, s0, s1;
            int counter;
            for (int j = 0; j < outUnits; j++) {

                boolean[][] isPath = new boolean[inUnits][elasticity];
                int[] degree = new int[inUnits];

                int[][] path = A[j].path();
                int len = path.length;
                for (int l = 0; l < len; l++) {
                    r = path[l][0];         // max_idx of input x
                    s = path[l][1];         // max_idx of elasticity
                    isPath[r][s] = true;
                    degree[r]++;
                }

                r_prev = 0;
                counter = 0;
                for (int l = 0; l < len; l++) {
                    r = path[l][0];         // max_idx of input x
                    s = path[l][1];         // max_idx of elasticity
                    double grad = delta[j] * x[r] + lambda * m_W[j][r][s];
                    grad = sgd.apply(grad, j, r, s);
                    m_W[j][r][s] -= grad;
                    m_C[j][r][s]++;

                    if(r_prev == r) {
                        counter++;
                    } else {
                        r_prev = r;
                        counter = 1;
                    }
                    if(1 == degree[r]) {
                        s0 = Math.max(0, s-width);
                        s1 = Math.min(s+width+1, elasticity);
                    } else {
                        if(counter == 1) {
                            s0 = Math.max(0, s-width);;
                            s1 = s;
                        } else if(counter == degree[r]) {
                            s0 = Math.min(s+1, elasticity);
                            s1 = Math.min(s+width+1, elasticity);;
                        } else {
                            s0 = 0;
                            s1 = 0;
                        }
                    }
                    for(int i = s0; i < s1; i++) {
                        if(isPath[r][i]) {
                            continue;
                        }
                        m_W[j][r][i] -= grad*Math.exp(-Math.abs(s-i)*sigma);
                    }
                }
            }
        }

        private void setSGD() {
            if(type == Parameter.A_SGD) {
                sgd = (grad, j, r, s) -> eta * grad;
            } else if(type == Parameter.A_MOMENTUM) {
                sgd = (grad, j, r, s) -> {
                    M[j][r][s] = mu * M[j][r][s] - eta * grad;
                    return -M[j][r][s];
                };
            } else if(type == Parameter.A_ADAGRAD) {
                sgd = (grad, j, r, s) -> {
                    M[j][r][s] += grad * grad;
                    return eta * grad / (Math.sqrt(M[j][r][s]) + 10E-8);
                };
            } else if(type == Parameter.A_ADADELTA) {
                sgd = (grad, j, r, s) -> {
                    M[j][r][s] = rho1 * M[j][r][s] + (1 - rho1) * grad * grad;
                    return eta * grad / (Math.sqrt(M[j][r][s]) + 10E-8);
                };
            } else if(type == Parameter.A_ADAM) {
                sgd = (grad, j, r, s) -> {
                    M[j][r][s] = (rho1 * M[j][r][s] + (1 - rho1) * grad);
                    V[j][r][s] = (rho2 * V[j][r][s] + (1 - rho2) * grad * grad);
                    double m = M[j][r][s] / (1.0 - rho1);
                    double v = V[j][r][s] / (1.0 - rho2);
                    return eta * m / (Math.sqrt(v) + 10E-8);
                };
            }
        }
    }


    class Monitor {
        Dataset X;              // training set
        Dataset Z;              // validation set
        boolean loggable;       // output mode
        double curAcc;          // current accuracy
        double maxAcc;          // maximum accuracy
        double curTestAcc;      // current test accuracy
        double optTestAcc;      // test accuracy at maxAcc
        double curLoss;         // current loss
        double minLoss;         // minimum loss
        int numStable;          // epochs without improvement
        int maxStable;          // maximum number of stable epochs
        TestSoftmax optModel;       // optimal model

        Monitor(Dataset train, Dataset val) {
            X = train;
            Z = val;
            loggable = 0 < m_params.o;
            curAcc = -1;
            maxAcc = 0;
            curTestAcc = -1;
            optTestAcc = -1;
            curLoss = -1;
            minLoss = Double.POSITIVE_INFINITY;
            numStable = 0;
            maxStable = m_params.S;
            optModel = new TestSoftmax(m_W);
        }

        boolean check(int t) {
            // check convergence
            curLoss = loss(X);
            if(!Double.isFinite(curLoss)) {
                Msg.error("Error! Loss is not a number.");
            }
            curAcc = score(X);
            if (maxAcc <= curAcc) {
                maxAcc = curAcc;
            }
            if(curLoss < minLoss) {
                minLoss = curLoss;
                numStable = 0;
                if(Z != null) {
                    curTestAcc = score(Z);
                    optTestAcc = curTestAcc;
                    optModel.m_W = Array.cp(m_W);
                }
            } else {
                numStable++;
            }

            // report
            if(loggable) {
                info(t);
            }

            double ratio = numStable / ((double)t);
            if(20 <= t && t <= 100 && 0.2 < ratio) {
                Msg.warn("Warning! Decrease learning rate.");
                System.out.println("Elasticity = " + m_W[0][0].length);
                System.exit(0);
            }

            return maxAcc < 1.0 && numStable < maxStable;
        }

        double[][][] optWeights() {
            return optModel.m_W;
        }

        private void info(int t) {
            String s = "[ESMR] %5d  loss = %7.5f (%7.5f)  test = %1.3f (%1.3f)  train = %1.3f (%1.3f)%n";
            System.out.printf(s, t, curLoss, minLoss, curTestAcc, optTestAcc, curAcc, maxAcc);
        }
    }

}
