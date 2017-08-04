package classifier;

import core.Options;
import data.Dataset;
import data.Pattern;
import functions.Func;
import functions.FuncSoftMax;
import util.Array;
import util.Msg;
import util.Rand;

public class MaxLinSoftmax extends Classifier {

    int m_numLabels;        // number of labels
    double[][][] m_W;       // weight time series
    Func m_F;               // output function
    Parameter m_params;     // parameters
    Status m_status;        // state of algorithm
    Rand m_random;          // random number generator


    public MaxLinSoftmax(String opts) {
        m_random = Rand.getInstance();
        m_params = new Parameter(opts);
    }

    public Options getOptions() {
        return m_params.getOptions();
    }

    public Status fit(Dataset train) {
        m_numLabels = train.numLabels();
        m_F = new FuncSoftMax();
        m_status = new Status(m_params.S);
        SSG ssg = new SSG(train);
        ssg.fit();
        return m_status;
    }

    public int predict(Pattern x) {
        return m_F.predict(activate(x));
    }

    private double[] activate(Pattern x) {
        double[] a = new double[m_numLabels];
        for (int j = 0; j < m_numLabels; j++) {
            double[] y = mult(m_W[j], x.sequence());
            a[j] = Array.max(y);
        }
        return a;
    }

    private double[] mult(double[][] W, double[] x) {
        int e = W.length;
        int d = x.length;
        double[] y = new double[e];
        for (int i = 0; i < e; i++) {
            y[i] = 0;
            for (int j = 0; j < d; j++) {
                y[i] += W[i][j] * x[j];
            }
        }
        return y;
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
        double[][] a;       // activation
        double[] max_a;     // max activation
        int[] max_idx;      // index of max activation
        Update ssg;         // update method


        SSG(Dataset data) {

            // set data
            X = data.patterns();
            y = data.labels();

            // set hyper-parameters
            type = m_params.A;
            elasticity = (int) Math.max(1, m_params.e);
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
                M = new double[outUnits][elasticity][inUnits];
            }
            if (type == Parameter.A_ADAM) {
                V = new double[outUnits][elasticity][inUnits];
            }
            a = new double[outUnits][elasticity];
            max_a = new double[outUnits];
            max_idx = new int[outUnits];

            // initialize weights
            double sigma = Math.sqrt(inUnits);
            m_W = m_random.nextArray(outUnits, elasticity, inUnits, sigma);

            // set optimization technique
            setSSG();
        }

        void fit() {
            // optimal model
            double[][][] optW = Array.cp(m_W);
            // output mode
            boolean loggable = 0 < m_params.o;
            // max iterations
            int T = m_params.T;
            for (int t = 1; t <= T && m_status.state == 0; t++) {
                int[] f = m_random.shuffle(numX);
                for (int i = 0; i < numX; i++) {

                    // get next example
                    double[] xi = X[f[i]];
                    int yi = y[f[i]];

                    // update
                    update(xi, yi);
                }
                // check convergence
                double[] result = eval();
                m_status.set(result[0], result[1], t);

                // store best model
                if (m_status.updateModel) {
                    optW = Array.cp(m_W);
                }

                // progress info
                if (loggable) {
                    m_status.info(t);
                }
            }
            m_W = optW;
            if (m_status.decrLearningRate()) {
                Msg.warn("Warning! Decrease learning rate.");
            }
        }

        private double[] eval() {
            double acc = 0;
            double loss = 0;
            double[] a = new double[m_numLabels];
            for (int i = 0; i < numX; i++) {
                for (int j = 0; j < m_numLabels; j++) {
                    double[] y = mult(m_W[j], X[i]);
                    a[j] = Array.max(y);
                }
                if (y[i] == m_F.predict(a)) {
                    acc++;
                }
                loss += m_F.loss(a, y[i]);
            }
            acc /= (double) numX;
            loss /= (double) numX;
            return new double[]{acc, loss};
        }

        private void update(double[] x, int y) {
            for (int j = 0; j < outUnits; j++) {
                a[j] = mult(m_W[j], x);
                max_idx[j] = Array.indexOfMax(a[j]);
                max_a[j] = a[j][max_idx[j]];
            }
            double[] z = m_F.apply(max_a);
            double[] delta = m_F.derivative(z, y);

            for (int j = 0; j < outUnits; j++) {
                double[][] wj = m_W[j];
                int e = max_idx[j];
                for (int d = 0; d < inUnits; d++) {
                    double grad = delta[j] * x[d] + lambda * wj[e][d];
                    wj[e][d] -= ssg.apply(grad, j, e, d);
                }
            }
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
}
