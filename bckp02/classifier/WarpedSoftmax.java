package classifier;

import core.Alignment;
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
public class WarpedSoftmax extends Classifier {

    int m_numLabels;        // number of labels
    double[][] m_W;         // weight time series
    Func m_F;               // output function
    Status m_status;        // state of algorithm

    Rand m_random;
    Parameter m_params;

    public WarpedSoftmax(String opts) {
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
        if (m_status.decrLearningRate()) {
            Msg.warn("Warning! Decrease learning rate.");
        }
        return m_status;
    }

    public int predict(Pattern x) {
        return m_F.predict(activate(x));
    }

    private double[] activate(Pattern x) {
        double[] a = new double[m_numLabels];
        for (int j = 0; j < m_numLabels; j++) {
            a[j] = Alignment.sim(x.sequence(), m_W[j]);
        }
        return a;
    }

    @FunctionalInterface
    interface Update {
        double apply(double grad, int j, int s);
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
        double[][] M;       // first moment
        double[][] V;       // second moment
        double[] a;         // activate
        Alignment[] A;      // alignment
        Update ssg;         // update method


        SSG(Dataset data) {

            // set data
            X = data.patterns();
            y = data.labels();

            // set hyper-parameters
            type = m_params.A;
            elasticity = (int)Math.max(1, m_params.e * data.maxLength());
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
                M = new double[outUnits][elasticity];
            }
            if (type == Parameter.A_ADAM) {
                V = new double[outUnits][elasticity];
            }
            a = new double[outUnits];
            A = new Alignment[outUnits];

            // initialize weights
            double sigma = Math.sqrt(inUnits);
            m_W = m_random.nextArray(outUnits, elasticity, sigma);

            // set optimization technique
            setSSG();
        }

        void fit() {
            // optimal model
            double[][] optW = Array.cp(m_W);
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

                    // compute activate
                    for (int j = 0; j < outUnits; j++) {
                        A[j] = new Alignment(xi, m_W[j]);
                        a[j] = A[j].sim();
                    }

                    // update
                    double[] z = m_F.apply(a);
                    double[] delta = m_F.derivative(z, yi);
                    for (int j = 0; j < outUnits; j++) {
                        double[] wj = m_W[j];
                        int[][] path = A[j].path();
                        int len = path.length;
                        int r, s;
                        for (int l = 0; l < len; l++) {
                            r = path[l][0]; // index of input x
                            s = path[l][1]; // index of weight w
                            double grad = delta[j] * xi[r] + lambda * wj[s];
                            wj[s] -= ssg.apply(grad, j, s);
                        }
                    }
                }

                // check convergence
                double acc = 0;
                double loss = 0;
                for (int i = 0; i < numX; i++) {
                    for (int j = 0; j < m_numLabels; j++) {
                        a[j] = Alignment.sim(X[i], m_W[j]);
                    }
                    if (y[i] == m_F.predict(a)) {
                        acc++;
                    }
                    loss += m_F.loss(a, y[i]);
                }
                acc /= (double) numX;
                loss /= (double) numX;
                m_status.set(acc, loss, t);
                if(m_status.updateModel) {
                    optW = Array.cp(m_W);
                }
            }
            m_W = optW;
        }


        private Update setSSG() {
            ssg = null;
            if (type == Parameter.A_SGD) {
                ssg = (grad, j, s) -> eta * grad;
            } else if (type == Parameter.A_MOMENTUM) {
                ssg = (grad, j, s) -> {
                    M[j][s] = mu * M[j][s] - eta * grad;
                    return -M[j][s];
                };
            } else if (type == Parameter.A_ADAGRAD) {
                ssg = (grad, j, s) -> {
                    M[j][s] += grad * grad;
                    return eta * grad / (Math.sqrt(M[j][s]) + 10E-8);
                };
            } else if (type == Parameter.A_ADADELTA) {
                ssg = (grad, j, s) -> {
                    M[j][s] = rho1 * M[j][s] + (1 - rho1) * grad * grad;
                    return eta * grad / (Math.sqrt(M[j][s]) + 10E-8);
                };
            } else if (type == Parameter.A_ADAM) {
                ssg = (grad, j, s) -> {
                    M[j][s] = (rho1 * M[j][s] + (1 - rho1) * grad);
                    V[j][s] = (rho2 * V[j][s] + (1 - rho2) * grad * grad);
                    double m = M[j][s] / (1.0 - rho1);
                    double v = V[j][s] / (1.0 - rho2);
                    return eta * m / (Math.sqrt(v) + 10E-8);
                };
            }
            return ssg;
        }
    }
}
