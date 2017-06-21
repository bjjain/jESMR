import core.Alignment;
import data.Dataset;
import data.Pattern;
import functions.LossFunc;
import functions.LossSoftmax;
import classifier.softmax.Parameter;
import util.Array;
import util.Msg;
import util.Rand;

/**
 * Created by jain on 24/03/2017.
 */
public class Softmax {

    int m_numLabels;        // number of labels
    double[][][] m_W;       // weights
    double[] m_b;           // bias
    LossFunc m_loss;        // loss function

    Rand m_random;
    Parameter m_params;

    public Softmax(String opts) {
        m_loss = new LossSoftmax();
        m_random = Rand.getInstance();
        m_params = new Parameter(opts);
    }

    private Softmax(double[][][] W, double[] b) {
        m_W = Array.cp(W);
        m_b = Array.cp(b);
    }

    public void fit(Dataset X) {
        fit(X, null);
    }


    public void fit(Dataset train, Dataset val, int bs) {

        // set data
        double[][] X = train.patterns();
        int[] labels = train.labels();

        // initialize hyper-parameters
        int elasticity = m_params.e;                // elasticity
        double eta = m_params.l;                    // current learning rate
        double mu = m_params.m;                     // momentum
        double lambda = m_params.r;                 // regularization
        int T = m_params.T;                         // max epochs


        // set dimensions
        int inUnits = train.maxLength();                // input dimension
        int outUnits = train.numLabels();               // output dimension
        m_numLabels = outUnits;                         // number of labels
        int numX = train.size();                        // sample size

        // set batch size
        bs = bs <= 0? numX : bs;
        int numB = numX/bs;

        // initialize weights
        double sigma = Math.sqrt(inUnits);
        m_W = m_random.nextArray(outUnits, inUnits , elasticity, sigma);
        m_b = m_random.nextArray(outUnits, sigma);

        // momentum
        double[][][] v = new double[outUnits][inUnits][elasticity];

        // monitoring
        boolean loggable = 0 < m_params.o;          // output mode
        double curAcc;                              // current accuracy
        double maxAcc = 0;                          // maximum accuracy
        double curTestAcc = -1;                     // current test accuracy
        double optTestAcc = -1;                     // test accuracy at maxAcc
        double curLoss;                             // current loss
        double minLoss = Double.POSITIVE_INFINITY;  // minimum loss
        int numStable = 0;                          // epochs without improvement
        Softmax optModel = new Softmax(m_W, m_b);   // optimal model


        // learn
        for (int t = 1; t <= T && maxAcc < 1 && numStable < T; t++) {

            int[] f = m_random.shuffle(numX);

            for (int b = 0; b < numB; b++) {

                int[] y = new int[bs];                            // labels
                double[][] x = new double[bs][];                  // training examples
                double[][] a = new double[bs][outUnits];          // activation
                double[][] z = new double[bs][outUnits];          // output
                double[][] delta = new double[bs][outUnits];      // error
                Alignment[][] A = new Alignment[bs][outUnits];    // alignments
                LossSoftmax[] h = new LossSoftmax[bs];            // loss function

                // forward propagation
                for (int i = 0; i < bs; i++) {

                    // get example
                    int i0 = f[b * bs + i];
                    x[i] = X[i0];
                    y[i] = labels[i0];

                    // compute activation
                    for (int j = 0; j < outUnits; j++) {
                        A[i][j] = new Alignment(x[i], m_W[j]);
                        a[i][j] = A[i][j].sim() + m_b[j];
                    }

                    // compute output
                    h[i] = new LossSoftmax();
                    z[i] = h[i].apply(a[i]);
                }

                // backward propagation
                double[][] grad_w = new double[inUnits][elasticity];
                double[] grad_b = new double[outUnits];
                for (int i = 0; i < bs; i++) {

                    // compute error
                    double[] di = h[i].derivative(y[i]);
                    for(int j = 0; j < outUnits; j++) {
                        delta[i][j] += di[j];
                    }

                    // compute gradients
                    for (int j = 0; j < outUnits; j++) {
                        grad_b[j] += delta[i][j];
                        int[][] path = A[i][j].path();
                        int len = path.length;
                        int r, s;
                        for (int l = 0; l < len; l++) {
                            r = path[l][0];                // index of input x
                            s = path[l][1];                // index of elasticity
                            grad_w[r][s] += delta[i][j] * x[i][r];
                        }
                    }
                }

                // apply weights
                for (int i = 0; i < outUnits; i++) {
                    double[][] vj = v[i];
                    double[][] wj = m_W[i];
                    for (int r = 0; r < inUnits; r++) {
                        for (int s = 0; s < elasticity; s++) {
                            vj[r][s] = mu * vj[r][s] - eta * (grad_w[r][s] / bs + lambda * wj[r][s]);
                            wj[r][s] += vj[r][s];
                        }
                    }
                    m_b[i] -= eta * grad_b[i] / bs;
                }
            }


            // apply learning rate
            //eta = m_params.l *(T-t) / ((double)T);
            //eta = Math.max(eta, m_params.l/20.0);

            // check convergence
            curLoss = loss(train);
            if(!Double.isFinite(curLoss)) {
                Msg.error("Error! Loss is not a number.");
            }
            curAcc = score(train);
            if (maxAcc <= curAcc) {
                maxAcc = curAcc;
            }
            if(curLoss < minLoss) {
                minLoss = curLoss;
                numStable = 0;
                if(val != null) {
                    curTestAcc = score(val);
                    optTestAcc = curTestAcc;
                    optModel.m_W = Array.cp(m_W);
                    optModel.m_b = Array.cp(m_b);
                }
            } else {
                numStable++;
            }

            // report
            if(loggable) {
                info(t, curLoss, minLoss, curTestAcc, optTestAcc, curAcc, maxAcc);
            }
            //maxvalue();
        }
        m_W = optModel.m_W;
        m_b = optModel.m_b;
    }

    private void info(int t, double curLoss, double minLoss, double curTestAcc,
                      double optTestAcc, double curAcc, double maxAcc) {
        String s = "[CNN] %5d  loss = %7.5f (%7.5f)  test = %1.3f (%1.3f)  train = %1.3f (%1.3f)%n";
        System.out.printf(s, t, curLoss, minLoss, curTestAcc, optTestAcc, curAcc, maxAcc);
    }

    public void fit(Dataset train, Dataset val) {

        SGD sgd = new SGD(train);
        Monitor monitor = new Monitor(train, val);

        int T = m_params.T;
        boolean ok = true;
        for (int t = 1; t <= T && ok; t++) {
            sgd.fit();
            ok = monitor.check(t);
            //maxvalue();
        }
        m_W = monitor.optWeights();
        m_b = monitor.optBias();
    }
    

    public int predict(Pattern x) {
        double[] y = f(x.sequence());
        return Array.indexOfMax(y);
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
            double[] z = f(x.sequence());
            loss -= Math.log(z[x.label()]);
        }
        loss /= (double) X.size();
        return loss;
    }

    private double[] f(double[] x) {
        double[] a = new double[m_numLabels];
        for(int j = 0; j < m_numLabels; j++) {
            a[j] = Alignment.sim(x, m_W[j]) + m_b[j];
        }
        return m_loss.apply(a);
    }


    @FunctionalInterface
    interface Update {
        double apply(double err, int j, int r, int s);
    }

    class SGD {

        // data
        double[][] X;       // data set
        int[] y;            // labels

        // hyper-parameters
        int method;         // optimization technique
        int elasticity;     // elasticity
        double eta;         // current learning rate
        double mu;          // momentum
        double lambda;      // regularization

        // sizes
        int inUnits;        // input size
        int outUnits;       // output size
        int numX;           // sample size

        // auxiliary variables
        double[][][] v;     // momentum
        double[] a;         // activation
        Alignment[] A;      // alignment

        // optimization technique
        Update sgd;


        SGD(Dataset data) {

            // set data
            X = data.patterns();
            y = data.labels();

            // set hyper-parameters
            method = m_params.A;
            elasticity = m_params.e;
            eta = m_params.l;
            mu = m_params.m;
            lambda = m_params.r;

            // set sizes
            inUnits = data.maxLength();
            outUnits = data.numLabels();
            numX = data.size();
            m_numLabels = outUnits;

            // set auxiliary variables
            v = new double[outUnits][inUnits+1][elasticity];
            a = new double[outUnits];
            A = new Alignment[outUnits];

            // initialize weights
            double sigma = Math.sqrt(inUnits);
            m_W = m_random.nextArray(outUnits, inUnits , elasticity, sigma);
            m_b = m_random.nextArray(outUnits, sigma);

            // set optimization technique
            setSGD();
        }

        void fit() {


            int[] f = m_random.shuffle(numX);
            for (int i = 0; i < numX; i++) {

                double[] xi = X[f[i]];
                int yi = y[f[i]];

                // compute activation
                for(int j = 0; j < outUnits; j++) {
                    A[j] = new Alignment(xi, m_W[j]);
                    a[j] = A[j].sim() + m_b[j];
                }
                // compute output
                m_loss.apply(a);

                // apply
                double[] delta = m_loss.derivative(yi);
                for (int j = 0; j < outUnits; j++) {
                    double[][] wj = m_W[j];
                    int[][] path = A[j].path();
                    int len = path.length;
                    int r, s;
                    for (int l = 0; l < len; l++) {
                        r = path[l][0]; // index of input x
                        s = path[l][1]; // index of elasticity
                        double err = delta[j] * xi[r] + lambda * wj[r][s];
                        wj[r][s] -= sgd.apply(err, j, r, s);
                    }
                    // apply bias
                    double err = delta[j];
                    m_b[j] -= sgd.apply(err, j, inUnits, 0);
                }
            }
        }

        private void setSGD() {
            if(method == Parameter.A_SGD) {
                sgd = (err, j, r, s) -> eta * err;
            } else if(method == Parameter.A_MOMENTUM) {
                sgd = (err, j, r, s) -> {
                    v[j][r][s] = mu * v[j][r][s] - eta * err;
                    return -v[j][r][s];
                };
            } else if(method == Parameter.A_ADAGRAD) {
                sgd = (err, j, r, s) -> {
                    v[j][r][s] += err * err;
                    return eta * err / (Math.sqrt(v[j][r][s]) + 10E-8);
                };
            } else if(method == Parameter.A_ADADELTA) {
                sgd = (err, j, r, s) -> {
                    v[j][r][s] = mu * v[j][r][s] + (1 - mu) * err * err;
                    return eta * err / (Math.sqrt(v[j][r][s]) + 10E-8);
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
        Softmax optModel;       // optimal model

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
            optModel = new Softmax(m_W, m_b);
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
                    optModel.m_b = Array.cp(m_b);
                }
            } else {
                numStable++;
            }

            // report
            if(loggable) {
                info(t);
            }

            return maxAcc < 1.0;
        }

        double[][][] optWeights() {
            return optModel.m_W;
        }

        double[] optBias() {
            return optModel.m_b;
        }

        private void info(int t) {
            String s = "[CNN] %5d  loss = %7.5f (%7.5f)  test = %1.3f (%1.3f)  train = %1.3f (%1.3f)%n";
            System.out.printf(s, t, curLoss, minLoss, curTestAcc, optTestAcc, curAcc, maxAcc);
        }
    }

}
