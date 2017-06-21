package mean;

import core.Options;
import util.Rand;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by jain on 11/07/16.
 */
public class Mean {

    // name of mean algorithms
    private static final String[] T_ALG = {"DBA", "SSG"};
    private final Rand rnd;
    // distance function
    private final DTW dtw = new DTW();

    // length of mean
    private int length;

    // type of mean algorithm
    private int t_alg = 1;

    // initial learning rate
    private double eta0 = 0.1;

    // final learning rate (SSG)
    private double eta1 = 0.001;

    // termination: maximum number of epochs
    private int T = 50;

    // termination: maximum number of epochs without improvement
    private int t = 5;

    // output mode
    private int o = 2;

    public Mean(String options) {
        length = -1;
        rnd = Rand.getInstance();
        setOptions(options);
    }

    public double[] compute(double[][] x) {
        return compute(null, x);
    }

    public double[] compute(double[] m, double[][] x) {

        int N = x.length;
        if (m == null) {
            double[] z = x[rnd.nextInt(N)];
            m = warp(z);
        }
        if (t_alg == 0) {
            return runMMM(m, x);
        } else if (t_alg == 1) {
            return runSSG(m, x);
        } else {
            System.out.flush();
            System.err.println("Error. Unknown type of mean algorithm.");
            System.exit(0);
        }
        return null;
    }

    private double[] warp(double[] x) {
        int n = x.length;
        if (length <= 0 || length == n) {
            return Arrays.copyOf(x, n);
        }
        if (length < n) {
            return downsample(x);
        }
        if (n < length) {
            return upsample(x);
        }
        return null;
    }

    private double[] downsample(double[] x) {
        int n = x.length;
        int[] f = rnd.choose(length, n);
        double[] z = new double[length];
        for (int i = 0; i < length; i++) {
            z[i] = x[f[i]];
        }
        return z;
    }

    private double[] upsample(double[] x) {
        int n = x.length;
        List<Double> list = new LinkedList<Double>();
        for (double val : x) {
            list.add(val);
        }
        while (n < length) {
            int k = rnd.nextInt(n);
            double val = list.get(k);
            if (k < n - 1) {
                val += list.get(k + 1);
                list.add(k, val / 2.0);
            } else {
                list.add(k, val);
            }
            n++;
        }
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            z[i] = list.get(i);
        }
        return z;
    }

    private double[] runMMM(double[] mean, double[][] x) {

        // sample size
        int N = x.length;
        // dimension
        int n = mean.length;
        // previous V
        double fprev = -1;
        // next V
        double fnext = fprev;
        // warping path
        int[][] w;
        // valence matrix
        int[] V = new int[n];

        // auxiliary variables
        double[] z = new double[n];
        double diff = 1.0;
        double eps = 1E-4;
        for (int epoch = 0; epoch < T && eps < diff; epoch++) {
            Arrays.fill(z, 0);
            Arrays.fill(V, 0);
            for (int i = 0; i < N; i++) {
                w = dtw.align(mean, x[i]);
                int l = w.length;
                for (int j = 0; j < l; j++) {
                    V[w[j][0]]++;
                    z[w[j][0]] += x[i][w[j][1]];
                }
            }
            for (int i = 0; i < n; i++) {
                mean[i] = z[i] / V[i];
            }
            fprev = fnext;
            fnext = f(mean, x);
            diff = Math.abs(fprev - fnext) / Math.max(fnext, eps);

            // output
            if (o != 0) {
                info(epoch, fprev, fnext);
            }
        }
        if (o != 0) {
            System.out.println();
        }
        return mean;
    }


    private double[] runSSG(double[] mean, double[][] x) {

        // sample size
        int N = x.length;
        // dimension
        int n = mean.length;
        // best solution
        double[] m0 = Arrays.copyOf(mean, n);
        // best V
        double f0 = Double.POSITIVE_INFINITY;
        // current V
        double ft = 0;
        // number of epochs without improvements
        int numStable = 0;
        // learning rate
        double eta = eta0;
        // decay of learning rate
        double delta = (eta0 - eta1) / N;

        // auxiliary variables
        int[][] p;
        double[] z = new double[n];

        for (int epoch = 0; epoch < T && numStable <= t; epoch++) {
            int[] pi = rnd.shuffle(N);
            for (int i = 0; i < N; i++) {
                int i0 = pi[i];
                p = dtw.align(mean, x[i0]);

                int l = p.length;
                Arrays.fill(z, 0);
                for (int j = 0; j < l; j++) {
                    z[p[j][0]] += mean[p[j][0]] - x[i0][p[j][1]];
                }
                for (int j = 0; j < n; j++) {
                    mean[j] -= eta * z[j];
                }
                if (epoch == 0) {
                    eta -= delta;
                }
            }
            ft = f(mean, x);

            // apply stability
            if (ft < f0) {
                f0 = ft;
                m0 = Arrays.copyOf(mean, n);
                numStable = 0;
            } else {
                numStable++;
            }

            // output
            if (o != 0) {
                info(epoch, ft, f0);
            }
        }
        if (o != 0) {
            System.out.println();
        }
        mean = m0;
        return mean;
    }

    private double f(double[] z, double[][] x) {
        int n = x.length;
        double val = 0;
        for (int i = 0; i < n; i++) {
            val += dtw.d(z, x[i]);
        }
        return val / n;
    }

    private void info(int epoch, double ft, double f0) {
        if (epoch == 0 && o == 2) {
            System.out.println("Progress info: [epoch, current variation, best variation]");
        }
        if (o == 1) {
            System.out.print(".");
            if ((epoch + 1) % 50 == 0) {
                System.out.println();
            }
            return;
        } else if (o == 2) {
            System.out.format("%d  %7.4f  %7.4f%n", epoch, ft, f0);
        }
    }

    private void setOptions(String options) {
        Options opts = new Options(options);
        String flag = "-A";
        if (opts.containsKey(flag)) {
            String strType = opts.getString(flag);
            int i = 0;
            for (; i < T_ALG.length; i++) {
                if (T_ALG[i].equals(strType)) {
                    t_alg = i;
                    break;
                }
            }
            if (i == T_ALG.length) {
                error(flag, strType);
            }
        }
        flag = "-L";
        if (opts.containsKey(flag)) {
            length = opts.getInt(flag);
        }
        flag = "-l0";
        if (opts.containsKey(flag)) {
            eta0 = opts.getDouble(flag);
            if (eta0 <= 0) {
                error(flag, eta0);
            }
        }
        flag = "-l1";
        if (opts.containsKey(flag)) {
            eta1 = opts.getDouble(flag);
            if (eta1 <= 0) {
                error(flag, eta1);
            }
        }
        flag = "-t";
        if (opts.containsKey(flag)) {
            t = opts.getInt(flag);
            if (t < 1) {
                error(flag, t);
            }
        }
        flag = "-T";
        if (opts.containsKey(flag)) {
            T = opts.getInt(flag);
            if (T < 1) {
                error(flag, T);
            }
        }
        flag = "-o";
        if (opts.containsKey(flag)) {
            o = opts.getInt(flag);
            if (o < 0 || o > 2) {
                error(flag, o);
            }
        }
    }

    private void error(String flag, Object value) {
        String OPTS = "ERROR: Invalid value for parameter %s: %s%n"
                + "OPTIONS:%n"
                + "-A <int> : t_alg of solver (default " + t_alg + ")%n"
                + "    0 -- DBA %n"
                + "    1 -- SSG %n"
                + "-L <int> : length of mean > 0 (default " + length + ")%n"
                + "-l0 <double> : initial learning rate > 0 (default " + eta0 + ")%n"
                + "-l1 <double> : final learning rate > 0 (default " + eta1 + ")%n"
                + "-T <int> : max number of epochs > 0 (default " + T + ")%n"
                + "-t <int> : max number of stable epochs > 0 (default " + t + ")%n"
                + "-o <int> : output mode (default " + o + ")%n"
                + "    0 -- quiet mode %n"
                + "    1 -- dot after each epoch %n"
                + "    2 -- report progress info after each epoch %n";
        System.out.flush();
        System.err.printf(String.format(OPTS, flag, value));
        System.exit(1);
    }

    public String toString() {
        String str = String.format("-A %s", T_ALG[t_alg]);
        str += " -T " + T + " -t " + t;
        if (t_alg == 1) {
            str += " -l0 " + eta0 + " " + " -l1 " + eta1;
        }
        str += " -o " + o;
        return str;
    }

}
