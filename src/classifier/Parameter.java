package classifier;

import core.Options;

class Parameter {

    //--- optimization techniques
    static final int A_SGD      = 0;    // ssg
    static final int A_MOMENTUM = 1;    // ssg + momentum
    static final int A_ADAGRAD  = 2;    // ssg + adagrad
    static final int A_ADADELTA = 3;    // ssg + adadelta
    static final int A_ADAM     = 4;    // ssg + adam

    int A = 0;                          // optimization technique
    int p = 1;                          // partitions
    int e = 3;                          // elasticity
    double l = 0.001;                   // learning rate
    double r = 0.0;                     // regularization: weight decay
    double m = 0.5;                     // momentum
    double r1 = 0.9;                    // decay rate (for adadelta, adam)
    double r2 = 0.95;                   // decay rate (for adam)
    int T = 1000;                       // maximum number of epochs
    int S = 1000;                       // maximum number of epochs without improvement
    int o = 3;                          // output mode

    Options opts;                       // options

    private Parameter() {
        setOptions();
    }

    Parameter(String opts) {
        this();
        setOptions(opts);
    }

    Parameter(Parameter param) {
        A = param.A;
        p = param.p;
        e = param.e;
        l = param.l;
        r = param.r;
        m = param.m;
        r1 = param.r1;
        r2 = param.r1;
        T = param.T;
        S = param.S;
        o = param.o;
        opts = new Options();
        opts.putAll(param.opts);
    }

    private void setOptions() {
        opts = new Options();
        opts.put("-A", Double.toString(A));
        opts.put("-p", Double.toString(p));
        opts.put("-e", Double.toString(e));
        opts.put("-l", Double.toString(l));
        opts.put("-r", Double.toString(r));
        opts.put("-m", Double.toString(m));
        opts.put("-r1", Double.toString(r1));
        opts.put("-r2", Double.toString(r2));
        opts.put("-T", Integer.toString(T));
        opts.put("-S", Integer.toString(S));
        opts.put("-o", Integer.toString(o));
    }

    private Options setOptions(String args) {
        opts.add(args);

        String flag = "-A";
        if (opts.containsKey(flag)) {
            A = opts.getInt(flag);
            if (A < 0) {
                error(flag, A);
            }
        }
        flag = "-p";
        if (opts.containsKey(flag)) {
            p = opts.getInt(flag);
            if (p <= 0) {
                error(flag, p);
            }
        }
        flag = "-e";
        if (opts.containsKey(flag)) {
            e = opts.getInt(flag);
            if (e <= 0) {
                error(flag, e);
            }
        }
        flag = "-l";
        if (opts.containsKey(flag)) {
            l = opts.getDouble(flag);
            if (l <= 0) {
                error(flag, l);
            }
        }
        flag = "-r";
        if (opts.containsKey(flag)) {
            r = opts.getDouble(flag);
            if (r < 0) {
                error(flag, r);
            }
        }
        flag = "-m";
        if (opts.containsKey(flag)) {
            m = opts.getDouble(flag);
            if (m < 0) {
                error(flag, m);
            }
        }
        flag = "-r1";
        if (opts.containsKey(flag)) {
            r1 = opts.getDouble(flag);
            if (r1 < 0) {
                error(flag, r1);
            }
        }
        flag = "-r2";
        if (opts.containsKey(flag)) {
            r2 = opts.getDouble(flag);
            if (r2 < 0) {
                error(flag, r2);
            }
        }
        flag = "-T";
        if (opts.containsKey(flag)) {
            T = opts.getInt(flag);
            if (T < 1) {
                error(flag, T);
            }
        }
        flag = "-S";
        if (opts.containsKey(flag)) {
            S = opts.getInt(flag);
            if (S < 0) {
                error(flag, S);
            }
        }
        flag = "-o";
        if (opts.containsKey(flag)) {
            o = opts.getInt(flag);
        }

        return opts;
    }

    String getParams() {
        return " -A " + A + " -p " + p + " -e " + e +
                " -l " + l + " -r " + r + " -m " + m + " -r1 " + r1 +
                " -r2 " + r2 + " -T " + T + " -S " + S + " -o " + o + " ";
    }

    Options getOptions() {
        return opts;
    }

    private void error(String flag, Object value) {
        String OPTS = "ERROR: Invalid value for parameter %sim: %sim%n"
                + "OPTIONS:%n"
                + "-A  <int>    : optimization technique (default " + A + ")%n"
                + "     0 -- SSG %n"
                + "     1 -- SSG + momentum %n"
                + "     2 -- SSG + adagrad %n"
                + "     3 -- SSG + adadelta %n"
                + "     4 -- SSG + adam %n"
                + "-p  <int>    : partitions > 0 (default " + p + ")%n"
                + "-e  <int>    : elasticity > 0 (default " + e + ")%n"
                + "-l  <double> : learning rate > 0 (default " + l + ")%n"
                + "-r  <double> : weight decay >= 0 (default " + r + ")%n"
                + "-m  <double> : momentum >= 0.5 (default + " + m + ")%n"
                + "-r1 <double> : decay rate >= 0 (default " + r1 + ")%n"
                + "-r2 <double> : decay rate >= 0 (default " + r2 + ")%n"
                + "-T  <int>    : max number of epochs (default " + T + ")%n"
                + "-S  <int>    : max number of stable epochs (default " + S + ")%n"
                + "-o  <int>    : output mode (default " + o + ")%n"
                + "    0 -- quiet mode %n"
                + "    1 -- cross-validation mode %n"
                + "    2 -- progress info after each epoch %n";
        System.out.flush();
        System.err.printf(String.format(OPTS, flag, value));
        System.exit(1);
    }
}
