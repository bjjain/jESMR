package esmr;

import esmr.regularizer.L1;
import esmr.regularizer.L2;
import esmr.regularizer.Regularizer;
import esmr.regularizer.Zero;
import util.Msg;
import util.Options;

class Parameter {

    //*** ADAM  ********************************************************************************************************
    double lr = 0.001;                  // initial learning rate
    double b1 = 0.9;                    // first momentum
    double b2 = 0.99;                   // second momentum

    //*** regularization ***********************************************************************************************
    int e = 3;                          // inner elasticity (max-lin)
    int R = 0;                          // type of regularization
    double r = 0.0;                     // regularization parameter

    //*** termination **************************************************************************************************
    int T = 1000;                       // maximum number of epochs
    int S = 1000;                       // maximum number of epochs without improvement

    //** logging *******************************************************************************************************
    int o = 2;                          // output mode

    Options opts;                       // options

    private Parameter() {
        setOptions();
    }

    Parameter(String opts) {
        this();
        setOptions(opts);
    }

    public Options setOptions(String options) {
        opts.add(options);
        String flag = "-l";
        if (opts.containsKey(flag)) {
            lr = opts.getDouble(flag);
            if (lr <= 0) {
                error(flag, lr);
            }
        }
        flag = "-b1";
        if (opts.containsKey(flag)) {
            b1 = opts.getDouble(flag);
            if (b1 < 0) {
                error(flag, b1);
            }
        }
        flag = "-b2";
        if (opts.containsKey(flag)) {
            b2 = opts.getDouble(flag);
            if (b2 < 0) {
                error(flag, b2);
            }
        }
        flag = "-e";
        if (opts.containsKey(flag)) {
            e = opts.getInt(flag);
            if (e <= 0) {
                error(flag, e);
            }
        }
        flag = "-R";
        if (opts.containsKey(flag)) {
            R = opts.getInt(flag);
        }
        flag = "-r";
        if (opts.containsKey(flag)) {
            r = opts.getDouble(flag);
            if (r < 0) {
                error(flag, r);
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

    Options getOptions() {
        setOptions();
        return opts;
    }

    Regularizer getRegularizer(double[][] w) {
        switch (R) {
            case 0:
                return new Zero();
            case 1:
                return new L1(w);
            case 2:
                return new L2(w);
            default:
                Msg.error("Error! Unknown type of regularizer: %d.", R);
        }
        return null;
    }

    private void setOptions() {
        opts = new Options();
        opts.put("-l", Double.toString(lr));
        opts.put("-b1", Double.toString(b1));
        opts.put("-b2", Double.toString(b2));
        opts.put("-e", Integer.toString(e));
        opts.put("-R", Integer.toString(R));
        opts.put("-r", Double.toString(r));
        opts.put("-T", Integer.toString(T));
        opts.put("-S", Integer.toString(S));
        opts.put("-o", Integer.toString(o));
    }

    private void error(String flag, Object value) {
        String OPTS = "ERROR: Invalid value for parameter %score: %score%n"
                + "OPTIONS:%n"
                + "-l    <double> : initial learning rate (default " + lr + ")%n"
                + "-b1   <double> : decay rate >= 0 (default " + b1 + ")%n"
                + "-b2   <double> : decay rate >= 0 (default " + b2 + ")%n"
                + "-e    <int>    : elasticity > 0 (default " + e + ")%n"
                + "-R    <int>    : type of regularization (default " + R + ")%n"
                + "         0 -- void %n"
                + "         1 -- L1 %n"
                + "         2 -- L2 %n"
                + "-r    <double> : weight decay >= 0 (default " + r + ")%n"
                + "-T    <int>    : max number of epochs (default " + T + ")%n"
                + "-S    <int>    : max number of stable epochs (default " + S + ")%n"
                + "-o    <int>    : output mode (default " + o + ")%n"
                + "         0 -- quiet mode %n"
                + "         1 -- dots %n"
                + "         2 -- progress info after each epoch %n"
                + "         3 -- writes progress to file %n";
        System.out.flush();
        Msg.error(String.format(OPTS, flag, value));
        System.err.print(String.format(OPTS, flag, value));
    }
}
