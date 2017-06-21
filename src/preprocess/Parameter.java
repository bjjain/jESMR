package preprocess;

import core.Options;

class Parameter {

    int zc = 0;                 // z-transformation column-wise (features)
    int zr = 0;                 // z-transformation row-wise    (patterns)
    int p0 = 0;                 // number of zero paddings at beginning
    int p1 = 0;                 // number of zero paddings at end
    int a = 0;                  // augment patterns for bias
    double b = 0;               // scales bias

    Options opts;               // options

    private Parameter() {
        setOptions();
    }

    Parameter(String opts) {
        this();
        setOptions(opts);
    }

    Parameter(Parameter param) {
        zc = param.zc;
        zr = param.zr;
        p0 = param.p0;
        p1 = param.p1;
        a = param.a;
        b = param.b;
        opts = new Options();
        opts.putAll(param.opts);
    }

    private void setOptions() {
        opts = new Options();
        opts.put("-zc", Double.toString(zc));
        opts.put("-zr", Double.toString(zr));
        opts.put("-p0", Double.toString(p0));
        opts.put("-p1", Double.toString(p1));
        opts.put("-a", Double.toString(a));
        opts.put("-b", Double.toString(b));
    }

    private Options setOptions(String args) {
        opts.add(args);

        String flag = "-zc";
        if (opts.containsKey(flag)) {
            zc = opts.getInt(flag);
        }
        flag = "-zr";
        if (opts.containsKey(flag)) {
            zr = opts.getInt(flag);
        }
        flag = "-p0";
        if (opts.containsKey(flag)) {
            p0 = opts.getInt(flag);
        }
        flag = "-p1";
        if (opts.containsKey(flag)) {
            p1 = opts.getInt(flag);
        }
        flag = "-a";
        if (opts.containsKey(flag)) {
            a = opts.getInt(flag);
        }
        flag = "-b";
        if (opts.containsKey(flag)) {
            b = opts.getDouble(flag);
        }
        return opts;
    }

    String getOptions() {
        return " -zc " + zc + " -zr " + zr + " -p0 " + p0 + " -p1 " + p1 + " -a " + a + " -b " + b;
    }

    private void error(String flag, Object value) {
        String OPTS = "ERROR: Invalid value for parameter %sim: %sim%n"
                + "OPTIONS:%n"
                + "-zc <int>    : z-transformation column/feature-wise (default " + zc + ")%n"
                + "-zr <int>    : z-transformation row/pattern-wise (default " + zr + ")%n"
                + "-p0 <int>    : number of zero paddings at start (default " + p0 + ")%n"
                + "-p1 <int>    : number of zero paddings at end (default " + p1 + ")%n"
                + "-a  <int>    : augment patterns (default " + a + ")%n"
                + "-b  <double> : scales bias (default " + b + ")%n";
        System.out.flush();
        System.err.printf(String.format(OPTS, flag, value));
        System.exit(1);
    }
}
