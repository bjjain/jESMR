package esmr;

import data.Dataset;
import data.Pattern;
import util.Msg;
import util.Options;

public abstract class Classifier {

    public static final double EPS = 10E-15;

    public abstract Options getOptions();

    public abstract void fit(Dataset X);

    public abstract int predict(Pattern p);

    public abstract String getName();

    public double eval(Dataset X) {
        if (X == null || X.size() == 0) {
            return 0;
        }
        int err = 0;
        for (Pattern p : X) {
            if (p.y != predict(p)) {
                err++;
            }
        }
        return ((double) err) / X.size();
    }

    protected final void check(Dataset X) {
        if (X == null || X.size() == 0) {
            Msg.error("Error! Empty dataset!");
        }
    }
}
