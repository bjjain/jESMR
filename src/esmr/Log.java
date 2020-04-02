package esmr;

import java.util.Arrays;

public class Log {

    private static final int NUM_ERRS = 2;

    int epoch;
    double[] err;
    double[] minErr;
    int maxEpochs;
    int maxStable;
    int[] numStable;
    boolean[] hasImproved;
    int verbosity;

    Log(int maxEpochs, int maxStable, int verbosity) {
        this.maxEpochs = maxEpochs;
        this.maxStable = maxStable;
        this.verbosity = verbosity;
        numStable = new int[NUM_ERRS];
        minErr = new double[NUM_ERRS];
        hasImproved = new boolean[NUM_ERRS];
        Arrays.fill(minErr, Double.POSITIVE_INFINITY);
    }

    void log(double[] error, int t) {
        epoch = t;
        err = error;
        for (int i = 0; i < NUM_ERRS; i++) {
            if (err[i] < minErr[i]) {
                minErr[i] = err[i];
                numStable[i] = 0;
                hasImproved[i] = true;
            } else {
                numStable[i]++;
                hasImproved[i] = false;
            }
        }
        info();
    }

    boolean proceed() {
        boolean proc = 0 < minErr[0];
        proc &= 0 < minErr[1];
        proc &= numStable[0] < maxStable;
        proc &= numStable[1] < 0.1 * maxEpochs;
        return proc;
    }

    void info() {
        if (verbosity == 0) {
            return;
        }
        if (verbosity == 1) {
            System.out.printf("\rTraining: Epoch %d", epoch);
            return;
        }
        String sErr = toString(err, "%7.4f ");
        String sMinErr = toString(minErr, "%7.4f ");
        System.out.printf("%5d  %s  %s  %n", epoch, sErr, sMinErr);
    }

    boolean decreaseLearningRate(int t) {
        boolean decrease = t < 20 && 0 < numStable[0];
        if (decrease) {
            Arrays.fill(minErr, Double.POSITIVE_INFINITY);
            Arrays.fill(numStable, 0);
        }
        return decrease;
    }

    String toString(double[] x, String format) {
        if (x == null) {
            return "<null>";
        }
        StringBuilder s = new StringBuilder();
        for (double val : x) {
            s.append(String.format(format, val));
        }
        return s.toString();
    }
}
