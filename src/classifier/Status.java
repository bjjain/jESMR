package classifier;

public class Status {

    public static final int LOSS_NAN = -2;   // loss is not a number
    public static final int LOSS_OSC = -1;   // loss oscillates
    public static final int LOSS_DECR = 0;   // loss decreases
    public static final int LOSS_CONV = 1;   // loss has converged
    // tracking period
    static final int trPeriod = 30;
    // state
    public int state;
    // best results
    public double maxAcc;
    public double minLoss;
    // tracking data
    public double[] trAcc;
    public double[] trLoss;
    boolean updateModel;
    // convergence
    int numStable;
    int maxStable;
    // pointer
    private int trNext;

    Status(int maxStable) {
        this.maxStable = maxStable;
        this.minLoss = Double.POSITIVE_INFINITY;
        trAcc = new double[trPeriod];
        trLoss = new double[trPeriod];
    }

    int set(double acc, double loss, int t) {
        updateModel = false;
        trAcc[trNext] = acc;
        trLoss[trNext] = loss;
        trNext = (trNext + 1) % trPeriod;
        if (!Double.isFinite(loss)) {
            state = LOSS_NAN;
            return state;
        }
        if (maxAcc <= acc) {
            maxAcc = acc;
        }
        if (loss < minLoss) {
            minLoss = loss;
            updateModel = true;
            numStable = 0;

        } else {
            numStable++;
        }

        // check convergence
        double ratio = numStable / ((double) t);
        if (20 <= t && t <= 100 && 0.2 < ratio) {
            state = LOSS_OSC;
        } else {
            state = maxAcc < 1.0 && numStable < maxStable ? LOSS_DECR : LOSS_CONV;
        }
        return state;
    }

    boolean decrLearningRate() {
        return state < 0;
    }

    void info(int t) {
        int i = trNext == 0 ? trPeriod - 1 : trNext - 1;
        String s = "[ESMR] %5d  loss = %7.5f (%7.5f)  train = %1.3f (%1.3f)%n";
        System.out.printf(s, t, trLoss[i], minLoss, trAcc[i], maxAcc);

    }

}
