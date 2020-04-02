package esmr.regularizer;

/**
 * L1 Regularization
 */
public class L1 implements Regularizer {

    double[][] m_w;

    public L1(double[][] w) {
        m_w = w;
    }

    @Override
    public double derivative(int r, int s) {
        return Math.signum(m_w[r][s]);
    }

    @Override
    public double loss(int r, int s) {
        return Math.abs(m_w[r][s]);
    }
}
