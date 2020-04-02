package esmr.regularizer;


public class L2 implements Regularizer {

    double[][] m_w;

    public L2(double[][] w) {
        m_w = w;
    }

    @Override
    public double derivative(int r, int s) {
        return m_w[r][s];
    }

    @Override
    public double loss(int r, int s) {
        return m_w[r][s] * m_w[r][s];
    }
}
