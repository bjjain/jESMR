package esmr.regularizer;

public class Zero implements Regularizer {

    @Override
    public double derivative(int r, int s) {
        return 0;
    }

    @Override
    public double loss(int r, int s) {
        return 0;
    }
}
