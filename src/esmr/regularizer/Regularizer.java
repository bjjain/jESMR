package esmr.regularizer;


public interface Regularizer {

    double derivative(int r, int s);

    double loss(int r, int s);
}
