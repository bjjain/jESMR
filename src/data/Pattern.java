package data;

import util.Array;
import util.Msg;

import java.util.Arrays;

/**
 * Represents augmented feature sequence. An extra dimension is added to m_x and set to 1.
 *
 * @author jain
 */
public class Pattern {

    int m_label;                // class label
    double[] m_x;               // feature sequence
    private int m_id;           // internal id

    /**
     * Creates a labeled time series pattern. The specified matrix must be rectangular and non-empty.
     *
     * @param x     time series
     * @param label class label
     */
    public Pattern(double[] x, int label) {
        set(x);
        m_label = label;
    }

    private Pattern(double[] x, int label, int id) {
        set(x);
        m_label = label;
        m_id = id;
    }

    private static void check(double[] x) {
        if (x == null || x.length == 0) {
            Msg.error("Error. Feature sequence is empty");
        }
    }

    public void set(double[] x) {
        check(x);
        m_x = x;
    }

    public void set(int id) {
        m_id = id;
    }

    public void augment() {
        m_x = Arrays.copyOf(m_x, m_x.length + 1);
    }

    public void pad(int k0, int k1) {
        int n = m_x.length;
        double[] x = new double[n + k0 + k1];
        System.arraycopy(m_x, 0, x, k0, n);
        m_x = x;
    }

    public void setBias(double b) {
        m_x[m_x.length - 1] = b;
    }

    /**
     * Returns feature sequence.
     *
     * @return time series
     */
    public double[] sequence() {
        return m_x;
    }

    /**
     * Returns class label.
     *
     * @return class label
     */
    public int label() {
        return m_label;
    }

    /**
     * Returns dimension of time series.
     *
     * @return length
     */
    public int dim() {
        return m_x.length;
    }

    /**
     * Returns length of time series.
     *
     * @return length
     */
    public int length() {
        return m_x.length;
    }


    /**
     * Returns deep copy of this pattern.
     *
     * @return copy of this pattern
     */
    public Pattern cp() {
        return new Pattern(Array.cp(m_x), m_label, m_id);
    }

}
