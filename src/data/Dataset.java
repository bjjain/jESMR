package data;

import util.Array;
import util.Msg;
import util.Rand;

import java.util.ArrayList;
import java.util.stream.Collectors;

/**
 * @author jain
 */
@SuppressWarnings("serial")
public class Dataset extends ArrayList<Pattern> {

    private static final Rand RND = Rand.getInstance();
    private final int m_numLabels;      // number of labels
    private String m_name;              // name of dataset

    public Dataset(ArrayList<Pattern> patterns, int numLabels) {
        m_numLabels = numLabels;
        m_name = "unknown dataset";
        addAll(patterns);
    }

    public Dataset(String name, int numLabels) {
        m_name = name;
        m_numLabels = numLabels;
    }

    public Dataset(int numLabels) {
        m_name = "unknown dataset";
        m_numLabels = numLabels;
    }

    public Dataset cp() {
        Dataset X = new Dataset(m_name, m_numLabels);
        X.addAll(stream().map(Pattern::cp).collect(Collectors.toList()));
        return X;
    }

    public double[][] patterns() {
        int n = size();
        double[][] x = new double[n][];
        for (int i = 0; i < n; i++) {
            x[i] = get(i).sequence();
        }
        return x;
    }

    public int[] labels() {
        int n = size();
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            y[i] = label(i);
        }
        return y;
    }

    public void augment() {
        for (Pattern x : this) {
            x.augment();
        }
    }

    public void pad(int k0, int k1) {
        for (Pattern x : this) {
            x.pad(k0, k1);
        }
    }

    public void setBias(double b) {
        for (Pattern x : this) {
            x.setBias(b);
        }
    }

    public double[] pattern(int index) {
        return get(index).m_x;
    }

    public int label(int index) {
        return get(index).m_label;
    }

    public int numLabels() {
        return m_numLabels;
    }

    public int maxLength() {
        int maxlen = 0;
        for (Pattern x : this) {
            maxlen = Math.max(maxlen, x.length());
        }
        return maxlen;
    }

    public String name() {
        return m_name;
    }

    public void setName(String name) {
        m_name = name;
    }

    public void normalize() {
        for (Pattern x : this) {
            double[] z = Array.normalize(x.sequence());
            x.set(z);
        }
    }

    /**
     * Returns class-wise partition of specified datasets.
     */
    public Dataset[] splitClasswise() {
        Dataset[] folds = new Dataset[m_numLabels];
        for (int i = 0; i < m_numLabels; i++) {
            folds[i] = new Dataset(m_numLabels);
        }
        for (Pattern x : this) {
            folds[x.label()].add(x);
        }
        return folds;
    }

    public Dataset[] split(double perc) {
        Dataset[] folds = splitClasswise();
        Dataset train = new Dataset(numLabels());
        Dataset test = new Dataset(numLabels());
        for (Dataset fold : folds) {
            int n = fold.size();
            int k = (int) Math.round(perc * n);
            int[] f = RND.shuffle(n);
            for (int i = 0; i < k; i++) {
                test.add(fold.get(f[i]));
            }
            for (int i = k; i < n; i++) {
                train.add(fold.get(f[i]));
            }
        }
        return new Dataset[]{train, test};
    }


    /**
     * Returns partition of specified datasets into the specified number of folds.
     *
     * @param numFolds the number of folds
     * @return class-wise partition of this dataset
     */
    public Dataset[] splitFolds(int numFolds) {

        if (numFolds < 1) {
            Msg.error("Invalid number of folds: %d.", numFolds);
        }
        Dataset[] fold = new Dataset[numFolds];
        for (int i = 0; i < numFolds; i++) {
            fold[i] = new Dataset(m_numLabels);
        }

        if (numFolds == size()) {
            for (int i = 0; i < numFolds; i++) {
                fold[i].add(get(i));
            }
            return fold;
        }
        Dataset[] split = splitClasswise();
        for (int i = 0; i < m_numLabels; i++) {
            int m = split[i].size();
            int[] pi = RND.shuffle(m);
            for (int j = 0; j < m; j++) {
                int k = pi[j] % numFolds;
                fold[k].add(split[i].get(j));
            }

        }
        return fold;
    }
}
