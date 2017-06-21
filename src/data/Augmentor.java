package data;

import util.Array;
import util.Rand;

import java.util.Arrays;

/**
 * Created by jain on 06.04.17.
 */
public class Augmentor {

    private static final Rand RND = Rand.getInstance();

    Dataset m_X;
    Dataset m_Y;

    public Augmentor(Dataset X) {
        m_X = X;
    }

    public Dataset augmentedData() {
        return m_Y;
    }

    public Dataset augment(int k) {
        if (k <= 0) {
            return m_X;
        }
        m_Y = new Dataset(m_X.numLabels());
        Dataset Y = new Dataset(m_X.numLabels());
        for (Pattern x : m_X) {
            Y.add(x);
            for (int i = 0; i < k; i++) {
                Pattern y = x.cp();
                double[] yx = y.sequence();
                yx = noise(yx);
                if (RND.nextDouble() < 0.5) {
                    yx = crop(yx, 0.2 * RND.nextDouble());
                }
                if (RND.nextDouble() < 0.5) {
                    yx = swap(yx, 0.2 * RND.nextDouble());
                }
                y.set(yx);
                y.set(Array.normalize(yx));
                Y.add(y);
                m_Y.add(y);
            }
        }
        return Y;
    }


    double[] noise(double[] x) {
        int n = x.length;
        for (int i = 0; i < n; i++) {
            x[i] = (1.0 + 0.1 * RND.nextGaussian()) * x[i];
        }
        return x;
    }

    double[] crop(double[] x, double p) {
        int n = x.length;

        int amount = (int) p * n;
        if (amount == 0) {
            return x;
        }
        int maxLength = Math.max(1, RND.nextInt(amount));
        int numCrops = Math.max(1, amount / maxLength);

        int step = n / numCrops;
        for (int k = 0; k < numCrops; k++) {
            int len = RND.nextInt(maxLength);
            int k0 = k * step;
            int k1 = Math.min(k0 + len, n);
            for (int i = k0; i < k1; i++) {
                x[i] = 0.1 * RND.nextGaussian();
                if (n <= k0 + len) {
                    break;
                }
            }
        }
        return x;
    }

    double[] swap(double[] x, double p) {
        int n = x.length;
        int maxLength = (int) p * n;
        if (maxLength < 2) {
            return x;
        }
        int len = Math.max(2, RND.nextInt(maxLength));
        int r0 = RND.nextInt((n - len) / 2);
        int r1 = r0 + len;
        int gap = RND.nextInt(n - r1 - 2 * len);
        int s0 = r1 + gap;
        int s1 = s0 + len;
        double[] z0 = Arrays.copyOfRange(x, r0, r1);
        double[] z1 = Arrays.copyOfRange(x, s0, s1);
        System.arraycopy(z0, 0, x, s0, s1 - s0);
        System.arraycopy(z1, 0, x, r0, r1 - r0);

        return x;
    }
}
