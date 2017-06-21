package classifier;

import core.Options;
import data.Dataset;
import data.Pattern;

/**
 * Created by jain on 19.06.17.
 */
public abstract class Classifier {

    public abstract Options getOptions();

    /**
     * Returns state S of convergence:
     * S < 0 : diverges or oscillates
     * S = 0 : decreases loss
     * S > 0 : has converged
     */
    public abstract int fit(Dataset train);

    public abstract int predict(Pattern x);

    public double score(Dataset X) {
        int acc = 0;
        for (Pattern x : X) {
            if (x.label() == predict(x)) {
                acc++;
            }
        }
        return ((double) acc) / X.size();
    }
}
