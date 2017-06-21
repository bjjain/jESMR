package mean;

import data.Dataset;
import data.Pattern;

public class NearestNeighbor {

    Dataset m_X;
    DTW m_dtw;

    public NearestNeighbor() {
        m_dtw = new DTW();
    }


    public void fit(Dataset X) {
        m_X = X;
    }

    public int predict(Pattern z) {
        double minDist = Double.MAX_VALUE;
        int predictedLabel = -1;
        for (Pattern x : m_X) {
            double d = m_dtw.d(x.sequence(), z.sequence());
            if (d < minDist) {
                minDist = d;
                predictedLabel = x.label();
            }
        }
        return predictedLabel;
    }


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
