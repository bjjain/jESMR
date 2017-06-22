package app.utils;

import classifier.Classifier;
import data.Dataset;
import data.Pattern;
import util.Array;
import util.Writer;

import java.io.File;

/**
 * Created by jain on 04.05.17.
 */
public class LogResults {

    Classifier m_model;
    String m_path;
    String m_data;
    String m_type;       // "0" = training set; "1" = test set

    public LogResults(Classifier model, String path, String data, int numPartitions, int elasticity) {
        m_model = model;
        m_path = path;
        m_data = data + "_P" + numPartitions + "_E" + elasticity;
    }

    public void log(Dataset X, int type) {

        m_type = type <= 0 ? "0" : "1";
        int numX = X.size();
        int dim = X.maxLength() + 2;

        // data matrix with labels and prediction
        double[][] D = new double[numX][dim];
        for (int i = 0; i < numX; i++) {
            Pattern x = X.get(i);
            int yTrue = X.label(i);
            int yPred = m_model.predict(x);

            System.arraycopy(x.sequence(), 0, D[i], 0, x.length());
            D[i][dim - 2] = yTrue;
            D[i][dim - 1] = yPred;
        }
        write(D);
    }

    private void write(double[][] Z) {
        String filename = m_path + m_data + "_D" + m_type + ".txt";
        File file = new File(filename);
        if (file.exists()) {
            file.delete();
        }
        for (double[] z : Z) {
            String str = Array.toString(z, "%7.4f ");
            Writer.append(str + "\n", filename);
        }
    }
}
