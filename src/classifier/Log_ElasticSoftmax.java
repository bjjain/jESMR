package classifier;

import core.MaxAlignment;
import data.Dataset;
import functions.Func;
import util.Array;
import util.Writer;

import java.io.File;
import java.util.Arrays;
import java.util.HashSet;

/**
 * Created by jain on 04.05.17.
 */
public class Log_ElasticSoftmax {

    ElasticSoftmax m_model;
    String m_path;
    String m_data;
    String m_type;       // "0" = training set; "1" = test set

    public Log_ElasticSoftmax(ElasticSoftmax model, String path, String data, int elasticity) {
        m_model = model;
        m_path = path;
        m_data = data + "_E" + elasticity;
    }

    public void log(Dataset X, int type) {

        m_type = type <= 0 ? "0" : "1";
        Func F = m_model.m_F;
        double[][][] W = m_model.m_W;
        int outUnits = m_model.m_numLabels;
        int inUnits = W[0].length;
        int elasticity = W[0][0].length;
        int numLabels = m_model.m_numLabels;

        int numX = X.size();
        int dim = X.maxLength() + 2;

        int[][][] L = new int[outUnits][inUnits][elasticity];       // count point visits in lattice
        double[][] Z = new double[numX][dim];                       // data matrix with labels and prediction
        PathSet[] paths = new PathSet[outUnits];                    // used paths
        int[][] C = new int[numLabels][numLabels];                  // confusion matrix (true vs predicted)

        for (int j = 0; j < outUnits; j++) {
            paths[j] = new PathSet();
        }

        for (int i = 0; i < numX; i++) {
            double[] x = X.pattern(i);
            MaxAlignment[] A = new MaxAlignment[outUnits];
            double[] a = new double[outUnits];
            for (int j = 0; j < outUnits; j++) {
                A[j] = new MaxAlignment(x, W[j]);
                a[j] = A[j].sim();

                int[][] P = A[j].path();
                paths[j].add(new Path(P, W[j]));
                int len = P.length;
                int r, s;
                for (int l = 0; l < len; l++) {
                    r = P[l][0];
                    s = P[l][1];
                    L[j][r][s]++;
                }
            }
            System.arraycopy(x, 0, Z[i], 0, x.length);
            int yTrue = X.label(i);
            int yPred = F.predict(a);
            Z[i][dim - 2] = yTrue;
            Z[i][dim - 1] = yPred;
            C[yTrue][yPred]++;
        }
        write(L);
        write(Z);
        write(C);
        write(paths);
    }


    private void write(int[][] C) {
        String filename = m_path + m_data + "_C" + m_type + ".txt";
        File file = new File(filename);
        if (file.exists()) {
            file.delete();
        }
        String text = Array.toString(C) + "\n";
        Writer.append(text, filename);
    }


    private void write(int[][][] L) {
        String filename = m_path + m_data + "_L" + m_type + ".txt";
        File file = new File(filename);
        if (file.exists()) {
            file.delete();
        }
        int outUnits = L.length;
        int elasticity = L[0][0].length;
        int[] header = new int[elasticity];
        header[0] = L[0].length;
        String text = Array.toString(header) + "\n";
        Writer.append(text, filename);
        for (int i = 0; i < outUnits; i++) {
            Writer.append(Array.toString(L[i]), filename);
        }
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

    private void write(PathSet[] paths) {
        String filename = m_path + m_data + "_P" + m_type + ".txt";
        File file = new File(filename);
        if (file.exists()) {
            file.delete();
        }
        int outUnits = paths.length;
        for (int i = 0; i < outUnits; i++) {
            for (Path p : paths[i]) {
                String str = i + " " + p.toString() + "\n";
                Writer.append(str, filename);
            }
        }
    }


    class Path {
        int[][] p;
        double[] w;

        Path(int[][] P, double[][] W) {
            p = Array.cp(P);
            setweights(W);
        }

        private void setweights(double[][] W) {
            int d = W.length;
            int len = p.length;
            int r, s;
            w = new double[d];
            for (int l = 0; l < len; l++) {
                r = p[l][0];
                s = p[l][1];
                w[r] += W[r][s];
            }
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Path path = (Path) o;
            return Arrays.deepEquals(p, path.p);
        }

        @Override
        public int hashCode() {
            return Arrays.deepHashCode(p);
        }

        @Override
        public String toString() {
            return Array.toString(w, "%9.6f ");
        }
    }

    class PathSet extends HashSet<Path> {
    }
}
