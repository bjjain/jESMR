package util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Reader {

    /**
     * Reads delimiter separated values of the specified file. The regular expression regexp specifies the delimiter.
     * The method returns a double[][] array, where each line of the file corresponds to a row of the matrix.
     * <p>
     * Common examples of delimiters:
     * <p>
     * "\\s+"  spaces
     * ","     comma
     * ";"     semicolon
     *
     * @param file   name of file
     * @param regexp delimiter
     * @return double[][] array of values
     */
    public static double[][] load(String file, String regexp) {
        BufferedReader br = null;
        String line;

        ArrayList<double[]> X = new ArrayList<>();
        try {
            br = new BufferedReader(new FileReader(file));
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.equals("")) {
                    continue;
                }
                String[] row = line.split(regexp);
                double[] x = toDouble(row);
                X.add(x);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            finalize(br);
        }
        return transform(X);
    }

    public static double[][] loadSSV(String filename) {
        return load(filename, "\\s+");
    }

    public static double[][] loadCSV(String filename) {
        return load(filename, ",");
    }

    private static double[][] transform(ArrayList<double[]> X) {
        int n = X.size();
        double[][] data = new double[n][];
        HashSet<Integer> labels = new HashSet<>();
        for (int i = 0; i < n; i++) {
            data[i] = X.get(i);
            labels.add((int) data[i][0]);
        }
        // relabel
        List<Integer> sortedLabels = new ArrayList<>(labels);
        Collections.sort(sortedLabels);
        int numLabels = sortedLabels.size();
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < numLabels; i++) {
            map.put(sortedLabels.get(i), i);
        }
        for (int i = 0; i < n; i++) {
            data[i][0] = map.get((int) data[i][0]);
        }
        return data;
    }

    private static double[] toDouble(String[] vals) {
        int n = vals.length;
        double[] x = new double[n];
        int countDoubles = 0;
        for (int i = 0; i < n; i++) {
            x[i] = Double.valueOf(vals[i]);
            if (!Double.isNaN(x[i])) {
                countDoubles++;
            }
        }
        if (countDoubles < n) {
            x = Arrays.copyOfRange(x, 0, countDoubles);
        }
        return x;
    }

    private static void finalize(BufferedReader br) {
        if (br != null) {
            try {
                br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}