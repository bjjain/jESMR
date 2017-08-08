package app.experiments;

import classifier.Status;
import util.Array;

public class Log {

    int m_num;
    int m_next;
    Status[] m_status;
    double[] m_accTrain;
    double[] m_accTest;
    double[] m_elasticity;

    int m_maxLength;
    int m_numLabels;

    String m_data;
    String m_method;


    Log(String method, int n) {
        m_num = n;
        m_status = new Status[n];
        m_accTrain = new double[n];
        m_accTest = new double[n];
        m_elasticity = new double[n];
        m_method = method;
    }

    void set(String data, int maxLength, int numLabels) {
        m_data = data;
        m_maxLength = maxLength;
        m_numLabels = numLabels;
    }

    void set(Status status, double accTrain, double accTest, double elasticity) {
        m_status[m_next] = status;
        m_accTrain[m_next] = 100 * accTrain;
        m_accTest[m_next] = 100 * accTest;
        m_elasticity[m_next] = elasticity;
        m_next = (m_next + 1) % m_num;
    }

    void setPointer(int i) {
        m_next = i % m_num;
    }

    void info() {

        System.out.println();
        System.out.println("Legend:");
        System.out.println("\t 000: test accuracy");
        System.out.println("\t 111: train accuracy");
        System.out.println("\t 222: elasticity");
        System.out.println("\t 333: train loss");
        System.out.println("\t 444: train accuracy oscillations");
        System.out.println("\t 555: train accuracy differences");
        System.out.println("\t 666: max train accuracy");
        System.out.println("\t 555: max length of time series");
        System.out.println("\t 666: number of labels");
        System.out.println();

        String exp = " " + m_data + " " + m_method + " ";
        System.out.println("Results: ");
        System.out.println("000 acc_test   " + exp + Array.toString(m_accTest, "%5.2f "));
        System.out.println("111 acc_train  " + exp + Array.toString(m_accTrain, "%5.2f "));
        System.out.println("222 elasticity " + exp + Array.toString(m_elasticity, "%5.2f "));


        System.out.print("333 min_loss   " + exp);
        for (int i = 0; i < m_num; i++) {
            System.out.format("%7.5f ", m_status[i].minLoss);
        }
        System.out.println();
        System.out.print("444 avg_oscill " + exp);
        for (int i = 0; i < m_num; i++) {
            System.out.format("%5.2f ", 100 * m_status[i].meanOscillations());
        }
        System.out.println();

        System.out.print("555 avg_diff   " + exp);
        for (int i = 0; i < m_num; i++) {
            System.out.format("%5.2f ", 100 * m_status[i].meanDifferences());
        }
        System.out.println();

        System.out.print("666 max_acc_tr " + exp);
        for (int i = 0; i < m_num; i++) {
            System.out.format("%5.2f ", 100 * m_status[i].maxAcc);
        }
        System.out.println();

        System.out.print("777 max_length " + exp);
        for (int i = 0; i < m_num; i++) {
            System.out.format("%d ", m_maxLength);
        }
        System.out.println();

        System.out.print("888 num_labels " + exp);
        for (int i = 0; i < m_num; i++) {
            System.out.format("%d ", m_numLabels);
        }
        System.out.println();

    }
}
