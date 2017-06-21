package preprocess;

import data.Dataset;
import data.Standardization;

/**
 * Created by jain on 04.05.17.
 */
public class Preprocessor {

    Parameter m_params;
    Standardization m_std;

    public Preprocessor(String opts) {
        m_params = new Parameter(opts);
    }


    public void fit(Dataset X) {
        if (0 < m_params.zc) {
            m_std = new Standardization();
            m_std.fit(X);
        }
    }

    public Dataset apply(Dataset X) {

        if (0 < m_params.zc) {
            X = m_std.normalize(X);
        }

        if (0 < m_params.zr) {
            X.normalize();
        }

        if (0 < m_params.a) {
            X.augment();
            X.setBias(m_params.b);
        }
        int p0 = m_params.p0;
        int p1 = m_params.p1;

        if (0 < p0 || 0 < p1) {
            X.pad(p0, p1);
        }
        return X;
    }


}
