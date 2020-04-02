package transform;

import data.Dataset;
import data.Pattern;

public abstract class Transform {

    public Dataset transform(Dataset X) {
        Dataset T = new Dataset();
        for (Pattern p : X) {
            T.add(transform(p));
        }
        return T;
    }

    abstract Pattern transform(Pattern p);
}
