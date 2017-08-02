package app.utils;

import classifier.*;
import util.Msg;

/**
 * Created by jain on 22.06.17.
 */
public class Util {

    public static Classifier getClassifier(int typeClf, String opts) {
        if (typeClf == 0) {
            return new MaxSoftmax(opts);
        } else if (typeClf == 1) {
            return new PolySoftmax(opts);
        } else if (typeClf == 2) {
            return new MinSoftmax(opts);
        } else if (typeClf == 3) {
            return new MaxMinSoftmax(opts);
        } else if (typeClf == 4) {
            return new MinMaxSoftmax(opts);
        } else if (typeClf == 5) {
            return new DTWSoftmax(opts);
        } else if (typeClf == 6) {
            return new SemiMaxSoftmax(opts);
        } else {
            Msg.error("Error! Unknown type of classifier: %d.", typeClf);
        }
        return null;
    }
}
