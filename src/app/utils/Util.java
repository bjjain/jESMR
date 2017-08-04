package app.utils;

import classifier.*;
import util.Msg;

public class Util {

    public static Classifier getClassifier(int typeClf, String opts) {
        if (typeClf == 0) {
            return new ElasticSoftmax(opts);
        } else if (typeClf == 1) {
            return new MaxLinSoftmax(opts);
        } else if (typeClf == 2) {
            return new MinElasticSoftmax(opts);
        } else if (typeClf == 3) {
            return new WarpedSoftmax(opts);
        } else if (typeClf == 4) {
            return new SemiElasticSoftmax(opts);
        } else {
            Msg.error("Error! Unknown type of classifier: %d.", typeClf);
        }
        return null;
    }
}
