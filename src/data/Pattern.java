package data;

public class Pattern {

    public int y;               // class label
    public double[] x;          // sequence

    public Pattern(double[] x, int y) {
        this.x = x;
        this.y = y;
    }

    public int length() {
        return x.length;
    }


}
