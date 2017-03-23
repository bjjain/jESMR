package core;

import util.Array;

/**
 * This class represents a Pattern. A pattern consists of a matrix and an a 
 * class label. Matrices either represent time series or are used for computing
 * elastic embeddings. 
 * 
 * @author jain
 *
 */
public class Pattern {
	
	final String strLabel;		// original label
	int intLabel;			    // internal label
	double[][] seq;
	boolean isTimeSeries;

	/**
	 * Creates a labeled time series pattern.
	 * @param x time series
	 * @param label class label
	 */
	public Pattern(double[] x, String label) {
		this.seq = new double[1][];
		this.seq[0] = x;
		this.strLabel = label;
		this.isTimeSeries = true;
	}
	
	/**
	 * Creates an unlabeled matrix-pattern. 
	 * @param x matrix
	 */
	public Pattern(double[][] x) {
		this(x, "", 0);
	}
	
	/**
	 * Creates a labeled matrix-pattern. 
	 * @param x matrix
	 * @param label class label
	 */
	private Pattern(double[][] x, String strLabel, int label) {
		this.seq = x;
		this.strLabel = strLabel;
		this.intLabel = label;
		this.isTimeSeries = x != null && x.length == 1;
	}
	
	/**
	 * Sets pattern matrix.
	 * @param seqmat pattern matrix
	 */
	public void set(double[][] seqmat) {
		this.seq = seqmat;
	}
	
	/**
	 * Returns true if and only if matrix represents time series.
	 * @return true iff time series
	 */
	public boolean isTimeSeries() {
		return isTimeSeries;
	}
	
	/**
	 * Returns pattern matrix.
	 * @return pattern matrix
	 */
	public double[][] seqmatrix() {
		return seq;
	}
	
	/**
	 * Returns time series as first row of pattern matrix. Should only be 
	 * called if this pattern represents a time series.
	 * 
	 * @return time series
	 */
	public double[] seq() {
		return seq[0];
	}
	
	/**
	 * Returns class label.
	 * @return class label
	 */
	public int label() {
		return intLabel;
	}
	
	/**
	 * Returns length of time series as number of colums of pattern matrix.
	 * @return length
	 */
	public int length() {
		if(seq == null) {
			return 0;
		}
		return seq[0].length;
	}
	
	/**
	 * Performs z-normalization with zero mean uand unit standard deviation if
	 * this pattern is a time series.
	 */
	public void normalize() {
		if(isTimeSeries) {
			seq[0] = Array.normalize(seq[0]);
		}
	}
	
	/**
	 * Returns L1-norm of this matrix pattern along specified path. The path 
	 * must be a feasible warping path within this matrix patterns.
	 * 
	 * @param path warping path
	 * @return L1-norm of this pattern along a path
	 */
	public double l1(int[][] path) {
		double l1 = 0;
		int i, j;
		int n = path.length;
		for(int t = 0; t < n; t++) {
			i = path[t][0];
			j = path[t][1];
			l1 += Math.abs(seq[i][j]);
		}
		return l1;
	}
		
	/**
	 * Returns L2-norm of this matrix pattern along specified path. The path 
	 * must be a feasible warping path within this matrix patterns.
	 * 
	 * @param path warping path
	 * @return L2-norm of this pattern along a path
	 */
	public double l2(int[][] path) {
		double l2 = 0;
		int i, j;
		int n = path.length;
		for(int t = 0; t < n; t++) {
			i = path[t][0];
			j = path[t][1];
			l2 += seq[i][j]*seq[i][j];
		}
		return l2;
	}
			
	/**
	 * Returns deep copy of this pattern.
	 * @return copy of this pattern
	 */
	public Pattern cp() {
		return new Pattern(Array.cp(seq), strLabel, intLabel); 
	}
}
