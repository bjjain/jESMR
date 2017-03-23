package core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import util.Msg;
import util.Rand;

/**
 * Represents a dataset of labeled patterns. Class labels are internally 
 * represented as integers from <code>0</code> to <code>n-1</code>, where 
 * <code>n</code> is the number of distinct class labels. 
 * 
 * @author jain
 *
 */
@SuppressWarnings("serial")
public class Patterns extends ArrayList<Pattern> {	
	
	private static final HashMap<String, Integer> LABELS = new HashMap<String, Integer>();
	private static Integer[] labels = {};
	String name;
	int maxLength;
		
	/**
	 * Creates empty dataset with specified name.
	 * 
	 * @param name of dataset
	 */
	public Patterns(String name) {
		super();
		this.name = name;	
		this.maxLength = 0;
	}
	
	/**
	 * Adds pattern to this dataset.
	 * @param x pattern
	 * @return true if this dataset has changed
	 */
	public boolean add(Pattern x) {
		String strLabel = x.strLabel;
		if(!LABELS.containsKey(strLabel)) {
			LABELS.put(strLabel, LABELS.size());
		}
		x.intLabel = LABELS.get(strLabel);
		maxLength = Math.max(maxLength, x.length());
		return super.add(x);
	}
	
	/**
	 * Adds a dataset.
	 * @param X a dataset
	 */
	public void add(Collection<? extends Pattern> X) {
		if(X == null) {
			return;
		}
		for(Pattern x : X) {
			add(x);
		}
	}
	
	/**
	 * Returns a deep copy of this dataset.
	 * @return copy of this dataset
	 */
	public Patterns cp() {
		Patterns X = new Patterns(name);
		for(Pattern x : this) {
			X.add(x.cp());
		}
		return X;
	}
	
	/**
	 * Returns pattern matrix of pattern at specified index.
	 * @param index index of pattern
	 * @return pattern matrix
	 */
	public double[][] pattern(int index) {
		return get(index).seq;
	}
	
	/**
	 * Returns label of pattern at specified index. 
	 * @param index of pattern
	 * @return class label
	 */
	public int label(int index) {
		return get(index).intLabel;
	}
	
	/**
	 * Returns ordered set of class labels.
	 * @return class labels
	 */
	public int[] labels() {
		if(labels.length != LABELS.size()) {
			labels = LABELS.values().toArray(new Integer[LABELS.size()]);
			Arrays.sort(labels);
		}
		int n = labels.length;
		int[] l = new int[n];
		for(int i = 0; i < n; i++) {
			l[i] = labels[i];
		}
		return l;
	}
	
	/**
	 * Returns maximum length over all time series in this dataset.
	 * @return maximum length
	 */
	public int maxLength() {
		return maxLength;
	}
	
	/**
	 * Returns name of this dataset.
	 * @return name
	 */
	public String name() {
		return name;
	}
	
	
	/**
	 * Sets name of this dataset.
	 */
	public void setName(String name) {
		this.name = name;
	}
	
	/**
	 * Z-Normalization of all time series patterns in this dataset.
	 */
	public void normalize() {
		for(Pattern x : this) {
			x.normalize();
		}
	}
	
	/**
	 * Returns partition of this datasets into subsets of identical class.
	 * @return class-wise partition of this dataset
	 */
	public Patterns[] splitClasswise() {
		int numLabels = LABELS.size();
		Patterns[] folds = new Patterns[numLabels];
		for(int i = 0; i < numLabels; i++) {
			folds[i] = new Patterns(name);
		}
		for (Pattern x : this) {
			int label = x.intLabel;
			if(label < 0 || numLabels <= label) {
				label = LABELS.get(x.strLabel);
				x.intLabel = label;
			}
			folds[label].add(x);
		}
		return folds;
	}
	
	
	/**
	 * Splits this dataset into two folds. The parameter <code>percentage</code>
	 * specifies the number of patterns of the second fold as the percentage over
	 * all patterns of this dataset.
	 * 
	 * @param percentage size of second fold
	 * @return two-fold partition of this dataset
	 */
	public Patterns[] split(double percentage) {
		if(percentage < 0 || 1 < percentage) {
			Msg.error("Invalid split value: %f.", percentage);
		}
		if(percentage == 0) {
			return new Patterns[] {this, null};
		}
		return split((int) Math.round(percentage*size()));
	}
	
	Patterns[] split(int n) {
		if(n < 0 || size() < n) {
			Msg.error("Invalid split value: %d.", n);
		}
		if(n == size()) {
			return new Patterns[] {this, null};
		}
		int numFolds = 2;
		Patterns[] fold = new Patterns[numFolds];
		for (int i = 0; i < numFolds; i++) {
			fold[i] = new Patterns(name);
		}
		
		Patterns[] split = splitClasswise();
		int nPatterns = size();
		int nLabels = split.length;
		double[] w = new double[nLabels];
		for(int i = 0; i < nLabels; i++) {
			w[i] = ((double)split[i].size())/nPatterns;
			w[i] *= (double)n;
			w[i] = Math.round(w[i]);
		}	
		
		for (int i = 0; i < nLabels; i++) {
			int n0 = (int)w[i];
			int n1 = split[i].size();
			int[] pi = Rand.nextPermutation(n1);
			for(int j = 0; j < n0; j++) {
				fold[0].add(split[i].get(pi[j]));
			}
			for(int j = n0; j < n1; j++) {
				fold[1].add(split[i].get(pi[j]));
			}
		}
		return fold;
	}
	
	public Patterns[] splitFolds(int numFolds) {
		
		if(numFolds < 1) {
			Msg.error("Error: Invalid number of folds: %d.", numFolds);
		}
		Patterns[] fold = new Patterns[numFolds];
		for (int i = 0; i < numFolds; i++) {
			fold[i] = new Patterns(name);
		}
		
		if(numFolds >= size()) {
			numFolds = size();
			for (int i = 0; i < numFolds; i++) {
				fold[i].add(this.get(i));
			}
			return fold;
		}
		
		Patterns[] split = splitClasswise();
		int n = split.length;
		for (int i = 0; i < n; i++) {
			int m = split[i].size();
			int[] index = Rand.nextPermutation(m);
			for(int j = 0; j < m; j++) {
				int k = j%numFolds;
				fold[k].add(split[i].get(index[j]));
			}
		}
		return fold;
	}
}
