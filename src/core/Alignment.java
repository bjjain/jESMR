package core;

/**
 * Implements an optimal dynamic time warping alignment between pattern 
 * <code>x</code> and <code>w</code>. The first pattern <code>x</code> must be 
 * a time series pattern. The second pattern <code>w</code> is a matrix pattern
 * representing the weight matrix of an elastic linear classifier.
 */
public final class Alignment {
	
	private DTWS m_similarity;
	private Pattern m_x;
	private Pattern m_w;
	private int[][] m_path;
	private double m_sim;

	/**
	 * Creates an optimal alignment of time series <code>x</code> and weight 
	 * matrix <code>w</code>.
	 * 
	 * @param x time series pattern
	 * @param w pattern
	 */
	public Alignment(Pattern x, Pattern w) {
		this.m_similarity = new DTWS();
		align(x, w);
	}

	private void align(Pattern x, Pattern w) {
		m_w = w;
		m_x = x;
		m_sim = 0;
		m_sim = m_similarity.s(x, w);
		m_path = m_similarity.path();
	}
	
	/**
	 * Updates and returns weight matrix <code>w</code>. The update rule is
	 * <pre>
	 *     w = ax*x + aw*w 
	 * </pre>
	 * where the parameters <code>ax</code> and <code>aw</code> are scalars 
	 * that weight to which extent each pattern contributes to the sum. Addition
	 * follows along the optimal warping path of this alignment. 
	 * 
	 * @param ax weight of pattern <code>x</code>
	 * @param aw weight of pattern <code>w</code>
	 * @return updated pattern <code>w = ax*x + aw*w </code>
	 */
	public Pattern update(double ax, double aw) {
		double[] seqx = m_x.seq();
		double[][] seqmatw = m_w.seqmatrix();
		int n = m_path.length;
		int i, j;
		for(int t = 0; t < n; t++) {
			i = m_path[t][0];
			j = m_path[t][1];
			seqmatw[i][j] = ax*seqx[i] + aw*seqmatw[i][j];
		}
		m_w.set(seqmatw);
		return m_w;
	}
	
	/**
	 * Updates and returns weight matrix pattern using L1-regularization. The 
	 * update rule is
	 * <pre>
	 *     w = w + ax*x + l1*d|w| 
	 * </pre>
	 * where <code>ax</code> weights the contribution of time series 
	 * <code>x</code> and <code>l1</code> is the regularization parameter. The 
	 * quantity <code>d|w|</code> refers to the subgradient of the L1-norm 
	 * <code>|w|</code> with clipping. Regularization 
	 * considers only weights along the optimal warping path.
	 * 
	 * @param ax  weight of pattern <code>x</code>
	 * @param l1 L1-regularization parameter
	 * @return updated pattern <code>w = w + ax*x + l1*d|w| </code>
	 */
	public Pattern updateL1(double ax, double l1) {
		double[] seqx = m_x.seq();
		double[][] seqmatw = m_w.seqmatrix();
		int n = m_path.length;
		int i, j;
		for(int t = 0; t < n; t++) {
			i = m_path[t][0];
			j = m_path[t][1];
			seqmatw[i][j] += ax*seqx[i];
			double wij = seqmatw[i][j];
			if(wij > 0) {
				seqmatw[i][j] = Math.max(0, wij - l1);
			} else if (wij < 0) {
				seqmatw[i][j] = Math.min(0, wij + l1);
			}
		}
		m_w.set(seqmatw);
		return m_w;
	}
	
	/**
	 * Updates and returns weight matrix pattern using L1-regularization. The 
	 * update rule is
	 * <pre>
	 *     w = w + l1*d|w| 
	 * </pre>
	 * where <code>l1</code> is the regularization parameter. The quantity 
	 * <code>d|w|</code> refers to the subgradient of the L1-norm 
	 * <code>|w|</code> with clipping. Regularization 
	 * considers only weights along the optimal warping path.
	 * 
	 * @param l1 L1-regularization parameter
	 * @return updated pattern <code>w = w + l1*d|w| </code>
	 */
	public Pattern updateL1(double l1) {
		double[][] seqmatw = m_w.seqmatrix();
		int n = m_path.length;
		int i, j;
		for(int t = 0; t < n; t++) {
			i = m_path[t][0];
			j = m_path[t][1];
			double yij = seqmatw[i][j];
			if(yij > 0) {
				seqmatw[i][j] = Math.max(0, yij - l1);
			} else if (yij < 0) {
				seqmatw[i][j] = Math.min(0, yij + l1);
			}
		}
		m_w.set(seqmatw);
		return m_w;
	}
	
	/**
	 * Updates and returns weight matrix pattern using L2-regularization. The 
	 * update rule is
	 * <pre>
	 *     w = w + l2*w 
	 * </pre>
	 * where <code>l2</code> is the regularization parameter. Regularization 
	 * considers only weights along the optimal warping path.
	 * 
	 * @param l2 L2-regularization parameter
	 * @return updated pattern <code>w = w + l2*w </code>
	 */
	public Pattern updateL2(double l2) {
		double[][] seqmatw = m_w.seqmatrix();
		int n = m_path.length;
		int i, j;
		for(int t = 0; t < n; t++) {
			i = m_path[t][0];
			j = m_path[t][1];
			seqmatw[i][j] *= l2;
		}
		m_w.set(seqmatw);
		return m_w;
	}
	
	/**
	 * Returns optimal warping path of this alignment. 
	 * @return optimal warping path
	 */
	public int[][] path() {
		return m_path;
	}
	
	/**
	 * Returns optimal dynamic time warping similarity of this alignment.
	 * @return optimal dynamic time warping similarity 
	 */
	public double s() {
		return m_sim;
	}


}
