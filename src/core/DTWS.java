package core;

import util.Msg;

class DTWS  {
	
	double[][] m_s;
	int m_nx;
	int m_ny;
	
	double s(Pattern p, Pattern q) {
		if(!p.isTimeSeries()) {
			Msg.error("Error: First Argument must be a sequence.");
		}
		
		double[] x = p.seq();
		double[][] y = q.seqmatrix();
	
		m_nx = x.length;
		m_ny = y[0].length;

		if(m_nx == 0 || m_ny == 0) {
			return Double.NaN;
		}
		
		if(m_nx > y.length) {
			y = stretch(y, m_nx);
			q.set(y);
		}
		
		int i, j;
		m_s = new double[m_nx][m_ny];
		m_s[0][0] = x[0]*y[0][0];
		for(i = 1; i < m_nx; i++) {
			m_s[i][0] = m_s[i-1][0] + x[i]*y[i][0];
		}
		for(j = 1; j < m_ny; j++) {
			m_s[0][j] = m_s[0][j-1] + x[0]*y[0][j];
		}
		
		double max = Double.NEGATIVE_INFINITY;
		for (i = 1; i < m_nx; i++) {
			for (j = 1; j < m_ny; j++) {
				max = m_s[i-1][j];
				if(m_s[i][j-1] > max) {
					max = m_s[i][j-1];
				}
				if(m_s[i-1][j-1] > max) {
					max = m_s[i-1][j-1];
				}
				m_s[i][j] = max + x[i]*y[i][j];
			}
		}
		return m_s[m_nx-1][m_ny-1];
	}
	
	private double[][] stretch(double[][] y, int nx) {
		int n = y.length;
		int m = y[0].length;
		double[][] _y = new double[nx][m];
		for(int i = 0; i < n; i++) {
			System.arraycopy(y[i], 0, _y[i], 0, m);
		}
		return _y;
	}

	int[][] path() {
		
		final int LEFT 	= 0;
		final int UP 	= 1;
		final int DIAG 	= 2;
		
		int[][] path = new int[m_nx+m_ny][2];
		int ix = m_nx - 1;
		int iy = m_ny - 1;
	
		path[0][0] = ix;
		path[0][1] = iy;
		
		int n = 1;
		int direction = -1;
		double max = 0;
		while(ix != 0 || iy != 0) {
			
			if(ix == 0) {
				iy--;
			} else if(iy == 0) {
				ix--;
			} else {
				max = m_s[ix-1][iy-1];
				direction = DIAG;
				if(m_s[ix][iy-1] > max) {
					max = m_s[ix][iy-1];
					direction = UP;
				}
				if(m_s[ix-1][iy] > max) {
					direction = LEFT;
				}
				if(direction == LEFT) {
					ix--;
				} else if(direction == UP) {
					iy--;
				} else {
					ix--;
					iy--;
				}
			}
			path[n][0] = ix;
			path[n][1] = iy;
			n++;
		}
		
		// reverse order and trim
		int[][] p = new int[n][2];
		int m = n-1;
		for(int i = 0; i < n; i++) {
			p[i] = path[m-i];
		}
		return p;
	}

	public String toString() {
		return "DTWS";
	}
}
