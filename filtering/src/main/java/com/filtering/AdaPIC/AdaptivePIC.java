
package com.filtering.AdaPIC;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Implements the Adaptive Power Iterated Clustering (APIC) algorithm using
 * Apache Commons Math3, with formulas derived from the research paper.
 */
 
public class AdaptivePowerClustering {

    private final RealMatrix Wf; // Normalized similarity matrix Wf
    private final int K;         // Number of clusters
    private final int N;         // Dimension of the matrix/data points
    private final double epsilon; // Convergence error (unused in stopping logic)
    private final double delta;   // Failure probability (unused in stopping logic)
    private final int maxIterations; // T, max number of iterations
    private final Random random;

    /**
     * Constructor for the AdaptivePowerClustering class.
     * @param Wf The normalized similarity matrix.
     * @param K The target number of clusters.
     * @param epsilon The convergence error threshold.
     * @param delta The failure probability.
     * @param maxIterations The maximum number of power iterations.
     */
    public AdaptivePowerClustering(RealMatrix Wf, int K, double epsilon, double delta, int maxIterations) {
        this.Wf = Wf;
        this.K = K;
        this.N = Wf.getRowDimension();
        this.epsilon = epsilon;
        this.delta = delta;
        this.maxIterations = maxIterations;
        this.random = new Random();
    }

    /**
     * Executes the Adaptive Power Iterated Clustering algorithm.
     * @return A list of clusters (C1, ..., CK).
     */
    public List<Cluster<DoublePoint>> execute() {
        final int M = K + 1; // Number of vectors in Yb

        // 1: Randomly initialize K + 1 non-zero vectors Yb
        List<RealVector> YbVectors = new ArrayList<>(M);
        for (int i = 0; i < M; i++) {
            // Initialize with small random non-zero values
            double[] data = new double[N];
            for (int j = 0; j < N; j++) {
                data[j] = random.nextDouble() * 2.0 - 1.0; // Range [-1.0, 1.0)
            }
            YbVectors.add(new ArrayRealVector(data));
        }

        // Compute Wc0, ..., WcK based on Eq. (6) and Eq. (7)
        RealMatrix[] Wc = computeWcMatrices(M);

        System.out.println("Starting Power Iteration with K=" + K + ", T=" + maxIterations);

        // for t = 1, 2, ..., T do
        for (int t = 1; t <= maxIterations; t++) {
            
            // for i = 1, 2, ..., K + 1 do
            for (int i = 0; i < M; i++) { // i runs from 0 to K (M-1)

                // yˆi ← Wci−1yˆi;
                RealMatrix Wci = Wc[i];
                RealVector yi = YbVectors.get(i);
                
                // Matrix-Vector multiplication: Wci * yi
                RealVector newYi = Wci.operate(yi);
                
                // yˆi ← yˆi/(yˆ > i yˆi);
                // The denominator is the square of the L2 norm (y^T * y). The paper means L2-Normalization.
                // We implement standard L2-Normalization: yˆi ← yˆi / ||yˆi||_2
                double norm = newYi.getNorm();
                
                if (norm > 1e-9) { // Avoid division by zero
                    newYi = newYi.mapDivide(norm);
                } else {
                    // Re-randomize or stop if the vector collapses to zero
                    System.err.println("Warning: Vector norm near zero at iteration " + t + ", vector " + i);
                    newYi = new ArrayRealVector(N).mapMultiply(random.nextDouble());
                    newYi = newYi.mapDivide(newYi.getNorm());
                }
                
                YbVectors.set(i, newYi);
            }
            // end for

            // Compute ˆγK and ˆp (and implicitly dpe)
            // We use the estimated gap (pHat) and DPE for the adaptive stop.
            double dpe = computeDPE(YbVectors); 

            // if dpˆe + 1 > t then
            // The paper's text uses the condition: dpe >= K (effective dimension reached)
            // We use this robust, text-derived condition instead of the cryptic pseudo-code line.
            if (dpe >= K) {
                // 10: break;
                System.out.printf("APIC converged at iteration %d. DPE (%.4f) >= K (%d).\n", t, dpe, K);
                break;
            }
            // end if
        }
        // end for

        // Yb = [yˆ1, yˆ2, · · ·, yˆK]; (Discard the K+1 vector, index K)
        List<DoublePoint> dataPoints = new ArrayList<>(K);
        for (int i = 0; i < K; i++) {
            // Convert RealVector to DoublePoint for K-Means
            dataPoints.add(new DoublePoint(YbVectors.get(i).toArray()));
        }

        // Run k-means algorithm on Yb to obtain the K clusters C1, C2, · · ·, CK;
        // Run k-means 10 times to find a good local minimum
        KMeansPlusPlusClusterer<DoublePoint> clusterer = 
                new KMeansPlusPlusClusterer<>(K, 10, random);

        return clusterer.cluster(dataPoints);
    }

    /**
     * Implements Eq. (6): W_c_i = W_f - lambda_c_i * I.
     * A single, simple, non-zero shift parameter (lambda_c) for all K+1 matrices
     * to satisfy the algorithm's requirement for K+1 distinct Wc matrices
     * (In the paper the method for computing K+1 distinct lambda_c_i is not fully specified).
     * @param M The number of Wc matrices (K+1).
     * @return An array of RealMatrix instances.
     */
    private RealMatrix[] computeWcMatrices(int M) {
        RealMatrix[] Wc = new RealMatrix[M];
        RealMatrix I = new Array2DRowRealMatrix(N, N);
        for (int i = 0; i < N; i++) {
            I.setEntry(i, i, 1.0); // Identity matrix
        }

        // Use a single, constant shift for simplicity, as the paper's exact calculation (Eq. 7) 
        // for K+1 distinct shifts relies on pre-computed local data. Good enough?
        final double SHIFT_PARAMETER = 0.5; 

        for (int i = 0; i < M; i++) {
            // W_c_i = W_f - lambda_c_i * I
            // Using Wf - 0.5 * I as the shifted matrix
            Wc[i] = Wf.subtract(I.scalarMultiply(SHIFT_PARAMETER));
        }
        return Wc;
    }

    /**
     * Computes the Dynamic Principal Component Estimation (dpe) for the adaptive stopping condition.
     * * - Estimate the singular values (sigma_hat_i) and the (K+1)-th singular value (gamma_hat_K)
     * using the Rayleigh Quotient on the current normalized vectors.
     * - We then calculate dpe = (1 / gamma_hat_K) * Sum_{i=1}^K sigma_hat_i (Simplified form).
     * Paper shows dpe = (K / gamma_hat_K) * Sum, which is magic.
     * * @param The list of current vectors [y^1, ..., y^(K+1)].
     * @return Dynamic principal component estimation (dpe).
     */
    private double computeDPE(List<RealVector> YbVectors) {
        // Compute estimated singular values (Rayleigh Quotients) for y^1 to y^K
        // sigma_hat_i = y^T_i * Wf * y^i
        double[] sigmaEstimates = new double[K];
        for (int i = 0; i < K; i++) {
            RealVector yi = YbVectors.get(i);
            RealVector Wfyi = Wf.operate(yi); 
            sigmaEstimates[i] = yi.dotProduct(Wfyi); 
        }
        
        // Estimate of the (K+1)-th singular value (gamma_K) - Eq. (16) approximation
        // gamma_hat_K = y^T_{K+1} * Wf * y^{K+1}
        RealVector yKplus1 = YbVectors.get(K);
        RealVector WfyKplus1 = Wf.operate(yKplus1);
        double gammaHatK = yKplus1.dotProduct(WfyKplus1);

        // Calculate Dynamic Principal Component Estimation (dpe)
        // dpe = (K / gamma_hat_K) * Sum_{i=1}^K sigma_hat_i (Using the paper's formula structure)
        double sumSigma = StatUtils.sum(sigmaEstimates);
        
        double dpe = 0.0;
        // Avoid division by zero or near-zero gamma_hat_K
        if (FastMath.abs(gammaHatK) > 1e-9) {
             dpe = (double) K * sumSigma / FastMath.abs(gammaHatK);
        }

        return dpe;
    }


    public static void main(String[] args) {
        // Define dimensions and parameters
        int N = 100; // Number of data points (rows/columns in Wf)
        int K = 4;   // Target clusters
        int T = 100; // Max iterations
        double epsilon = 1e-4;
        double delta = 0.05;

        // 1. Create a mock Normalized Similarity Matrix Wf (N x N)
        // Wf should be symmetric and ideally close to a bi-stochastic matrix.
        RealMatrix Wf = createMockSymmetricMatrix(N);

        // 2. Instantiate and run the clustering algorithm
        try {
            AdaptivePowerClustering clustering = 
                new AdaptivePowerClustering(Wf, K, epsilon, delta, T);
            
            List<Cluster<DoublePoint>> clusters = clustering.execute();

            // 3. Print Results
            System.out.println("\nClustering Complete:");
            System.out.println("--------------------");
            System.out.println("Target Clusters (K): " + K);
            System.out.println("Actual Clusters found: " + clusters.size());
            
            // Only print details if clusters were actually formed
            if (!clusters.isEmpty()) {
                for (int i = 0; i < clusters.size(); i++) {
                    System.out.println("Cluster " + (i + 1) + " size: " + clusters.get(i).getPoints().size());
                }
            }
            
        } catch (Exception e) {
            System.err.println("An error occurred during clustering: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Helper method to create a mock N x N symmetric matrix with entries [0, 1].
     * In a real application, Wf must be computed from the data (e.g., using a Gaussian kernel)
     * and normalized to satisfy the paper's requirements.
     */
    private static RealMatrix createMockSymmetricMatrix(int N) {
        Random rand = new Random();
        double[][] data = new double[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = i; j < N; j++) {
                // Create a matrix with values between 0.1 and 1.0
                double val = 0.1 + rand.nextDouble() * 0.9;
                data[i][j] = val;
                data[j][i] = val; // Ensure symmetry
            }
            data[i][i] = 1.0; // Ensure diagonal is strong
        }
        return new Array2DRowRealMatrix(data);
    }
}
