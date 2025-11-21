package com.filtering.AdaPIC;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.util.FastMath;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays; 
import java.util.List;
import java.util.stream.Collectors;

/**
 * Implements the Adaptive Power Iteration Clustering (APIC) algorithm, 
 * from the Adaptive Power Iteration Clustering paper.
 */
public class AdaptivePowerClustering {

    private final RealMatrix Wf; // Similarity matrix (Wf)
    private final int K;        // Number of clusters

    public AdaptivePowerClustering(RealMatrix Wf, int K) {
        this.Wf = Wf;
        this.K = K;
    }

    
    public static RealMatrix loadMatrixFromCsv(String filePath) throws IOException {
        List<double[]> rows = new ArrayList<>();
        
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                
                String[] values = line.trim().split(",");
                double[] row = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    row[i] = Double.parseDouble(values[i].trim());
                }
                rows.add(row);
            }
        }
        
        if (rows.isEmpty()) {
            throw new IOException("The CSV file is empty or could not be read.");
        }
        
        double[][] data = new double[rows.size()][];
        for (int i = 0; i < rows.size(); i++) {
            data[i] = rows.get(i);
        }

        return new Array2DRowRealMatrix(data);
    }
    
    public static RealMatrix zScoreNormalize(RealMatrix matrix) {
        // ... (Normalization logic remains the same) ...
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        RealMatrix normalizedMatrix = matrix.copy();

        Mean meanCalc = new Mean();
        StandardDeviation stdDevCalc = new StandardDeviation();

        for (int j = 0; j < cols; j++) {
            double[] column = matrix.getColumn(j);
            double mean = meanCalc.evaluate(column);
            double stdDev = stdDevCalc.evaluate(column);

            if (stdDev == 0.0) {
                for (int i = 0; i < rows; i++) {
                    normalizedMatrix.setEntry(i, j, 0.0);
                }
            } else {
                for (int i = 0; i < rows; i++) {
                    double normalizedValue = (matrix.getEntry(i, j) - mean) / stdDev;
                    normalizedMatrix.setEntry(i, j, normalizedValue);
                }
            }
        }
        System.out.println("Data matrix successfully Z-score normalized.");
        return normalizedMatrix;
    }

    public static List<String> loadUserIdMap(String filePath) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            return br.lines()
                     .map(String::trim)
                     .filter(line -> !line.isEmpty())
                     .collect(Collectors.toList());
        }
    }
    
    // --- APIC Methods ---

    /**
     * Orthogonalizes using Gram-Schmidt
     */
    private void gramSchmidtOrthogonalize(List<RealVector> vectors) {
        for (int j = 0; j < vectors.size(); j++) {
            RealVector vj = vectors.get(j);
            
            // 1. Orthogonalize vj against v1, v2, ..., v_{j-1}
            for (int i = 0; i < j; i++) {
                RealVector vi = vectors.get(i);
                // Projection of vj onto vi: (vj . vi) * vi
                double projectionScalar = vj.dotProduct(vi);
                RealVector projection = vi.mapMultiply(projectionScalar);
                vj = vj.subtract(projection);
            }
            
            // 2. Normalize vj
            double norm = vj.getNorm();
            if (norm > 1e-12) {
                vectors.set(j, vj.mapDivide(norm));
            } else {
                // Handle near-zero vector
                vectors.set(j, new ArrayRealVector(vj.getDimension())); 
            }
        }
    }


    /**
     * Estimates singular values (sigma and gamma) using the Rayleigh Quotient.
     * @param YbVectors The list of current vectors [y^1, ..., y^(K+1)].
     * @return A list containing the estimates: [sigma_1, ..., sigma_K, gamma_K].
     */
    private List<Double> estimateSingularValues(List<RealVector> YbVectors) {
        List<Double> estimates = new ArrayList<>();
        int numVectors = YbVectors.size(); // K + 1
        
        for (int i = 0; i < numVectors; i++) {
            RealVector yi = YbVectors.get(i);
            RealVector Wfyi = Wf.operate(yi); 
            // Estimate = y^T_i * Wf * y^i
            estimates.add(yi.dotProduct(Wfyi)); 
        }
        return estimates;
    }


    
     // Estimates Dynamic Principal Components
    private double computeDPE(List<Double> singularEstimates) {
        if (singularEstimates.size() != K + 1) return 0.0;

        double sumSigma = 0.0;
        for (int i = 0; i < K; i++) {
            sumSigma += singularEstimates.get(i);
        }
        
        double gammaHatK = singularEstimates.get(K);

        // dpe = (K / gamma_hat_K) * Sum_{i=1}^K sigma_hat_i
        double dpe = 0.0;
        if (FastMath.abs(gammaHatK) > 1e-9) {
             dpe = (double) K * sumSigma / FastMath.abs(gammaHatK);
        }

        return dpe;
    }
    
     // Calculates the dynamic threshold tau_t based on Eq. 13 of the paper.
    private double computeDynamicThreshold(List<Double> currentSingularEstimates, double previousSumSigma) {
        if (currentSingularEstimates.size() != K + 1) return Double.MAX_VALUE;

        double gammaHatK = currentSingularEstimates.get(K);
        
        double tau = Double.MAX_VALUE;
        if (FastMath.abs(gammaHatK) > 1e-9) {
             // Use the current gamma_K and the previous sum of sigmas
             tau = (double) K * previousSumSigma / FastMath.abs(gammaHatK);
        }
        return tau;
    }


    /**
     * Adaptive Power Iteration Clustering (APIC)
     * @return list of K estimated singular vectors
     */
    public List<RealVector> runBlockAPIC(int maxIterations) {
        int n = Wf.getRowDimension();
        int numVectors = K + 1; // K+1 vectors for the DPE calculation

        // Initialize Yb
        List<RealVector> YbVectors = new ArrayList<>();
        RandomGenerator rg = new Well19937c();
        for (int i = 0; i < numVectors; i++) {
            double[] initialData = new double[n];
            for (int j = 0; j < n; j++) {
                initialData[j] = rg.nextDouble();
            }
            YbVectors.add(new ArrayRealVector(initialData));
        }

        // Initial Orthogonalization and Estimation
        gramSchmidtOrthogonalize(YbVectors);
        List<Double> currentEstimates = estimateSingularValues(YbVectors);
        
        // Sum of the K dominant singular estimates from previous iteration (t-1).
        // For t=1, we use the initial estimates.
        double previousSumSigma = 0.0;
        for (int i = 0; i < K; i++) {
            previousSumSigma += currentEstimates.get(i);
        }
        
        double dpe = computeDPE(currentEstimates);
        System.out.printf("APIC Initialization: DPE = %.4f\n", dpe);

        final int STABILIZATION_STEPS = 10;

        for (int t = 1; t <= maxIterations; t++) {
            
            // Power Iteration (Yb_new = Wf * Yb_old)
            List<RealVector> YbNewVectors = new ArrayList<>();
            for (RealVector y_old : YbVectors) {
                YbNewVectors.add(Wf.operate(y_old));
            }
            
            // Orthogonalization, Proj and Norm
            gramSchmidtOrthogonalize(YbNewVectors);
            
            // Recalculate estimates for the new vectors (t)
            currentEstimates = estimateSingularValues(YbNewVectors);
            
            // Check Convergence
            dpe = computeDPE(currentEstimates);
            double tau_t = computeDynamicThreshold(currentEstimates, previousSumSigma);
            
            System.out.printf("APIC Iteration %d: DPE = %.4f, Threshold (tau_t) = %.4f\n", t, dpe, tau_t);
            
            if (t > STABILIZATION_STEPS && dpe >= tau_t) {
                System.out.printf("APIC converged at Iteration %d. DPE (%.4f) >= tau_t (%.4f).\n", t, dpe, tau_t);
                // Only first K vectors used
                return YbNewVectors.subList(0, K);
            }

            // Update
            YbVectors = YbNewVectors;
            // Update previousSumSigma for next threshold
            previousSumSigma = 0.0;
            for (int i = 0; i < K; i++) {
                previousSumSigma += currentEstimates.get(i);
            }
        }
        
        System.out.printf("APIC did not converge within %d iterations. Returning the K dominant singular vectors.\n", maxIterations);
        // Return only the first K vectors
        return YbVectors.subList(0, K);
    }
    
    // --- Final Clustering and Output Methods ---

    public void clusterAndOutput(List<RealVector> KDominantVectors, List<String> userIdMap) {
        int size = Wf.getRowDimension();
        int k = KDominantVectors.size();

        // 1. Create the Final Data Matrix Y (N x K)
        List<DoublePoint> finalDataPoints = new ArrayList<>();
        DoublePoint.resetCounter();
        
        for (int i = 0; i < size; i++) {
            double[] featureVector = new double[k];
            for (int j = 0; j < k; j++) {
                featureVector[j] = KDominantVectors.get(j).getEntry(i);
            }
            finalDataPoints.add(new DoublePoint(featureVector));
        }

        // K-Means Clustering (N by K dims)
        int maxKMeansIterations = 1000;
        System.out.printf("\n--- 4. Running K-Means on the N x K Eigenvector Feature Matrix (K=%d, maxIterations=%d) ---\n", k, maxKMeansIterations);
        
        RandomGenerator randomGenerator = new Well19937c();
        KMeansPlusPlusClusterer<DoublePoint> clusterer = 
            new KMeansPlusPlusClusterer<>(k, maxKMeansIterations, new EuclideanDistance(), randomGenerator);
        
        List<CentroidCluster<DoublePoint>> clusters = (List<CentroidCluster<DoublePoint>>)(List<?>) clusterer.cluster(finalDataPoints);

        // Display Results
        System.out.println("\n--- FINAL CLUSTERING RESULTS ---");
        for (int i = 0; i < clusters.size(); i++) {
            CentroidCluster<DoublePoint> cluster = clusters.get(i);
            // Gentroid in K-dimensions
            String centroidStr = Arrays.stream(cluster.getCenter().getPoint())
                                       .mapToObj(v -> String.format("%.4f", v))
                                       .collect(Collectors.joining(", "));
                                       
            System.out.printf("\nCluster %d (Size: %d, Centroid: [%s])\n", (i + 1), cluster.getPoints().size(), centroidStr);
            
            System.out.println("Users in this cluster (Index -> UserID):");
            List<DoublePoint> points = cluster.getPoints();
            points.sort((a, b) -> Integer.compare(a.getOriginalIndex(), b.getOriginalIndex()));
            
            for (DoublePoint point : points) {
                int originalIndex = point.getOriginalIndex();
                String userId = userIdMap.get(originalIndex);
                System.out.printf("  - %d -> %s\n", originalIndex, userId);
            }
        }
    }
}

// --- DoublePoint Class (Required for Apache Commons Math to track original index) ---

class DoublePoint implements org.apache.commons.math3.ml.clustering.Clusterable {
    private final double[] point;
    private final int originalIndex;
    private static int counter = 0;

    public DoublePoint(double[] point) {
        this.point = point;
        this.originalIndex = counter++;
    }

    @Override
    public double[] getPoint() {
        return point;
    }

    public int getOriginalIndex() {
        return originalIndex;
    }
    
    public static void resetCounter() {
        counter = 0;
    }
}
