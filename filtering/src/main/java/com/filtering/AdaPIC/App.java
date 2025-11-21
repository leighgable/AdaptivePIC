package com.filtering.AdaPIC;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import com.filtering.AdaPIC.CsvDataReader;
import com.filtering.AdaPIC.Timer;
import com.filtering.AdaPIC.RatingRecord;
import com.filtering.AdaPIC.AdaptivePowerClustering;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.Set;
import java.util.Random;


public class App 
{
    public static void main( String[] args )
    {
        String matrixFilePath = "src/test/resources/similarity.csv"; 
        String mapFilePath = "src/test/resources/user_index_map.csv"; 
        int k = 4; // Number of clusters
        int maxIterations = 100; // Max Power Iterations (APIC)
        
        try (Timer t = Timer.log("AdaptivePIC Time: ", TimeUnit.MILLISECONDS)) {
            // Load Data
            System.out.println("--- Data Loading and Normalization ---");
            RealMatrix Wf_matrix = AdaptivePowerClustering.loadMatrixFromCsv(matrixFilePath);
            int size = Wf_matrix.getRowDimension();
            if (size == 0) return;
            System.out.println("Matrix loaded successfully. Dimensions: " + size + "x" + size);
            
            // Normalize 
            Wf_matrix = AdaptivePowerClustering.zScoreNormalize(Wf_matrix);

            System.out.println("\n--- Loading User ID Map ---");
            List<String> userIdMap = AdaptivePowerClustering.loadUserIdMap(mapFilePath);
            if (userIdMap.size() != size) {
                System.err.println("ERROR: Matrix size (" + size + ") does not match map size (" + userIdMap.size() + "). Check your data files.");
                return;
            }
            System.out.println("User ID map loaded successfully.");

            System.out.println("\n--- Running Adaptive Power Iteration Clustering (APIC) ---");
            AdaptivePowerClustering apic = new AdaptivePowerClustering(Wf_matrix, k);
            List<RealVector> KDominantVectors = apic.runBlockAPIC(maxIterations);
            
            apic.clusterAndOutput(KDominantVectors, userIdMap);
            
        } catch (IOException e) {
            System.err.println("\n--- FATAL I/O ERROR ---");
            System.err.println("Could not read the necessary files. Ensure Wf_matrix.csv and user_index_map.csv exist. Details: " + e.getMessage());
            e.printStackTrace();
            
        } catch (NumberFormatException e) {
            System.err.println("\n--- DATA PARSING ERROR ---");
            System.err.println("A value in the CSV file could not be parsed as a number. Details: " + e.getMessage());
            e.printStackTrace();
            
        } catch (Exception e) {
            System.err.println("\n--- AN UNEXPECTED ERROR OCCURRED ---");
            e.printStackTrace();
        }
        
        
    } // Main


} // App

        
