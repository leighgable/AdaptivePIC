package com.filtering.AdaPIC;

import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.ArrayList;

public class RatingRecord {
    private String userId;
    private String itemId;
    private double rating;

    public RatingRecord(String userId, String itemId, double rating) {
        this.userId = userId;
        this.itemId = itemId;
        this.rating = rating;
    }

    public String getUserId() {
        return userId;
    }
    public String getItemId() {
        return itemId;
    }
    public double getRating() {
        return rating;
    }
    public void setUserId(String userId) {
        this.userId = userId;
    }
    public void setItemId(String itemId) {
        this.itemId = itemId;
    }
    public void setRating(double rating) {
        this.rating = rating;
    }
    public static List<RatingRecord> buildRecords(List<String[]> allData) {
        // Initialize the list to store the final objects
        List<RatingRecord> records = new ArrayList<>();
        
        // Assuming the first row is a header and should be skipped
        boolean isHeader = true; 

        for (String[] row : allData) {
            if (isHeader) {
                isHeader = false;
                continue; // Skip the header row
            }

            // Ensure the row has the expected number of columns (e.g., 4: userId, itemId, rating, timestamp)
            if (row.length < 3) {
                System.err.println("Skipping malformed row: insufficient columns.");
                continue;
            }

            try {
                String userId = row[0]; // Assumes userId is column 0
                String itemId = row[1]; // Assumes itemId is column 1
                
                double rating = Double.parseDouble(row[2]); // Assumes rating is column 2

                // Create the new RatingRecord object
                RatingRecord record = new RatingRecord(userId, itemId, rating);
                
                // Add the validated record to the list
                records.add(record);

            } catch (NumberFormatException e) {
                System.err.println("Skipping row due to invalid number format for rating: " + row[2]);
            } catch (Exception e) {
                System.err.println("Skipping row due to general error: " + e.getMessage());
            }
        }
        return records;
    }
}
