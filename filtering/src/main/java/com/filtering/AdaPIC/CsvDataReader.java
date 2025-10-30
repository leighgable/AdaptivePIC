package com.filtering.AdaPIC;

import com.filtering.AdaPIC.RatingRecord;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;
import com.opencsv.exceptions.CsvValidationException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.Map;
import java.util.stream.Collectors;

// TODO replace opencsv with apache
// import org.apache.commons.csv.CSVFormat;
// import org.apache.commons.csv.CSVParser;
// import org.apache.commons.csv.CSVRecord;

public class CsvDataReader {

  public List<String[]> readCsvHeadOrAll(String filePath, Integer numLinesToRead) {
    List<String[]> records = new ArrayList<>();

    try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
      
      String[] nextLine;
      int linesRead = 0;

      boolean readAll = (numLinesToRead == null || numLinesToRead <= 0);

      while ((nextLine = reader.readNext()) != null) {
        records.add(nextLine);
        linesRead++;

        if (!readAll && linesRead >= numLinesToRead) {
          break;
        }
      }
    } catch (IOException e) {
      System.err.println("Error reading CSV file '" + filePath + "': " + e.getMessage());
    } catch (CsvValidationException e) {
      System.err.println("Error validating CSV data in '" + filePath + "': " + e.getMessage());
    }
    return records;
  }

    public static List<RatingRecord> filterByMinRatings(List<RatingRecord> electronics, int minRatings) {
        
        // Groupby: todo -- in the database 
        Map<String, Long> userRatingCounts = electronics.stream()
            .collect(Collectors.groupingBy(
                RatingRecord::getUserId, // Group by userId
                Collectors.counting()    // Count records for each group
            ));

        // Filter Users with >= minRatings 
        Set<String> usersWithSufficientRatings = userRatingCounts.entrySet().stream()
            .filter(entry -> entry.getValue() >= minRatings)
            .map(Map.Entry::getKey) // Extract the userId (the Map key)
            .collect(Collectors.toSet());

        // Filter data
        List<RatingRecord> filteredElectronics = electronics.stream()
            .filter(record -> usersWithSufficientRatings.contains(record.getUserId()))
            .collect(Collectors.toList());

        return filteredElectronics;
    }

}
