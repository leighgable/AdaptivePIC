package com.filtering.AdaPIC;

import com.filtering.AdaPIC.CsvDataReader;
import com.filtering.AdaPIC.Timer;
import com.filtering.AdaPIC.RatingRecord;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.Set;


public class App 
{
    public static void main( String[] args )
    {
        String filePath = null;
        int linesToRead = 5;
        
        CsvDataReader dataReader = new CsvDataReader();
        if (args.length == 0) {
            System.out.println("Usage: java AdaPIC <path_to_csv_file> [-n <number_of_lines>]");
            return;
        }
        filePath = args[0];

        if (args.length > 1) {
            if (args[1].equals("-n")) {
                if (args.length > 2) {
                    try {
                        linesToRead = Integer.parseInt(args[2]);
                        if (linesToRead <= 0) {
                            System.out.println("Number of lines must be a positive integer. Using default (5).");
                            linesToRead = 5;
                        }
                    } catch (NumberFormatException e) {
                        System.out.println("Invalid number of lines provided. Usings default (5).");
                    }
            } else {
                    System.out.println("Missing number of lines after -n. Using default (5).");
            }
        } else {
                    System.out.println("Unknown option: " + args[1] + ". Ignoring and using default lines (5).");
                }
            }

        System.out.println("Reading CSV from: " + filePath);
        System.out.println("Printing first " + linesToRead + " lines.");

        CsvDataReader reader = new CsvDataReader();
        List<String[]> headData = reader.readCsvHeadOrAll(filePath, linesToRead);
        for (String[] row : headData) {
            System.out.println(Arrays.toString(row));
        }

        List<String[]> allData = reader.readCsvHeadOrAll(filePath, null);

        System.out.println(allData.size() + " records loaded -- all data.");
                
       
        List<RatingRecord> allElectronicsRatings = RatingRecord.buildRecords(allData);

        int minReviews = 30;
        List<RatingRecord> filteredData = CsvDataReader.filterByMinRatings(allElectronicsRatings, minReviews);

        System.out.println(filteredData.size() + " records loaded -- users with over " + minReviews + " ratings.");
        

        
        // Timer t = Timer.log("OpenCSV read time: ", TimeUnit.MILLISECONDS);

          
           
    }

}

        
