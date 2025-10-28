#Test to ensure that the main functionality works as expected

import unittest
import pandas as pd
import os
from FileLoader import smart_load
from Core import clean_file as clean_csv
from Core import find_similar_col_to_remove

class TestDataSystem(unittest.TestCase):
    def setUp(self):
        """
        Runs before each test. Prepare a self made sample dataset and 
        expexted metadata used across said test.
        """
        # path to sample data using the repo root
        self.test_file = "test_data.csv"
        # expected raw columns as they appear in the CSV header
        self.expected_columns = ['Name', 'Age', 'City', 'Date Joined', 'Salary']
        # expected normalized column names the cleaner should produce
        self.expected_clean_columns = ['name', 'age', 'city', 'date_joined', 'salary']

        # ensures file exists for local manual runs
        if not os.path.exists(self.test_file):
            self.skipTest(f"Test data file not found: {self.test_file}")

    def test_file_loading(self):
            """
            Verify loader returns a pandas.DataFrame with the expected number
            of columns with column names present.
            """
            # basic function of returning DataFrames
            df = smart_load(self.test_file)
            self.assertIsInstance(df, pd.DataFrame, "Loader must return pandas DataFrame")

            # column checks (be robust to whitespace and case differences)
            loaded_cols = [c.strip() for c in df.columns.tolist()]
            loaded_cols_norm = [c.lower() for c in loaded_cols]
            for expected in self.expected_columns:
                self.assertIn(expected.lower(), loaded_cols_norm,
                              f"Expected column '{expected}' not found in loaded file.")

            #basic row/length check
            self.assertGreater(len(df), 0, "Loaded DataFrame should contain at least one row.")

    def test_data_cleaning(self):
            """
            verify the cleaning process:
            - removes dupes
            - fills or removes nulls,
            - normalizes column names
            - converts types (dates, salary to numeric)
            """
            cleaned = clean_csv(self.test_file)

            #type and basic structure
            self.assertIsInstance(cleaned, pd.DataFrame, "clean_csv should return DataFrame")

            # column normalization checks
            self.assertListEqual(sorted(cleaned.columns.tolist()), sorted(self.expected_clean_columns),
                                 "cleaned columns do not match expected normalized names.")
            
            #check that no missing values remain
            self.assertEqual(cleaned.isnull().sum().sum(), 0, "Cleaned DataFrame should have no missing values.")

            #duplicates are removed
            self.assertEqual(cleaned.duplicated().sum(), 0, "Cleaned DataFrame should have no duplicates.")

            #data type checks (date)joined is datetime, 'salary' is numeric
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned['date_joined']),
                            "'date_joined' should be datetime type after cleaning.")
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned['salary']),
                            "'salary' should be numeric type after cleaning.")
            
            #running cleaner again should not change data
            cleaned_again = clean_csv(self.test_file)
            #compare shapes and values, not every dataset will be the same
            self.assertEqual(cleaned.shape, cleaned_again.shape, "cleaner should not affect shape on re-run.")
            self.assertListEqual(sorted(cleaned.columns.tolist()), sorted(cleaned_again.columns.tolist()),
                                 "cleaner should not affect columns on re-run.")
            
    def test_removing_similar_col(self): 
        """
            Verify that columns with high correlation are removed and a new data set is given
        """

        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5], 
            'B': [2, 4, 6, 8, 10], # perfect correlation with A
            'C': [5, 3, 1, 0, -1] # not correlated 
        })

        # Save to temporary CSV file
        temp_file = 'temp_test_correlation.csv'
        df.to_csv(temp_file, index=False)

        try:
            removed_col = find_similar_col_to_remove(temp_file)

            #check if B was dropped due to high correlation
            self.assertNotIn('B', removed_col.columns)

            #check to make sure that A and C are still in the data
            self.assertIn('A', removed_col.columns)
            self.assertIn('C', removed_col.columns)
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def run_test():
        """
        Runs tests when this file is executed directly.
        """
    print("Starting tests...")
    unittest.main(argv=['first-arg-is-ignored'], verbosity=2, exit=False)
    print("Tests completed.")


if __name__ == '__main__':
    unittest.main()
            
