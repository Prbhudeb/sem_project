import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import lemmatize_text

class Preprocessing:
    def __init__(self):
        """
        Initialize the Preprocessing class with data path.
        """
        self.data_path_project = 'notebook/data/final_gen_data.xlsx'
        self.processed_data = None

    def processing_data_project(self):
        """
        Process the project data by combining tags and applying stemming.
        
        Returns:
        - processed_data (pd.DataFrame): Processed DataFrame with selected columns
        """
        try:
            logging.info("Starting data preprocessing...")

            # Validate data file path
            if not os.path.exists(self.data_path_project):
                raise FileNotFoundError(f"Data file not found at {self.data_path_project}")

            # Read the CSV file
            data = pd.read_excel(self.data_path_project)
            logging.info(f"Loaded data from {self.data_path_project}")

            # Combine tags from multiple columns with error handling
            data['tags'] = (
                data['Project Description'].fillna('') + ' ' +
                data['Skills Required'].fillna('') + ' ' +
                data['Framework'].fillna('') + ' ' +
                data['Tools & Technologies'].fillna('') + ' ' +
                data['Categorized Category'].fillna('') + ' ' +
                data['Categorized Domain'].fillna('')
            )

            # Select relevant columns
            self.processed_data = data[['Project Name', 'Project Description', 'tags','Skills Required']].copy()

            # Apply stemming to tags
            # we can also use lemmatization instead of stemming but due to performance issue we are using stemming
            # lemmatization is more accurate than stemming but it is slower than stemming 
            self.processed_data['tags'] = self.processed_data['tags'].apply(lemmatize_text)
            logging.info("Completed tag stemming")

            logging.info(f"Processed data shape: {self.processed_data.shape}")

            return self.processed_data

        except Exception as e:
            logging.error(f"Error in data processing: {str(e)}")
            raise CustomException(e, sys)


class PreprocessingCourse:
    def __init__(self, data_path_course='notebook/data/Coursera.csv'):
        self.data_path_course = data_path_course

    def preprocessing_data_course(self):
        """
        Process course data by cleaning and creating a 'tags' column.
        """
        try:
            logging.info("Starting data preprocessing...")

            if not os.path.exists(self.data_path_course):
                raise FileNotFoundError(f"Data file not found at {self.data_path_course}")

            # Read the CSV file
            data = pd.read_csv(self.data_path_course)
            logging.info(f"Loaded data from {self.data_path_course}, shape: {data.shape}")

            # Select necessary columns
            selected_columns = ['Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'Course URL']
            if not all(col in data.columns for col in selected_columns):
                raise ValueError(f"CSV file missing required columns: {selected_columns}")

            data = data[selected_columns].copy()

            # Clean text data using regex
            text_cleaning_rules = [
                (r'\s+', ' '),  # Replace multiple spaces with a single space
                (r'[,]+', ','),  # Reduce multiple commas
                (r'[_:]', ''),  # Remove underscores and colons
                (r'[()]', '')  # Remove parentheses
            ]

            for col in ['Course Name', 'Course Description']:
                for pattern, replacement in text_cleaning_rules:
                    data[col] = data[col].str.replace(pattern, replacement, regex=True)

            data['Skills'] = data['Skills'].str.replace('[()]', '', regex=True)

            # Create 'tags' column
            data['tags'] = (
                data['Course Name'] + " " +
                data['Difficulty Level'] + " " +
                data['Course Description'] + " " +
                data['Skills']
            )

            # Rename columns
            new_df = data[['Course Name', 'tags', 'Course URL', 'Course Description']].copy()
            new_df.rename(columns={'Course Name': 'course_name'}, inplace=True)

            # Lowercase and clean 'tags' column
            new_df['tags'] = new_df['tags'].str.lower().str.replace(',', ' ', regex=False)

            # Apply lemmatization
            new_df['tags'] = new_df['tags'].apply(lemmatize_text)
            logging.info(f"Processed data shape: {new_df.shape}")

            return new_df

        except Exception as e:
            logging.error(f"Error in data processing: {str(e)}")
            raise CustomException(e, sys)

