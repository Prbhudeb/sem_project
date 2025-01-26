import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import steming

class Preprocessing:
    def __init__(self):
        """
        Initialize the Preprocessing class with data path.
        """
        self.data_path = 'notebook/data/intelligent_project_mapping.csv'
        self.processed_data = None

    def processing_data(self):
        """
        Process the project data by combining tags and applying stemming.
        
        Returns:
        - processed_data (pd.DataFrame): Processed DataFrame with selected columns
        """
        try:
            logging.info("Starting data preprocessing...")

            # Validate data file path
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found at {self.data_path}")

            # Read the CSV file
            data = pd.read_csv(self.data_path)
            logging.info(f"Loaded data from {self.data_path}")

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
            self.processed_data = data[['Project Name', 'Project Description', 'tags']].copy()

            # Apply stemming to tags
            # we can also use lemmatization instead of stemming but due to performance issue we are using stemming
            # lemmatization is more accurate than stemming but it is slower than stemming 
            self.processed_data['tags'] = self.processed_data['tags'].apply(steming)
            logging.info("Completed tag stemming")

            logging.info(f"Processed data shape: {self.processed_data.shape}")

            return self.processed_data

        except Exception as e:
            logging.error(f"Error in data processing: {str(e)}")
            raise CustomException(e, sys)
