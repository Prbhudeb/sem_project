import os
import sys

from src.exception import CustomException

import pandas as pd
import numpy as np

from src.utils import steming


class Preprocessing:
    def __inti__(self):
        pass

    def processing_data(self):
        try:
            data = pd.read_csv('notebook/data/final_working_data.csv')
            data['tags'] = data['Project Description'] + data['Skills Required'] + data['Framework'] + data['Tools & Technologies'] + data['Categorized Category'] + data['Categorized Domain']
            processed_data = data[['Project Name','Project Description','tags']]

            processed_data['tags'] = processed_data['tags'].apply(steming)

            return processed_data
        except Exception as e:
            raise CustomException(e,sys)