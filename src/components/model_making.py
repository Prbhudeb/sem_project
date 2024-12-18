import os
import sys

from src.exception import CustomException

from sklearn.feature_extraction.text import CountVectorizer

from src.components.preprocessing import Preprocessing


class Model_Making:
    def __inti__(self):
        pass

    def model_building(self):
        try:
            cv = CountVectorizer(max_features=1100,stop_words='english')
            processed_data = Preprocessing.processing_data()
            vector = cv.fit_transform(processed_data['tags']).toarray()

            return vector
        except Exception as e:
            raise CustomException(e,sys)