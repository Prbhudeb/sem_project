import os
import sys
import numpy as np
import pandas as pd
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.exception import CustomException
from src.logger import logging
from src.components.prepare_processed_data import Preprocessing

class Model_Making:
    def __init__(self):
        """
        Initialize the Model_Making class.
        """
        self.count_vectorizer = None
        self.processed_data = None
        self.vector = None
        self.similarity_matrix = None

    def model_building(self):
        """
        Build the vectorization model for project tags.
        
        Returns:
        - Dictionary containing vector, vectorizer, and processed data
        """
        try:
            logging.info("Starting model building process...")

            # Create Preprocessing instance
            preprocessor = Preprocessing()

            # Get processed data
            self.processed_data = preprocessor.processing_data()
            logging.info(f"Processed data shape: {self.processed_data.shape}")
            
            self.processed_data = self.processed_data.reset_index(drop=True)

            # Initialize and fit CountVectorizer
            self.count_vectorizer = CountVectorizer(
                max_features=1100,
                stop_words='english'
            )

            # Transform tags to vector
            self.vector = self.count_vectorizer.fit_transform(
                self.processed_data['tags']
            ).toarray()

            # Calculate similarity matrix
            self.similarity_matrix = cosine_similarity(self.vector)

            logging.info(f"Vector shape: {self.vector.shape}")

            return {
                'vector': self.vector,
                'count_vectorizer': self.count_vectorizer,
                'processed_data': self.processed_data,
                'similarity_matrix': self.similarity_matrix
            }

        except Exception as e:
            logging.error(f"Error in model building: {str(e)}")
            raise CustomException(e, sys)
    
    def recommend_projects(self, input_skills=None, input_framework=None, 
                       input_tools=None, input_category=None, 
                       input_domain=None, top_n=5):
        """
        Recommend projects based on input attributes with randomness.
        
        Parameters:
        - Various input attributes for project recommendation
        - top_n: Number of top recommendations to return
        
        Returns:
        - List of recommended project details
        """
        try:
            # Ensure model is built
            if self.vector is None or self.processed_data is None:
                model_data = self.model_building()
                self.vector = model_data['vector']
                self.processed_data = model_data['processed_data']
                self.similarity_matrix = model_data['similarity_matrix']

            # Prepare input tags
            input_tags_list = []
            for attr in [input_skills, input_framework, input_tools, input_category, input_domain]:
                if attr and attr != ['']:
                    input_tags_list.extend([str(x).lower().strip() for x in attr])
            
            # Stem input tags
            from src.utils import steming
            stemmed_input_tags = " ".join([steming(tag) for tag in input_tags_list])

            # Vectorize input tags
            input_vector = self.count_vectorizer.transform([stemmed_input_tags]).toarray()

            # Calculate similarities
            similarities = cosine_similarity(input_vector, self.vector)[0]

            # Get top N similar projects
            similar_projects = sorted(
                list(enumerate(similarities)), 
                key=lambda x: x[1], 
                reverse=True
            )[1:top_n + 6]  # Get extra results to allow shuffling

            # Introduce randomness: shuffle the top results
            random.shuffle(similar_projects)

            # Pick the final top_n results
            similar_projects = similar_projects[:top_n]

            project_name = []
            project_description = []

            for idx, score in similar_projects:
                project_name.append(self.processed_data.loc[idx, 'Project Name'])
                project_description.append(self.processed_data.loc[idx, 'Project Description'])

            return project_name, project_description
        
        except Exception as e:
            logging.error(f"Error in project recommendation: {str(e)}")
            raise CustomException(e, sys)

    # def recommend_projects(self, input_skills=None, input_framework=None, 
    #                        input_tools=None, input_category=None, 
    #                        input_domain=None, top_n=5):
    #     """
    #     Recommend projects based on input attributes.
        
    #     Parameters:
    #     - Various input attributes for project recommendation
    #     - top_n: Number of top recommendations to return
        
    #     Returns:
    #     - List of recommended project details
    #     """
    #     try:
    #         # Ensure model is built
    #         if self.vector is None or self.processed_data is None:
    #             model_data = self.model_building()
    #             self.vector = model_data['vector']
    #             self.processed_data = model_data['processed_data']
    #             self.similarity_matrix = model_data['similarity_matrix']

    #         # Prepare input tags
    #         input_tags_list = []
    #         for attr in [input_skills, input_framework, input_tools, input_category, input_domain]:
    #             if attr and attr != ['']:
    #                 input_tags_list.extend([str(x).lower().strip() for x in attr])
            
    #         # Stem input tags
    #         from src.utils import steming
    #         stemmed_input_tags = " ".join([steming(tag) for tag in input_tags_list])

    #         # Vectorize input tags
    #         input_vector = self.count_vectorizer.transform([stemmed_input_tags]).toarray()

    #         # Calculate similarities
    #         similarities = cosine_similarity(input_vector, self.vector)[0]

    #         # Get top N similar projects
    #         #similar_projects = 
    #         index = sorted(
    #             list(enumerate(similarities)), 
    #             key=lambda x: x[1], 
    #             reverse=True
    #         )[1:6] # [1:top_n+1]  # Skip first index to avoid exact match

    #         # Prepare recommendations
    #         # recommendations = [
    #         #     [
    #         #         self.processed_data.iloc[idx]['Project Name'],
    #         #         self.processed_data.iloc[idx]['Project Description']
    #         #     ]
    #         #     for idx, score in similar_projects
    #         # ]

    #         # recommendations = []
    #         # for idx, score in similar_projects:
    #         #     project_name = self.processed_data.iloc[idx]['Project Name']
    #         #     project_description = self.processed_data.iloc[idx]['Project Description']
    #         #     recommendations.append({
    #         #         'name': project_name,
    #         #         'description': project_description,
    #         #         'similarity_score': float(score)
    #         #     })
            
    #         project_name = []
    #         project_description = []

    #         for idx, score in index:
    #             project_name.append(self.processed_data.loc[idx, 'Project Name'])
    #             project_description.append(self.processed_data.loc[idx, 'Project Description'])

    #         # project_name = self.processed_data.loc[index, 'Project Name']
    #         # project_description = self.processed_data.loc[index, 'Project Description']

    #         return project_name,project_description #recommendations

        # except Exception as e:
        #     logging.error(f"Error in project recommendation: {str(e)}")
        #     raise CustomException(e, sys)