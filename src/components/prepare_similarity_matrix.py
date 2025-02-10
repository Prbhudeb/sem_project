import sys
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.exception import CustomException
from src.logger import logging
from src.components.prepare_processed_data import Preprocessing
from src.components.prepare_processed_data import PreprocessingCourse
from src.utils import lemmatize_text


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
            self.processed_data = preprocessor.processing_data_project()
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
                       input_domain=None, top_n=20):
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
            # from src.utils import steming
            # stemmed_input_tags = " ".join([steming(tag) for tag in input_tags_list])
            lemmatized_input_tags = lemmatize_text(" ".join(input_tags_list))

            # Vectorize input tags
            input_vector = self.count_vectorizer.transform([lemmatized_input_tags]).toarray()

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
            project_skills = []
            index = []

            for idx, score in similar_projects:
                project_name.append(self.processed_data.loc[idx, 'Project Name'])
                project_description.append(self.processed_data.loc[idx, 'Project Description'])
                project_skills.append(self.processed_data.loc[idx, 'Skills Required'])
                index.append(idx)

            return project_name, project_description, project_skills, index
        
        except Exception as e:
            logging.error(f"Error in project recommendation: {str(e)}")
            raise CustomException(e, sys)
        

class ModelMakingCourse:
    def __init__(self):
        self.vector = None
        self.processed_data = None
        self.similarity_matrix = None
        self.count_vectorizer = None

    def model_building_course(self):
        """
        Build the course recommendation model by processing data and computing similarity matrix.
        """
        try:
            preprocessor = PreprocessingCourse()
            new_df = preprocessor.preprocessing_data_course()

            cv = CountVectorizer(max_features=5000, stop_words='english')
            vectors = cv.fit_transform(new_df['tags']).toarray()
            similarity_matrix = cosine_similarity(vectors)

            return {
                'vector': vectors,
                'processed_data': new_df,
                'similarity_matrix': similarity_matrix,
                'cv': cv
            }

        except Exception as e:
            logging.error(f"Error in model building: {str(e)}")
            raise CustomException(e, sys)

    def recommend_courses(self, input_skills=None, input_difficulty=None, input_domain=None, top_n=20):
        """
        Recommend courses based on input attributes with randomness.
        """
        try:
            # Ensure model is built
            if self.vector is None or self.processed_data is None:
                model_data = self.model_building_course()
                self.vector = model_data['vector']
                self.processed_data = model_data['processed_data']
                self.similarity_matrix = model_data['similarity_matrix']
                self.count_vectorizer = model_data['cv']

            # Validate processed data
            required_columns = {'course_name', 'Course Description', 'Course URL'}
            if not required_columns.issubset(set(self.processed_data.columns)):
                raise ValueError("Processed data does not contain required columns")

            # Prepare input tags
            input_tags_list = [
                str(x).lower().strip()
                for attr in [input_skills, input_difficulty, input_domain]
                if attr and attr != ['']
                for x in attr
            ]

            if not input_tags_list:
                logging.warning("No valid input attributes provided for recommendation.")
                return [], [], []

            # Apply lemmatization
            lemmatized_input_tags = lemmatize_text(" ".join(input_tags_list))

            # Vectorize input tags
            input_vector = self.count_vectorizer.transform([lemmatized_input_tags]).toarray()

            # Calculate similarities
            similarities = cosine_similarity(input_vector, self.vector)[0]

            # Get top N similar courses
            similar_courses = sorted(
                enumerate(similarities),
                key=lambda x: x[1],
                reverse=True
            )[1:top_n + 6]  # Extra results for better randomness

            # Shuffle the top results for randomness
            random.shuffle(similar_courses)
            similar_courses = similar_courses[:top_n]

            course_name, course_description, course_url = [], [], []
            for idx, score in similar_courses:
                if idx < len(self.processed_data):
                    course_name.append(self.processed_data.loc[idx, 'course_name'])
                    course_description.append(self.processed_data.loc[idx, 'Course Description'])
                    course_url.append(self.processed_data.loc[idx, 'Course URL'])

            return course_name, course_description, course_url

        except Exception as e:
            logging.error(f"Error in course recommendation: {str(e)}")
            raise CustomException(e, sys)

