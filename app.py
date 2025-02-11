from flask import Flask, request, render_template, jsonify
import sys
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os
import pandas as pd
from flask_cors import CORS

import psycopg2
from psycopg2.extras import RealDictCursor

from src.exception import CustomException
from src.components.prepare_similarity_matrix import Model_Making
from src.components.prepare_similarity_matrix import ModelMakingCourse
from src.logger import logging
from src.api_responce import api_response
# Load environment variables
load_dotenv()

application = Flask(__name__)
app = application

# Enable CORS
CORS(app)
# Database connection settings
DATABASE_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'sslmode': os.getenv('DB_SSLMODE'),
}

# Connect to the database
def connect_to_database():
    try:
        logging.info("Attempting to connect to the database.")
        connection = psycopg2.connect(**DATABASE_CONFIG)
        logging.info("Database connection successful.")
        return connection
    except psycopg2.Error as e:
        logging.error(f"Error connecting to the database: {e}")
        raise CustomException(f"Database connection error: {e}", sys)

# Fetch user data by username
def fetch_user_data(username):
    connection = None
    try:
        logging.info(f"Fetching data for username: {username}")
        connection = connect_to_database()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            # Query to fetch basic user data
            query1 = "SELECT id, username, email, first_name, last_name FROM auth_user WHERE username = %s"
            cursor.execute(query1, (username,))
            user_data1 = cursor.fetchone()
            logging.info(f"Basic user data fetched: {user_data1}")

            if not user_data1:
                logging.error(f"No user found for username: {username}")
                raise CustomException(f"User not found for username: {username}", sys)

            # Query to fetch user profile data
            query2 = "SELECT * FROM rec_system_userprofiledata WHERE user_id = %s"
            cursor.execute(query2, (username,))
            user_data2 = cursor.fetchone()
            logging.info(f"User profile data fetched: {user_data2}")

            if not user_data2:
                logging.warning(f"No profile data found for user ID: {username}")

            return user_data1, user_data2

    except CustomException as ce:
        logging.error(f"Custom exception occurred: {ce}")
        raise

    except psycopg2.Error as e:
        logging.error(f"Database query error: {e}")
        raise CustomException(f"Error fetching user data: {e}", sys)

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise CustomException(f"Unexpected error while fetching user data: {e}", sys)

    finally:
        if connection:
            connection.close()
            logging.info("Database connection closed.")


# Initialize model once
model_maker = Model_Making()
model_maker.model_building()

course_maker = ModelMakingCourse()
course_maker.model_building_course()

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/ml_api/<string:username>')
def ml_api(username):
    try:
        logging.info(f"API call for user: {username}")

        # Fetch user data
        user_data1, user_data2 = fetch_user_data(username)
        if not user_data1 or not user_data2:
            logging.error(f"User data not found for username: {username}")
            raise CustomException(f"User data not found for username: {username}", sys)

        logging.info(f"User data successfully fetched for: {username}")

        # Extract user-specific details
        interest_field = user_data2.get('interest_field', None)
        interest_domain = user_data2.get('interest_domain', None)
        programming_language = user_data2.get('programming_language', None)
        frameworks = user_data2.get('frameworks', None)
        cloud_and_database = user_data2.get('cloud_and_database', None)
        projects = user_data2.get('projects', None)
        achievements_and_awards = user_data2.get('achievements_and_awards', None)
        academic_year = user_data2.get('academic_year', None)
        branch = user_data2.get('branch', None)

        # Validate required fields
        if not all([interest_field, interest_domain, programming_language, frameworks]):
            logging.error(f"Incomplete user data for username: {username}")
            raise CustomException(f"Incomplete user data for username: {username}", sys)

        # Get recommendations
        logging.info(f"Fetching recommendations for: {username}")
        projects,descriptions,skills,index = model_maker.recommend_projects(
            input_skills=programming_language,
            input_framework=frameworks,
            input_tools=cloud_and_database,
            input_category=interest_field,
            input_domain=interest_domain
        )
        if not projects or not descriptions:
            logging.error("No recommendations found")
            raise CustomException("No recommendations found", sys)

        # Format results
        final_results = {
            "project": projects,
            "description": descriptions,
            "skills": skills,
            "index": index
        }
        
        final_results = pd.DataFrame(final_results)
        df_json = final_results.to_json(orient="records")
        logging.info(f"Recommendations successfully generated for: {username}")

        return api_response(success=True, message="Recommendations successfully generated",response_code = 200 ,data=df_json)

    except CustomException as ce:
        logging.error(f"Custom exception occurred: {ce}")
        return jsonify({"error": str(ce)}), 400


@app.route('/ml_index/<int:index>')
def project_details(index):
    try:
        # Fetch project details
        df = pd.read_csv('notebook/data/final_gen_data.csv')
        project_details = {
            "project_name": df.loc[index, 'Project Name'],
            "project_description": df.loc[index, 'Project Description'],
            "project_skills": df.loc[index, 'Skills Required']
        }

        # f_data = pd.DataFrame(project_details)

        if isinstance(project_details, dict):
            project_details = [project_details]  # Convert to list if it's a single dictionary

        try:
            f_data = pd.DataFrame(project_details)  # Now safe
            json_data = f_data.to_json(orient="records")  # Convert to JSON
        except Exception as e:
            raise CustomException(e, sys)
        
        return api_response(success=True, message="Recommendations successfully generated",response_code = 200 ,data=json_data)

    except Exception as e:
        raise CustomException(e, sys)


# @app.route('/predict_project', methods=['GET','POST'])
# def predict_project():
#     try:
#         if request.method == 'GET':
#             return render_template('index.html')
#         else:
#             # Collect input attributes
#             input_skills = request.form.get('skills', '').split(",") if request.form.get('skills') else []
#             input_framework = request.form.get('framework', '').split(",") if request.form.get('framework') else []
#             input_tools = request.form.get('tools', '').split(",") if request.form.get('tools') else []
#             input_category = request.form.get('category', '').split(",") if request.form.get('category') else []
#             input_domain = request.form.get('domain', '').split(",") if request.form.get('domain') else []

#             # Get recommendations
#             projects,descriptions,skills,index = model_maker.recommend_projects(
#                 input_skills=input_skills, #programming_language
#                 input_framework=input_framework,#frameworks
#                 input_tools=input_tools,#cloud_and_database
#                 input_category=input_category,#interest_field
#                 input_domain=input_domain #interest_domain
#             )
#             # for i, recommendation in enumerate(results, start=1):
#             #     print(f"Recommendation {i}:")Prbhudeb
#             #     print(f"Project Name: {recommendation[0]}")
#             #     print(f"Project Description: {recommendation[1]}")
#             #     print("-" * 40)  # Separator for better readability
#             # return render_template('index.html', project=results[0],description = results[1])

#             if projects and descriptions:
#                 return render_template('index.html', 
#                                      project=projects[0],
#                                      description=descriptions[0],
#                                      skills=skills[0],
#                                      index=index[0])
#             else:
#                 return render_template('index.html', 
#                                      project="No matching projects found",
#                                      description="")

#     except Exception as e:
#         raise CustomException(e, sys)
    
# @app.route('/course_api', methods=['POST'])
# def course():
#     skills_str = request.form.get('skills', '')  # Get raw comma-separated string
    
#     course,description,url = ModelMakingCourse.recommend_courses(
#         self=course_maker,
#         input_skills=skills_str,
#         input_domain='Computer Science'
#     )
#     final_results = {
#             "course": course,
#             "course_description": description,
#             "url":url
#     }
        
#     final_results = pd.DataFrame(final_results)
#     df_json = final_results.to_json(orient="records")

#     return api_response(success=True, message="Recommendations successfully generated",response_code = 200 ,data=df_json)


    
# @app.route('/course/<string:username>')
# def course_api(username):
#     try:
#         logging.info(f"API call for user: {username}")

#         # Fetch user data
#         user_data1, user_data2 = fetch_user_data(username)
#         if not user_data1 or not user_data2:
#             logging.error(f"User data not found for username: {username}")
#             raise CustomException(f"User data not found for username: {username}", sys)

#         logging.info(f"User data successfully fetched for: {username}")

#         # Extract user-specific details
#         interest_field = user_data2.get('interest_field', None)
#         interest_domain = user_data2.get('interest_domain', None)
#         programming_language = user_data2.get('programming_language', None)
#         frameworks = user_data2.get('frameworks', None)
#         cloud_and_database = user_data2.get('cloud_and_database', None)
#         projects = user_data2.get('projects', None)
#         achievements_and_awards = user_data2.get('achievements_and_awards', None)
#         academic_year = user_data2.get('academic_year', None)
#         branch = user_data2.get('branch', None)

#         # Validate required fields
#         if not all([interest_field, interest_domain, programming_language, frameworks]):
#             logging.error(f"Incomplete user data for username: {username}")
#             raise CustomException(f"Incomplete user data for username: {username}", sys)

#         # Get recommendations
#         logging.info(f"Fetching recommendations for: {username}")
#         skills = programming_language + ',' + frameworks + ',' + cloud_and_database + ',' + interest_field
#         course,course_description,url = ModelMakingCourse.recommend_courses(
#             self=course_maker,
#             input_skills=skills,
#             input_domain=interest_domain
#         )
#         if not course or not course_description:
#             logging.error("No recommendations found")
#             raise CustomException("No recommendations found", sys)

#         # Format results
#         final_results = {
#             "course": course,
#             "course_description": course_description,
#             "url":url
#         }
        
#         final_results = pd.DataFrame(final_results)
#         df_json = final_results.to_json(orient="records")
#         logging.info(f"Recommendations successfully generated for: {username}")

#         return api_response(success=True, message="Recommendations successfully generated",response_code = 200 ,data=df_json)

#     except CustomException as ce:
#         logging.error(f"Custom exception occurred: {ce}")
#         return jsonify({"error": str(ce)}), 400
    
# @app.route('/predict_course', methods=['GET','POST'])
# def predict_course():
#     try:
#         if request.method == 'GET':
#             return render_template('html_course.html')
#         else:
#             # Collect input attributes
#             input_skills = request.form.get('skills', '').split(",") if request.form.get('skills') else []
#             input_domain = request.form.get('domain', '').split(",") if request.form.get('domain') else []

#             # Get recommendations
#             course,course_descriptions,url = course_maker.recommend_courses(
#                 input_skills=input_skills, #programming_language
#                 input_domain=input_domain #interest_domain
#             )

#             # print(course,course_descriptions,url)

#             if course and course_descriptions:
#                 return render_template('html_course.html', 
#                                      course=course[0],
#                                      course_description=course_descriptions[0],
#                                      url=url[0]
#                 )
#             else:
#                 return render_template('html_course.html', 
#                                      course="No matching course found",
#                                      course_description="")

#     except Exception as e:
#         raise CustomException(e, sys)
    



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)