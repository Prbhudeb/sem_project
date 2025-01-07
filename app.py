from flask import Flask, request, render_template, jsonify
import sys
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os

import psycopg2
from psycopg2.extras import RealDictCursor

from src.exception import CustomException
from src.components.prepare_similarity_matrix import Model_Making
from src.logger import logging

# Load environment variables
load_dotenv()

application = Flask(__name__)
app = application

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
            cursor.execute(query2, (user_data1['id'],))
            user_data2 = cursor.fetchone()
            logging.info(f"User profile data fetched: {user_data2}")

            if not user_data2:
                logging.warning(f"No profile data found for user ID: {user_data1['id']}")

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


# def connect_to_database():
#     try:
#         connection = psycopg2.connect(**DATABASE_CONFIG)
#         return connection
#     except psycopg2.Error as e:
#         print(f"Error connecting to the database: {e}")
#         return None

# # Fetch user data by username
# def fetch_user_data(username):
#     connection = connect_to_database()
#     if not connection:
#         return {"error": "Failed to connect to the database"}
    
#     try:
#         with connection.cursor(cursor_factory=RealDictCursor) as cursor:
#             query1 = "SELECT id, username, email, first_name, last_name FROM auth_user WHERE username = %s"
#             cursor.execute(query1, (username,))
#             user_data1 = cursor.fetchone() 
#             query2 = "SELECT * FROM rec_system_userprofiledata WHERE user_id = %s"

#             cursor.execute(query2, (username,))
#             user_data2 = cursor.fetchone() 
#             if user_data1:
#                 return user_data1, user_data2
#             else:
#                 return {"error": "User not found"}
#     except psycopg2.Error as e:
#         print(f"Error fetching user data: {e}")
#         return {"error": "Error fetching user data"}
#     finally:
#         connection.close()



# Initialize model once
model_maker = Model_Making()
model_maker.model_building()

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
        results = model_maker.recommend_projects(
            input_skills=programming_language,
            input_framework=frameworks,
            input_tools=cloud_and_database,
            input_category=interest_field,
            input_domain=interest_domain
        )
        if not results:
            logging.error("No recommendations found")
            raise CustomException("No recommendations found", sys)

        # Format results
        final_results = {
            "project": results[0],
            "description": results[1],
        }

        logging.info(f"Recommendations successfully generated for: {username}")
        return jsonify(final_results)

    except CustomException as ce:
        logging.error(f"Custom exception occurred: {ce}")
        return jsonify({"error": str(ce)}), 400


# @app.route('/ml_api/<string:username>')
# def ml_api(username):
#     user_data1,user_data2 = fetch_user_data(username)

#     interest_field = user_data2['interest_field']
#     interest_domain = user_data2['interest_domain']
#     programming_language = user_data2['programming_language']
#     frameworks = user_data2['frameworks']
#     cloud_and_database = user_data2['cloud_and_database']
#     projects = user_data2['projects']
#     achievements_and_awards = user_data2['achievements_and_awards']
#     academic_year = user_data2['academic_year']
#     branch = user_data2['branch']

#     # Get recommendations
#     results = model_maker.recommend_projects(input_skills=programming_language, input_framework=frameworks, input_tools=cloud_and_database, input_category=interest_field, input_domain=interest_domain)
    
#     final_results = {
#         "project":results[0],
#         "description":results[1],
#     }

#     return jsonify(final_results)


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
#             results = model_maker.recommend_projects(
#                 input_skills=input_skills, #programming_language
#                 input_framework=input_framework,#frameworks
#                 input_tools=input_tools,#cloud_and_database
#                 input_category=input_category,#interest_field
#                 input_domain=input_domain #interest_domain
#             )
#             # for i, recommendation in enumerate(results, start=1):
#             #     print(f"Recommendation {i}:")
#             #     print(f"Project Name: {recommendation[0]}")
#             #     print(f"Project Description: {recommendation[1]}")
#             #     print("-" * 40)  # Separator for better readability
#             # return render_template('index.html', project=results[0],description = results[1])

#             if results:
#                 return render_template('index.html', 
#                                      project=results[0],
#                                      description=results[1])
#             else:
#                 return render_template('index.html', 
#                                      project="No matching projects found",
#                                      description="")

#     except Exception as e:
#         raise CustomException(e, sys)

if __name__ == "__main__":
    # user_data1,user_data2 = fetch_user_data("rashi@18")
    # print(user_data1)
    # print(user_data2)
    app.run(host="0.0.0.0", debug=True)