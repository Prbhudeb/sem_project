from flask import Flask, request, render_template
import sys

from src.exception import CustomException
from src.components.prepare_similarity_matrix import Model_Making

application = Flask(__name__)
app = application

# Initialize model once
model_maker = Model_Making()
model_maker.model_building()

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict_project', methods=['GET','POST'])
def predict_project():
    try:
        if request.method == 'GET':
            return render_template('index.html')
        else:
            # Collect input attributes
            input_skills = request.form.get('skills', '').split(",") if request.form.get('skills') else []
            input_framework = request.form.get('framework', '').split(",") if request.form.get('framework') else []
            input_tools = request.form.get('tools', '').split(",") if request.form.get('tools') else []
            input_category = request.form.get('category', '').split(",") if request.form.get('category') else []
            input_domain = request.form.get('domain', '').split(",") if request.form.get('domain') else []

            # Get recommendations
            results = model_maker.recommend_projects(
                input_skills=input_skills,
                input_framework=input_framework,
                input_tools=input_tools,
                input_category=input_category,
                input_domain=input_domain
            )
            # for i, recommendation in enumerate(results, start=1):
            #     print(f"Recommendation {i}:")
            #     print(f"Project Name: {recommendation[0]}")
            #     print(f"Project Description: {recommendation[1]}")
            #     print("-" * 40)  # Separator for better readability
            # return render_template('index.html', project=results[0],description = results[1])

            if results:
                return render_template('index.html', 
                                     project=results[0],
                                     description=results[1])
            else:
                return render_template('index.html', 
                                     project="No matching projects found",
                                     description="")

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)