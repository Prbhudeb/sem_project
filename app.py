from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.components import model_making
from src.components import preprocessing

ps = PorterStemmer()
cv = CountVectorizer(max_features=1100,stop_words='english')

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict_project',methods=['GET','POST'])
def predict_project():
    if request.method=='GET':
        return render_template('index.html')
    else:
        input_skills=[request.form.get('skills').split(",")]
        input_framework=[request.form.get('framework').split(",")]
        input_tools=[request.form.get('tools').split(",")]
        input_category=[request.form.get('category').split(",")]
        input_domain=[request.form.get('domain').split(",")]

        input_tags_list = []
    
        # Add input attributes to input tags
        for attr in [input_skills, input_framework, input_tools, input_category, input_domain]:
            if attr:
                input_tags_list.extend([str(x).lower() for x in attr])
        
        # Stem input tags
        stemmed_input_tags = " ".join([ps.stem(tag) for tag in input_tags_list])

        # Vectorize input tags using the same vectorizer
        input_vector = cv.transform([stemmed_input_tags]).toarray()
        
        vector = model_making.Model_Making()
        processed_data = preprocessing.Preprocessing()

        # Calculate cosine similarity between input vector and all project vectors
        similarities = cosine_similarity(input_vector, vector)[0]
        index = sorted(list(enumerate(similarities[0])),reverse=True, key = lambda x:x[1])[1][0]
        project_name = preprocessing.Preprocessing().loc[index, 'Project Name']
        project_description = processed_data.loc[index, 'Project Description']

        results =  [project_name,project_description]

        return render_template('index.html',results=results)
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        