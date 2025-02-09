import os
import sys

# import dill
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from src.exception import CustomException

ps = PorterStemmer()

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def steming(text):
    try:
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
