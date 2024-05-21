import pickle
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from params import store_knn_classifier

def load_model():
    return pickle.load(open(store_knn_classifier, 'rb'))
def predict_with_PHOC(phocs, model):
    result = model.predict(phocs)
    return result