import pickle
import json
import numpy as np


class select_flower():
    def __init__(self,data):
        self.data = data
       

    def import_files(self):

        with open("artifacts/iris_logistic_model.pkl","rb") as file:
            self.model = pickle.load(file)
       
        with open("artifacts/flower_type.json","r") as flower_file:
            client_given_val = json.load(flower_file)

        input_list = client_given_val['columns']
        self.input_list = input_list


    def predict_flower_type(self):
            self.import_files() 

            values = np.zeros(len(self.input_list))
            print (f"values = {values}") 

            values[0] = self.data["html_sl"]
            values[1] = self.data["html_sw"]
            values[2] = self.data["html_pl"]
            values[3] = self.data["html_pw"]
            
            print (f"values_out = {values}")

            predict_flower = self.model.predict([values])
            # print(predict_flower)

            if predict_flower ==0:
                flower = "Setosa"
                
            if predict_flower ==1:
                flower = "Versicolor" 
                
            if predict_flower ==2:
                flower = "Virginica"

            return flower
