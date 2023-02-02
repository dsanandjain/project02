import pickle
import json
import numpy as np
import pandas as pd
import config


class iris_data():
    def __init__(self,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm):
        self.SepalLengthCm = SepalLengthCm
        self.SepalWidthCm = SepalWidthCm
        self.PetalLengthCm = PetalLengthCm
        self.PetalWidthCm = PetalWidthCm


    def load_model(self):

        with open (config.MODEL_FILE_PATH,'rb') as f:
            
            self.model=pickle.load(f)

        with open (config.JSON_FILE_PATH,'r') as f:
            self.json_data=json.load(f)

        
    def get_predict_flower(self):

            self.load_model()


            arr = np.zeros(len(self.json_data['columns']))
            arr[0] = self.SepalLengthCm
            arr[1] = self.SepalWidthCm
            arr[2] = self.PetalLengthCm
            arr[3] = self.PetalWidthCm

            print('test array>>',arr)
            predicted_flower=self.model.predict([arr])[0]
            return predicted_flower

if __name__=='__main__':
    SepalLengthCm = 4.5
    SepalWidthCm = 2
    PetalLengthCm = 1.5
    PetalWidthCm = 5



    flower = iris_data(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
    specie=flower.get_predict_flower()
    print('predicted flower is in >>',specie)
