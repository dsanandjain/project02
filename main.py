
from flask import Flask, jsonify,render_template,request
from model.utils import iris_data
import config

app=Flask(__name__)

@app.route('/')
def hello_world():
     print('welcome to  flower sspecies prediction')
     return render_template("index.html")


@app.route('/predict_flower', methods=["GET","POST"])
def get_predict_flower():

    if request.method =="GET":

        print('we are in get method')

        SepalLengthCm = eval(request.args.get("SepalLengthCm"))
        SepalWidthCm = eval(request.args.get("SepalWidthCm"))
        PetalLengthCm = eval(request.args.get("PetalLengthCm"))
        PetalWidthCm = eval(request.args.get("PetalWidthCm"))

        flower_pred = iris_data(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
        species=flower_pred.get_predict_flower()

        return render_template("index.html", prediction = species)


    else:
        print('we are in post method')

        SepalLengthCm = eval(request.form.get("SepalLengthCm"))
        SepalWidthCm = eval(request.form.get("SepalWidthCm"))
        PetalLengthCm = eval(request.form.get("PetalLengthCm"))
        PetalWidthCm = eval(request.form.get("PetalWidthCm"))
        

        flower_pred = iris_data(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
        species=flower_pred.get_predict_flower()
        return render_template("index.html", prediction = species)


if __name__=='__main__':
    app.run(host='0.0.0.0' , port= config.PORT_NUMBER, debug=True)

