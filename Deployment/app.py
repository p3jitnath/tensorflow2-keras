from flask import Flask, request, jsonify, render_template, session, url_for, redirect
from tensorflow.keras.models import load_model
from wtforms import TextField, SubmitField
from flask_wtf import FlaskForm
import numpy as np
import joblib


def return_predictions(model, scaler, sample_json):

    # sepal_length, sepal_width, petal_length, petal_width

    flower = []
    for key, value in sample_json.items():
        flower.append(value)
    flower = [flower]
    flower = scaler.transform(flower)
    class_ind = model.predict_classes(flower)[0]
    classes = np.array(['setosa', 'versicolor', 'virginica'])

    return classes[class_ind]


class FlowerForm(FlaskForm):
    sepal_length = TextField("Sepal Length")
    sepal_width = TextField("Sepal Width")
    petal_length = TextField("Petal Length")
    petal_width = TextField("Petal Width")
    submit = SubmitField("Analyze")


app = Flask(__name__)
app.config['SECRET_KEY'] = 'Alan_Turing'
flower_model = load_model('final_iris_model.h5')
flower_scaler = joblib.load('iris_scaler.pkl')


# @app.route("/", methods=["GET", "POST"])
# def index():
#     form = FlowerForm()
#     if form.validate_on_submit():
#         session['sepal_length'] = form.sepal_length.data
#         session['sepal_width'] = form.sepal_width.data
#         session['petal_length'] = form.petal_length.data
#         session['petal_width'] = form.petal_width.data

#         return redirect(url_for("prediction"))
#     return render_template('home.html', form=form)

@app.route("/status")
def status():
    return ""

# @app.route("/prediction")
# def prediction():
#     content = {}

#     content['sepal_length'] = float(session['sepal_length'])
#     content['sepal_width'] = float(session['sepal_width'])
#     content['petal_length'] = float(session['petal_length'])
#     content['petal_width'] = float(session['petal_width'])

#     result = return_predictions(flower_model, flower_scaler, content)
#     return render_template('prediction.html', result=result)

@app.route("/api/flower", methods=['POST'])
def flower_prediction():
    content = request.json
    result = return_predictions(flower_model, flower_scaler, content)
    return jsonify(result)


if __name__ == "__main__":
    app.run()