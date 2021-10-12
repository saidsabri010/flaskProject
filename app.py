import numpy as np
import os
from flask import Flask, render_template, url_for, request, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_login import login_user, logout_user, login_required
from flask_login import LoginManager, UserMixin
from model import KerasClassifier
from keras.models import load_model
from flask import send_from_directory
from ann_visualizer.visualize import ann_viz

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:''@localhost/flask'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'secret-key'
db = SQLAlchemy(app)

# a = KerasClassifier()
# a.run_tuner()
model = load_model("model.h5")

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)


class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pregnancies = db.Column(db.Integer, nullable=False)
    glucose = db.Column(db.Integer, nullable=False)
    bloodPressure = db.Column(db.Integer, nullable=False)
    skinThickness = db.Column(db.Integer, nullable=False)
    insulin = db.Column(db.Integer, nullable=False)
    bMI = db.Column(db.Integer, nullable=False)
    diabetesPedigreeFunction = db.Column(db.Integer, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    outcome = db.Column(db.Integer, nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow())

    def __init__(self, pregnancies, glucose, bloodPressure, skinThickness,
                 insulin, bMI, diabetesPedigreeFunction, age, outcome):
        self.pregnancies = pregnancies
        self.glucose = glucose
        self.bloodPressure = bloodPressure
        self.skinThickness = skinThickness
        self.insulin = insulin
        self.bMI = bMI
        self.diabetesPedigreeFunction = diabetesPedigreeFunction
        self.age = age
        self.outcome = outcome


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(11), nullable=False)
    email = db.Column(db.String(255), nullable=False, unique=True)
    password = db.Column(db.String(15), nullable=False)

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password

    def __repr__(self):
        return '<Data %r>' % self.id


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/')
def index():
    return render_template("main.html")


@app.route('/model')
@login_required
def model1():
    return render_template("index.html")


@app.route('/diabetes')
@login_required
def diabetes():
    data = Data.query.all()
    return render_template("diabetes.html", patients=data)


@app.route('/plot')
@login_required
def plot():
    # ann_viz(model, title="Diabetes Neural Network ")
    filepath = os.path.abspath(os.getcwd())
    return send_from_directory(filepath, 'network.gv.pdf')


@app.route('/predictshow')
@login_required
def predictshow():
    return render_template("predict.html")


@app.route('/editshow/<id>/', methods=['GET', 'POST'])
@login_required
def editshow(id):
    data = Data.query.filter_by(id=id).all()
    return render_template("edit.html", patients=data)


@app.route('/delete/<id>/', methods=['GET', 'POST'])
@login_required
def delete(id):
    data = Data.query.get(id)
    db.session.delete(data)
    db.session.commit()
    return redirect(url_for('diabetes'))


@app.route('/edit', methods=['GET', 'POST'])
@login_required
def edit():
    if request.form == 'GET':
        return redirect(url_for("editshow"))
    data = Data.query.get(request.form.get('id'))
    data.pregnancies = request.form['Pregnancies']
    data.glucose = request.form['Glucose']
    data.bloodPressure = request.form['BloodPressure']
    data.skinThickness = request.form['SkinThickness']
    data.insulin = request.form['Insulin']
    data.bMI = request.form['BMI']
    data.diabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
    data.age = request.form['Age']
    db.session.commit()
    return redirect(url_for('diabetes'))


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = np.expand_dims(float_features, 0)
    prediction = model.predict(features)
    pregnancies = request.form['Pregnancies']
    glucose = request.form['Glucose']
    bloodPressure = request.form['BloodPressure']
    skinThickness = request.form['SkinThickness']
    insulin = request.form['Insulin']
    bMI = request.form['BMI']
    diabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
    age = request.form['Age']
    outcome = np.argmax(prediction[0])
    data = Data(pregnancies, glucose, bloodPressure, skinThickness, insulin, bMI,
                diabetesPedigreeFunction, age, outcome)
    db.session.add(data)
    db.session.commit()
    return redirect(url_for('diabetes'))


@app.route('/login')
def login():
    return render_template("login.html")


@app.route('/register')
def register():
    return render_template("register.html")


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/register1', methods=['GET', 'POST'])
def register1():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    password_confirm = request.form['confirm_password']
    user = User.query.filter_by(email=email).first()
    if user:
        flash('email is already existed !')
        return render_template("register.html")
    elif password != password_confirm:
        flash('password must match !')
    else:
        new_user = User(username, email, password)
        db.session.add(new_user)
        db.session.commit()
    return render_template("login.html")


@app.route('/login1', methods=['GET', 'POST'])
def login1():
    username = request.form['username']
    password = request.form['password']
    user = User.query.filter_by(username=username).first()
    if not user:
        flash('this username does not exist !')
        return render_template("register.html")
    elif password != user.password:
        flash('password is wrong, please try again  !')
        return render_template("login.html")
    login_user(user)
    return redirect(url_for('model1'))


if __name__ == "__main__":
    app.run(debug=True)
