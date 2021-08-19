from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dataset.db'
db = SQLAlchemy(app)


class Todo(db.Model):
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

    def __repr__(self):
        return '<Task %r>' % self.id


@app.route('/')
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
