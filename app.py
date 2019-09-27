# import cs50
import csv

from flask import Flask, jsonify, redirect, render_template, request
from helpers import main
import random
# Configure application
app = Flask(__name__)

if __name__ == "__main__":
    app.run(debug=True)

# Reload templates when they are changed
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.after_request
def after_request(response):
    """Disable caching"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/", methods=["GET"])
def get_index():
    return redirect("/form")


@app.route("/form", methods=["GET"])
def get_form():
    return render_template("form.html")


@app.route("/form", methods=["POST"])
def post_form():
    if not request.form.get("name") or not request.form.get("pclass") or not request.form.get("sex") or not request.form.get("sibsp") or not request.form.get("parch") or not request.form.get("embarked"):
        return render_template("error.html", message="Please fill all the values.")
    with open("test.csv", "a") as file:
        writer = csv.DictWriter(file, fieldnames=["PassengerId", "Name", "Pclass", "Sex", "Age", "Cabin", "SibSp", "Parch", "Ticket", "Fare", "Embarked"])
        writer.writerow({"PassengerId": random.randrange(899, 1000), "Name": request.form.get("name"), "Pclass": request.form.get("pclass"), "Sex": request.form.get("sex"),
        "Age": request.form.get("age"), "Cabin": request.form.get("cabin"), "SibSp": request.form.get("sibsp"), "Parch": request.form.get("parch"),
        "Ticket": random.randrange(4000), "Fare": random.randrange(90), "Embarked": request.form.get("embarked")})
    return redirect("/sheet")

@app.route("/sheet", methods=["GET"])
def get_sheet():
    if main():
        return render_template("survived.html")
    else:
        return render_template("dead.html")
