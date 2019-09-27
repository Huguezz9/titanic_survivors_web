Titanic Survivors

Flask has to be installed and run before to work. Command line in the directory: flask run.

Be sure to download first the following libraries.
Python3:
  Numpy, Pandas, Seaborn, Scikit-learn

Simple website built with html, css, jinja that request the user to buy an imaginary ticket for the famous RMS Titanic trip. The values are written into a csv file
thanks to a python program that uses flask as a framework. The program imports a function called main() from helpers.py, the function reads 2 csv files: the first one,
called "train.csv" it's composed of a list made with the information from original Titanic's passengers; the second one, "test.csv" is a list of the imaginary tickets
bought by users in our website.

The function main uses the libraries mentioned before to run a Machine Learning code that checks the information from the original passengers, analyzes the chances of surviving
based on different features like age, sex, class of the ticket, cabin number etc. Then it passes the trained program to the user's values and returns "True" if the user would
survive and "False" if it wouldn't. Based on the result of this function flask will render "survived.html" or "dead.html" respectively.

Made by Hugo Palomar.
2019
Madrid, Spain

Inspiration from Kaggle ML competition: "Titanic: Machine Learning from disaster"
