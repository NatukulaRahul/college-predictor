import pandas as pd
import webbrowser
import tkinter as tk
from tkinterweb import TkinterWeb
from tkinter import *
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tkinter


data = pd.read_csv("/Users/natukularahul/Desktop/college-predictor/Admission_Prediction.csv")

data = data.drop('Serial No.', axis=1)
data.columns
x=data.drop('Chance of Admit ',axis=1)
y = data['Chance of Admit ']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test = sc.transform(x_test)

lr = LinearRegression()
lr.fit(x_train, y_train)

svm = SVR()
svm.fit(x_train, y_train)
rf = RandomForestRegressor()
rf.fit(x_train, y_train)

gr = GradientBoostingRegressor()
gr.fit(x_train, y_train)


y_pred1 = lr.predict(x_test)
y_pred2 = svm.predict(x_test)
y_pred3 = rf.predict(x_test)
y_pred4 = gr.predict(x_test)


score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)

final_data = pd.DataFrame({'Models':['LR','SVR','RF','GR'],'R2_SCORE':[score1,score2,score3,score4]})


joblib.dump(gr,'prediction_model')

model = joblib.load('prediction_model')

a = model.predict(sc.transform([[337,118,1,9.65]]))
b = model.predict(sc.transform([[330,112,3,8.65]]))
c = model.predict(sc.transform([[325,110,3,8.0]]))
d = model.predict(sc.transform([[228,108,3,7]]))
e = model.predict(sc.transform([[220,112,4,7]]))
f = model.predict(sc.transform([[200,105,5,7.5]]))
g = model.predict(sc.transform([[190,104,5,7]]))
h = model.predict(sc.transform([[155,100,6,7.0]]))
i = model.predict(sc.transform([[149,90,6,7.8]]))
j = model.predict(sc.transform([[146,90,7,6.80]]))
import tkinter as tk
import tkinter as tk
from PIL import Image, ImageTk
# Create a window for the login page
LARGEFONT = ("Verdana", 35)
root = tkinter.Tk()
# Set the geometry of the window
root.geometry("720x1080")


#heading od page
lab = tkinter.Label(root, text="COLLEGE PREDICTING" , font=LARGEFONT)
lab.pack()
# Create a label for the username
Gre_score = tkinter.Label(root, text="Enter Your GRE Score:")
Gre_score.pack()
# Create a text entry field for the username
Gre_entry = tkinter.Entry(root)
Gre_entry.pack()
# create
Tof_score = tkinter.Label(root, text="Enter Your TOFEL Score:")
Tof_score.pack()
# Create a text entry field for the username
Tof_entry = tkinter.Entry(root)
Tof_entry.pack()
#create
rank_score = tkinter.Label(root, text="Enter Your University Rating:")
rank_score.pack()
# Create a text entry field for the username
rank_entry = tkinter.Entry(root)
rank_entry.pack()
#create
cpga_score = tkinter.Label(root, text="Enter Your CPGA:")
cpga_score.pack()
# Create a text entry field for the username
cpga_entry = tkinter.Entry(root)
cpga_entry.pack()

# Create a submit button
submit_button = tkinter.Button(root, text="Submit")
submit_button.pack()

#define a callback funtion
def callback(url):
    webbrowser.open_new_tab(url)

# Bind the submit button to a function that will process the login information
def process_login_information():

    p1 = Gre_entry.get()
    p2 = Tof_entry.get()
    p3 = rank_entry.get()
    p4= cpga_entry.get()
    

    result = model.predict(sc.transform([[p1, p2, p3, p4]]))
    if result == a or result == b or result == c or result == d or result == e or result == f or result == g or result == h or result == i or result == j:
        # Open the second window
        second_window = tkinter.Tk()
        label = tkinter.Label(second_window)
        second_window.geometry("720x1080")

        if result == a:
            label = tkinter.Label(second_window, text="You May in Get Admissions")
            label.pack()
            label = tkinter.Button(second_window, text="1.Massachusetts Institute of Technology")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://mitadmissions.org/"))
            label.pack()
            label = tkinter.Button(second_window, text="2.Harvard University")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://college.harvard.edu/admissions"))


        elif result == b:
            label = tkinter.Label(second_window, text="You May in Get Admissions")
            label.pack()
            label = tkinter.Button(second_window, text="1.Massachusetts Institute of Technology")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://mitadmissions.org/"))
            label.pack()
            label = tkinter.Button(second_window, text="2.Harvard University")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://college.harvard.edu/admissions"))
            label = tkinter.Button(second_window, text="3.Stanford University")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://www.stanford.edu/admission/"))

        elif result == c:
            label = tkinter.Label(second_window, text="You May in Get Admissions")
            label.pack()
            label = tkinter.Button(second_window, text="1.Massachusetts Institute of Technology")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://mitadmissions.org/"))
            label.pack()
            label = tkinter.Button(second_window, text="2.Harvard University")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://college.harvard.edu/admissions"))
            label = tkinter.Button(second_window, text="3.Stanford University")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://www.stanford.edu/admission/"))

        elif result == d:
            label = tkinter.Label(second_window, text="You May in Get Admissions")
            label.pack()
            label = tkinter.Button(second_window, text="1.Massachusetts Institute of Technology")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://mitadmissions.org/"))
            label.pack()
            label = tkinter.Button(second_window, text="2.Harvard University")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://college.harvard.edu/admissions"))
            label = tkinter.Button(second_window, text="3.Stanford University")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://www.stanford.edu/admission/"))

        elif result == e:
            label = tkinter.Label(second_window, text="You May in Get Admissions")
            label.pack()
            label = tkinter.Button(second_window, text="1.purdue University")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://admissions.purdue.edu"))
            label.pack()
            label = tkinter.Button(second_window, text="2.yale university")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://www.yale.edu/admissions"))
            label = tkinter.Button(second_window, text="3. University of Pennsylvania")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://www.upenn.edu/admissions"))
        elif result == f:
            label = tkinter.Label(second_window, text="You May in Get Admissions")
            label.pack()
            label = tkinter.Button(second_window, text="1.purdue University")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://admissions.purdue.edu"))
            label.pack()
            label = tkinter.Button(second_window, text="2.yale university")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://www.yale.edu/admissions"))
            label = tkinter.Button(second_window, text="3. University of Pennsylvania")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://www.upenn.edu/admissions"))

        elif result == g:
            label = tkinter.Label(second_window, text="You May in Get Admissions")
            label.pack()
            label = tkinter.Button(second_window, text="1.purdue University")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://admissions.purdue.edu"))
            label.pack()
            label = tkinter.Button(second_window, text="2.yale university")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://www.yale.edu/admissions"))
            label = tkinter.Button(second_window, text="3. University of Pennsylvania")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://www.upenn.edu/admissions"))

        elif result == h:
            label = tkinter.Label(second_window, text="You May in Get Admissions")
            label.pack()
            label = tkinter.Button(second_window, text="1.purdue University")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://admissions.purdue.edu"))
            label.pack()
            label = tkinter.Button(second_window, text="2.yale university")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://www.yale.edu/admissions"))
            label = tkinter.Button(second_window, text="3. University of Pennsylvania")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://www.upenn.edu/admissions"))

        elif result == i:
            label = tkinter.Label(second_window, text="You May in Get Admissions")
            label.pack()
            label = tkinter.Button(second_window, text="1.dakota University")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://dsu.edu/admissions/index.html"))
            label.pack()
            label = tkinter.Button(second_window, text="2.west virginia university")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://admissions.wvu.edu"))
            label = tkinter.Button(second_window, text="3.rice University")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://admission.rice.edu/apply"))
        elif result == j:
            abel = tkinter.Label(second_window, text="You May in Get Admissions")
            label.pack()
            label = tkinter.Button(second_window, text="1.dakota University")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://dsu.edu/admissions/index.html"))
            label.pack()
            label = tkinter.Button(second_window, text="2.west virginia university")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://admissions.wvu.edu"))
            label = tkinter.Button(second_window, text="3.rice University")
            label.pack()
            label.bind("<Button-1>", lambda x: callback("https://admission.rice.edu/apply"))

        second_window.mainloop()
# Bind the submit button to a function that will open the second window
submit_button.configure(command=process_login_information)
# Run the main loop
root.mainloop()