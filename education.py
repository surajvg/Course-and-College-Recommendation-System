from flask import Flask,render_template,Markup,request,redirect,url_for
import pandas as pd
import numpy as np
#from IPython.display import HTML
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import time
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
#from sklearn.svm import SVR
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale

app=Flask(__name__)

C_course=['High School','Pre University','UG','PG'
]

#users={'prasad@gmail.com':'123'}

def fun(Name,Age,C_education,AreaofInterest,field,Skill,Grade):
	# Setting plot style
	plt.style.use('ggplot')
	info=[Name,Age,C_education,Grade,AreaofInterest,field,Skill]
	userinfo = pd.DataFrame([info], columns=['Name', 'Age','C_education','Grade','AreaofInterest','Field','Skills'])
	userinfo.to_csv('user_data.csv')
	print("userinfo",userinfo.shape)

	course_df = pd.read_csv("course_data.csv", encoding= 'unicode_escape')
	user_df = pd.read_csv("user_data.csv", encoding= 'unicode_escape')
	

	X = course_df.drop(columns='Age')
	y = course_df["Age"].values  
	X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=0.3, random_state=101)
	X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5, random_state=101)

	course_df = X_train.copy()
	course_df["Age"] = y_train

	#course_df


	baseline_y_pred = pd.DataFrame(course_df.groupby('C_education')['Age'].mean())
	val_course_dict = {'C_education': X_val["C_education"], 'Actual Age': y_val}  
	val_course_df = pd.DataFrame(val_course_dict)
	print("val_course_df",val_course_df)


	y_pred_and_y_true = pd.merge(baseline_y_pred, val_course_df, on='C_education')
	baseline_y_pred_vs_y_true = y_pred_and_y_true.rename(columns={"Age": "Predicted Age"})

	#baseline_y_pred_vs_y_true
	#Root Mean Square Error
	#print("RMSE baseline model: ", sqrt(mean_squared_error(baseline_y_pred_vs_y_true["Actual Age"], 
	#                                                       baseline_y_pred_vs_y_true["Predicted Age"])))
	content_train_df2  = course_df[course_df['Field'].str.contains(field,na=False,case=False,regex=True)]
	print("content_train_df2\\\\\\\\\\\\\\\\\\\\",content_train_df2)
	content_train_df = pd.merge(content_train_df2, user_df, on=['C_education','AreaofInterest','Skills'])
	print("content_train_df",content_train_df)
	print("content_train_df",content_train_df.shape)
        
	if content_train_df.shape[0]==0:
		df=pd.DataFrame()
		df1=pd.DataFrame()
		return df,df1
	else:
		content_train_df1 = pd.merge(course_df, user_df, on=['C_education','AreaofInterest','Skills'])
		#content_train_df
		y_grouped_by_user = content_train_df.groupby(["C_education"])
		y_train_listed = []
		for i, j in y_grouped_by_user:
			y_train_listed.append(j["Course"].values)  

		y_grouped_by_user1 = content_train_df1.groupby(["C_education"])
		y_train_listed1 = []
		for i, j in y_grouped_by_user1:
			y_train_listed1.append(j["Course"].values)    
		#y_train_listed[0]
		print("y_train_listed1",y_train_listed1[0])
		if y_train_listed[0].size==0:
			df=[]
			df2=[]
		else:
			df = pd.DataFrame(y_train_listed[0],columns=['Recommended Courses For You'])
			df1 = pd.DataFrame(y_train_listed1[0],columns=['Other courses available for you'])
		

		s = pd.Series([df.shape[0],df1.shape[0]])
		fig, ax = plt.subplots()
		labels = ["Recommended Course", "Other Course"]
		s.plot.pie(autopct="%.1f%%",labels=labels)
		fig.suptitle('Course Recommendation based on dataset')
		#s.plot.title("Course Recommendation based on dataset")
		
		#data=[df.shape[0],df1.shape[0]]
		#labels = ["Recommended", "Other"]
		#fig=plt.pie(x=data, autopct="%.1f%%",labels=labels, pctdistance=0.5)
		#plt.title("coursee", fontsize=14);
		#fig.savefig('static/images/my_plot.png')
		
		fig.savefig('static/images/my_plot.png')
		#print (df)
		#print (df1)
		return df,df1

@app.route('/')
def login():
    return render_template('education.html')
	
#@app.route('/education' ,methods=['GET','POST'])
#def abc():
#   return render_template('education.html')
    

@app.route('/eduresult',methods=['GET','POST'])
def result():
    if request.method=='POST':
        Name=request.form['name']
        Skill="1"
        Grade=request.form['grade']
        Age=request.form['age']
        C_education=request.form['education']
        print("C_education",C_education)
        if C_education=="1":
           AreaofInterest=request.form['sslc']
        if C_education=="2":
           AreaofInterest=request.form['puc']
        if C_education=="3":
           AreaofInterest=request.form['ug']
        if C_education=="4":
           AreaofInterest="5"
        field=request.form['field']
        #if C_education=="1":
        #   AreaofInterest=request.form['sslc']		   
        print ("Name,Age,C_education,AreaofInterest,field,Skill,Grade",Name,Age,C_education,AreaofInterest,field,Skill,Grade)
        df2=pd.DataFrame()
        df3=pd.DataFrame()
        df2,df3=fun(Name,Age,C_education,AreaofInterest,field,Skill,Grade)
        print("df22222222",type(df2))
        if df2.empty:
            return render_template('result.html',df='Sorry!! No Results Found')
        else:
            return render_template('eduresult.html',df=Markup(df2.to_html(index=False)+ "\n\n"+df3.to_html(index=False)))
           # with open("eduresult.html", 'w') as _file:
            #    _file.write(df.head().to_html() + "\n\n" + df1.head().to_html())			

if __name__=="__main__":
    app.run()