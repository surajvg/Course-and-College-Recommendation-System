from flask import Flask,render_template,Markup,request,redirect,url_for,session
#data manipulation and analysis
import pandas as pd
import numpy as np
# data visualizations, 
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import time
from scipy.sparse import csr_matrix
#data splitting
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale
from flask_mysqldb import MySQL
import MySQLdb.cursors


app=Flask(__name__)
app.secret_key = '1a2b3c4d5e'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root123'
app.config['MYSQL_DB'] = 'education'

mysql = MySQL(app)

states=['Jammu and Kashmir','Himachal Pradesh','Punjab','Chandigarh','Uttrakhand','Haryana',
'Delhi','Rajasthan','Uttar Pradesh','Bihar','Sikkim','Arunachal Pradesh','Nagaland','Manipur',
'Mizoram','Tripura','Meghalaya','Assam','West Bengal','Jharkhand','Odisha','Chhatisgarh',
'Madhya Pradesh','Gujarat','Daman & Diu','Dadra & Nagar Haveli','Maharashtra','Andhra Pradesh',
'Karnataka','Goa','Lakshadweep','Kerala','Tamil Nadu','Puducherry','Andaman & Nicobar Islands','Telangana'
]

accr_stud_density=['Medium','Very Low','High','Low']
unaccr_stud_density=['Medium','Very Low','High','Low']
accr_infra=['Satisfactory','Very Good','Bad','Good','Excellent']
unaccr_infra=['Satisfactory','Very Good','Good','Excellent','Bad']

users={'suraj@gmail.com':'123'}

C_course=['High School','Pre University','UG','PG'
]

#users={'suraj@gmail.com':'123'}

def fun(Name,Age,C_education,AreaofInterest,field,Skill,Grade):
	# Setting plot style
	plt.style.use('ggplot')

    #Creating a User DataFrame
	info=[Name,Age,C_education,Grade,AreaofInterest,field,Skill]
	userinfo = pd.DataFrame([info], columns=['Name', 'Age','C_education','Grade','AreaofInterest','Field','Skills'])
	userinfo.to_csv('user_data.csv')
	print("userinfo",userinfo.shape)

    #Reading Course Data:
	course_df = pd.read_csv("course_data.csv", encoding= 'unicode_escape')
	user_df = pd.read_csv("user_data.csv", encoding= 'unicode_escape')
	
    #Data Splitting:
	X = course_df.drop(columns='Age')
	y = course_df["Age"].values  
	X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=0.3, random_state=101)
	X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5, random_state=101)

	course_df = X_train.copy()
	course_df["Age"] = y_train

    #Data Merging
	baseline_y_pred = pd.DataFrame(course_df.groupby('C_education')['Age'].mean())
	val_course_dict = {'C_education': X_val["C_education"], 'Actual Age': y_val}  
	val_course_df = pd.DataFrame(val_course_dict)
	print("val_course_df",val_course_df)

    #Data Processing
	y_pred_and_y_true = pd.merge(baseline_y_pred, val_course_df, on='C_education')
	baseline_y_pred_vs_y_true = y_pred_and_y_true.rename(columns={"Age": "Predicted Age"})

	
	content_train_df2  = course_df[course_df['Field'].str.contains(field,na=False,case=False,regex=True)]
	print("content_train_df2\\\\\\\\\\\\\\\\\\\\",content_train_df2)
	content_train_df = pd.merge(content_train_df2, user_df, on=['C_education','AreaofInterest','Skills'])
	print("content_train_df",content_train_df)
	print("content_train_df",content_train_df.shape)
        
    #Initial Check:
	if content_train_df.shape[0]==0:
		df=pd.DataFrame()
		df1=pd.DataFrame()
		return df,df1
	else:
        #Data Processing (Non-Empty DataFrame)
		content_train_df1 = pd.merge(course_df, user_df, on=['C_education','AreaofInterest','Skills'])
		
        #The code groups the data in content_train_df by the 'C_education' column.
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
            #recommendation
			df = pd.DataFrame(y_train_listed[0],columns=['Recommended Courses For You'])
			df1 = pd.DataFrame(y_train_listed1[0],columns=['Other courses available for you'])
		

		s = pd.Series([df.shape[0],df1.shape[0]])
		fig, ax = plt.subplots()
		labels = ["Recommended Course", "Other Course"]
		s.plot.pie(autopct="%.1f%%",labels=labels)
		fig.suptitle('Course Recommendation based on dataset')
		
		
		fig.savefig('static/images/my_plot.png')
		#print (df)
		#print (df1)
		return df,df1

def accr_stud_density_not_req(df,accr,state_code,df_field,infra):
    if infra=='Not Required':
        #infra_code=accr_infra.index(infra)
        df=df[(df.state_code==state_code)&(df[df_field]==1)]
        df2=df.sort_values(['Total_Students','Accr_Score_Percentage','infrastructure_score','Total_Teachers'],ascending=False,inplace=False)
        df2=df2[['college_institution_id','name','city','autonomous','offers_scholarship','offers_loan','Accr_Score_Percentage','Total_Students']]
    else:
        infra_code=accr_infra.index(infra)
        df=df[(df.state_code==state_code)&(df.Infra_quality==infra_code)&(df[df_field]==1)]
        df2=df.sort_values(['Accr_Score_Percentage','Total_Students','infrastructure_score','Total_Teachers'],ascending=False,inplace=False)
        df2=df2[['college_institution_id','name','city','autonomous','offers_scholarship','offers_loan','Accr_Score_Percentage','Total_Students']]
    return df2

def accr_stud_density_req(df,accr,state_code,df_field,density,infra):
    if infra=='Not Required':
        density_code=accr_stud_density.index(density)
        #infra_code=accr_infra.index(infra)
        df=df[(df.state_code==state_code)&(df.Student_Density==density_code)&(df[df_field]==1)]
        df2=df.sort_values(['Accr_Score_Percentage','infrastructure_score','Total_Students','Total_Teachers'],ascending=False,inplace=False)
        df2=df2[['college_institution_id','name','city','autonomous','offers_scholarship','offers_loan','Accr_Score_Percentage','Total_Students']]
    else:
        density_code=accr_stud_density.index(density)
        infra_code=accr_infra.index(infra)
        df=df[(df.state_code==state_code)&(df.Infra_quality==infra_code)&(df.Student_Density==density_code)&(df[df_field]==1)]
        df2=df.sort_values(['Accr_Score_Percentage','infrastructure_score','Total_Students','Total_Teachers'],ascending=False,inplace=False)
        df2=df2[['college_institution_id','name','city','autonomous','offers_scholarship','offers_loan','Accr_Score_Percentage','Total_Students']]
    return df2

def unaccr_stud_density_not_req(df,state_code,df_field,infra):
    if infra=='Not Required':
        #infra_code=accr_infra.index(infra)
        df=df[(df.state_code==state_code)&(df[df_field]==1)]
        df2=df.sort_values(['infrastructure_score','Total_Students','Total_Teachers'],ascending=False,inplace=False)
        df2=df2[['college_institution_id','name','city','autonomous','offers_scholarship','offers_loan','Total_Students']]
    else:
        infra_code=unaccr_infra.index(infra)
        df=df[(df.state_code==state_code)&(df.Infra_quality==infra_code)&(df[df_field]==1)]
        df2=df.sort_values(['infrastructure_score','Total_Students','Total_Teachers'],ascending=False,inplace=False)
        df2=df2[['college_institution_id','name','city','autonomous','offers_scholarship','offers_loan','Total_Students']]
    return df2

def unaccr_stud_density_req(df,state_code,df_field,density,infra):
    if infra=='Not Required':
        density_code=unaccr_stud_density.index(density)
        #infra_code=accr_infra.index(infra)
        df=df[(df.state_code==state_code)&(df.Student_Density==density_code)&(df[df_field]==1)]
        df2=df.sort_values(['infrastructure_score','Total_Students','Total_Teachers'],ascending=False,inplace=False)
        df2=df2[['college_institution_id','name','city','autonomous','offers_scholarship','offers_loan','Total_Students']]
    else:
        density_code=unaccr_stud_density.index(density)
        infra_code=unaccr_infra.index(infra)
        df=df[(df.state_code==state_code)&(df.Infra_quality==infra_code)&(df.Student_Density==density_code)&(df[df_field]==1)]
        df2=df.sort_values(['infrastructure_score','Total_Students','Total_Teachers'],ascending=False,inplace=False)
        df2=df2[['college_institution_id','name','city','autonomous','offers_scholarship','offers_loan','Total_Students']]
    return df2

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))

def index():
   # Redirect to login page
    return render_template('index.html')

def education():
   # Redirect to login page
    return render_template('education.html')

@app.route('/home1', methods=['GET', 'POST'])
def home1():
   # Redirect to login page
    return render_template('home.html')

@app.route('/about', methods=['GET', 'POST'])
def about():
   # Redirect to login page
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
# Output message if something goes wrong...
    msg = ''
    print("login")
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'email' in request.form:
        # Create variables for easy access
        global name
        email = request.form['email']
        password = request.form['pass']

        # Check if account exists using MySQL
        # Fetch one record and return result
        cursor=mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * from users where email=%s and password=%s',(email, password,))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists!'

            session['loggedin'] = True
            session['username'] = email
            return render_template('home.html')
        else:
            msg = 'Account not exists!'
    return render_template('login.html', msg=msg)
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'name' in request.form and 'phone' in request.form:
        # Create variables for easy access
        name = request.form['name']
        phone = request.form['phone']
        email = request.form['email']
        password = request.form['pass']
        school = request.form['school']
        dob = request.form['dob']

                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s', (name,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        # elif not re.match(r'[A-Za-z0-9]+', name):
        #     msg = 'Username must contain only characters and numbers!'
        elif not name or not phone:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO users VALUES (NULL, %s, %s, %s, %s, %s, %s)', (name, phone, email, password, school, dob ))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)
#@app.route('/education' ,methods=['GET','POST'])
#def abc():
#   return render_template('education.html')
    
@app.route('/home',methods=['GET','POST'])
def home():
    if request.method=='POST':
        email=request.form['email']
        password=request.form['pass']
        if email in users and password==users[email]:
            return render_template('index.html')
        else:
            return redirect(url_for('login'))

#college recommendation
#Route Definition:
@app.route('/result',methods=['GET','POST'])
def result():
    #Request Method Check
    if request.method=='POST':
        #Form Data Retrieval:
        accr=(request.form['accredtion'])
        state=request.form['state']
        state_code=states.index(state)+1
        field=request.form['field']
        split_field=field.split(' ')
        df_field='c_'+'_'.join(split_field)
        density=request.form['density']
        infra=request.form['infra']

        if accr=='Yes':
            df=pd.read_csv('accr_clg_custer.csv')
            df2=pd.DataFrame()
            if density=='Not Required':
                #density_code=accr_stud_density.index(density)
                df2=accr_stud_density_not_req(df,accr,state_code,df_field,infra)
            else:
                df2=accr_stud_density_req(df,accr,state_code,df_field,density,infra)
            if df2.empty:
                return render_template('result.html',df='Sorry!! No Results Found')
            else:
                return render_template('result.html',df=Markup(df2.to_html(index=False)))
        
        else:
            df=pd.read_csv('unaccr_clg_custer.csv')
            df2=pd.DataFrame()
            if density=='Not Required':
                #density_code=accr_stud_density.index(density)
                df2=unaccr_stud_density_not_req(df,state_code,df_field,infra)
                    
            else:
                df2=unaccr_stud_density_req(df,state_code,df_field,density,infra)
            if df2.empty:
                return render_template('result.html',df='Sorry!! No Results Found')
            else:
                return render_template('result.html',df=Markup(df2.to_html(index=False)))
    return render_template('index.html')

#course recommendation
#Route Definition:
@app.route('/eduresult',methods=['GET','POST'])
def result1():
    #Request Method Check
    if request.method=='POST':
        #Form Data Retrieval:
        Name=request.form['name']
        Skill="1"
        Grade=request.form['grade']
        Age=request.form['age']
        C_education=request.form['education']
        print("C_education",C_education)
        #Conditional Assignment for 'AreaofInterest
        if C_education=="1":
           AreaofInterest=request.form['sslc']
        if C_education=="2":
           AreaofInterest=request.form['puc']
        if C_education=="3":
           AreaofInterest=request.form['ug']
        if C_education=="4":
           AreaofInterest="5"
        field=request.form['field']
       	   
        print ("Name,Age,C_education,AreaofInterest,field,Skill,Grade",Name,Age,C_education,AreaofInterest,field,Skill,Grade)

        #Function Call
        df2=pd.DataFrame()
        df3=pd.DataFrame()
        df2,df3=fun(Name,Age,C_education,AreaofInterest,field,Skill,Grade)
        print("df22222222",type(df2))

        #Result Handling and Rendering
        if df2.empty:
            return render_template('result.html',df='Sorry!! No Results Found')
        else:
            return render_template('eduresult.html',df=Markup(df2.to_html(index=False)+ "\n\n"+df3.to_html(index=False)))
           
    return render_template('education.html')

if __name__=="__main__":
    app.run()