import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from plotly import subplots
import plotly.express as px
from dash.dependencies import Input, Output, State
import cv2
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import seaborn as sns
from math import sqrt
import time
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale



# Setting plot style
plt.style.use('ggplot')
plt1.style.use('ggplot')
app = dash.Dash(__name__)

#course_data.describe()
def get_group_count(min_age, max_age, dataset):
	age_group = dataset.apply(lambda x: True if max_age > x['Age'] > min_age else False, axis=1)
	count = len(age_group[age_group == True].index)
	return count

def get_group():
	course_data = pd.read_csv("course_data.csv", encoding= 'unicode_escape')
	users_avg_course = pd.DataFrame(course_data.groupby('C_education')['Age'].mean())
	print(users_avg_course.head())


	users_avg_course = users_avg_course['Age']
	fig5, ax5 = plt1.subplots()
	n, bins, patches = ax5.hist(users_avg_course, label='avg age based course',
							stacked=True, color='royalblue', bins=50, rwidth=0.8)

	ax5.set_xlabel('Average course', size=12)
	ax5.set_ylabel('Count', size=12)
	ax5.set_title('The Distribution of Average course based on age group', size=12)


	plt1.axvline(course_data["Age"].mean(), color='chocolate', linestyle='dotted', dash_capstyle="round",
            linewidth=3, label="Mean")
	ax5.legend(bbox_to_anchor=(1, 0.92))

	plt1.show()
	
	#course_data = pd.read_csv("course_data.csv", encoding= 'unicode_escape')
	course_edu_gender_data = course_data[['C_education', 'Age']]
	sslc_course_ages = course_edu_gender_data.loc[course_edu_gender_data['C_education'] == 1].sort_values('Age')
	puc_course_ages = course_edu_gender_data.loc[course_edu_gender_data['C_education'] == 2].sort_values('Age')
	degree_course_ages = course_edu_gender_data.loc[course_edu_gender_data['C_education'] == 3].sort_values('Age')
	master_course_ages = course_edu_gender_data.loc[course_edu_gender_data['C_education'] == 4].sort_values('Age')

	G1_sslc = get_group_count(0, 18, sslc_course_ages)
	G2_sslc = get_group_count(17, 25, sslc_course_ages)
	G3_sslc = get_group_count(24, 35, sslc_course_ages)
	G4_sslc = get_group_count(34, 45, sslc_course_ages)
	G5_sslc = get_group_count(44, 50, sslc_course_ages)
	G6_sslc = get_group_count(49, 56, sslc_course_ages)
	G7_sslc = get_group_count(55, 200, sslc_course_ages)

	G1_puc = get_group_count(0, 18, puc_course_ages)
	G2_puc = get_group_count(17, 25, puc_course_ages)
	G3_puc = get_group_count(24, 35, puc_course_ages)
	G4_puc = get_group_count(34, 45, puc_course_ages)
	G5_puc = get_group_count(44, 50, puc_course_ages)
	G6_puc = get_group_count(49, 56, puc_course_ages)
	G7_puc = get_group_count(55, 200, puc_course_ages)

	G1_degree = get_group_count(0, 18, degree_course_ages)
	G2_degree = get_group_count(17, 25, degree_course_ages)
	G3_degree = get_group_count(24, 35, degree_course_ages)
	G4_degree = get_group_count(34, 45, degree_course_ages)
	G5_degree = get_group_count(44, 50, degree_course_ages)
	G6_degree = get_group_count(49, 56, degree_course_ages)
	G7_degree = get_group_count(55, 200, degree_course_ages)

	G1_master = get_group_count(0, 18, master_course_ages)
	G2_master = get_group_count(17, 25, master_course_ages)
	G3_master = get_group_count(24, 35, master_course_ages)
	G4_master = get_group_count(34, 45, master_course_ages)
	G5_master = get_group_count(44, 50, master_course_ages)
	G6_master = get_group_count(49, 56, master_course_ages)
	G7_master = get_group_count(55, 200, master_course_ages)


	labels = ['Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']
	sslc_grouped = [G1_sslc, G2_sslc, G3_sslc, G4_sslc, G5_sslc, G6_sslc, G7_sslc]
	puc_grouped = [G1_puc, G2_puc, G3_puc, G4_puc, G5_puc, G6_puc, G7_puc]
	degree_grouped = [G1_degree, G2_degree, G3_degree, G4_degree, G5_degree, G6_degree, G7_degree]
	master_grouped = [G1_master, G2_master, G3_master, G4_master, G5_master, G6_master, G7_master]

	x = np.arange(len(labels))  # the label locations
	width = 0.40  # the width of the bars

	fig1, ax1 = plt1.subplots()
	rects1 = ax1.bar(x - 0.2, sslc_grouped, width, label='sslc', color='royalblue')
	rects2 = ax1.bar(x , puc_grouped, width, label='puc', color='darkorange')
	rects3 = ax1.bar(x + 0.2, degree_grouped, width, label='degree', color='green')
	rects4 = ax1.bar(x + 0.4, master_grouped, width, label='master', color='red')

	# Setting the labels for the bar chart
	ax1.set_ylabel('Number of course', size=12)
	ax1.set_xlabel('Age', size=12)
	ax1.set_xticks(x)
	ax1.set_xticklabels(labels)
	ax1.set_title('Fig. 1: Users grouped by education and age', size=15)
	ax1.legend()
	plt.show()

	course_data["C_education"].hist()
	plt.show()
	data=pd.DataFrame(course_data.groupby('C_education')['C_education'].count())
	data.plot.pie(y='C_education', figsize=(8, 6))
	plt.show()

app.layout = html.Div([
	html.Div([
		html.Div([
			html.Img(
				src="/assets/logo.jpg",
				style={"height" : "40px", "width" : "40px", "border-radius":"20px"}
			)
		],style={"float":"left","padding" : "5px 0 5px 50px"}),
		html.Div(
			children="Classification and Quality Analysis of Rice",
			style={"float":"left","padding" : "10px 0 10px 10px","font-size": "17px", "font-weight" :"600"}
		),
		html.Div([
			html.Div([html.A("Home",href="#home")], style={"float":"left","padding":"0 10px 0 10px","align-items": "center","font-size": "15px", "font-weight" :"600"}),
			html.Div([html.A("About Project",href="#about-project")], style={"float":"left","padding":"0 10px 0 10px","align-items": "center","font-size": "15px", "font-weight" :"600"}),
			#html.Div([html.A("About Us",href="#about-us")], style={"float":"left","padding":"0 10px 0 10px","align-items": "center","font-size": "15px", "font-weight" :"600"}),
			
		],style={"float":"right", "padding": "10px 50px 10px 0px"})
	],className="nav"),
	html.Div([],style={"height":"50px"},id="home"),
	html.Div([
		html.H1(children="Visualisation of Results", style={"text-align":"center", "margin":"0", "padding-bottom" : "20px", "color" : "whitesmoke"}),
		html.Div([
			html.Div([
				dcc.Graph(get_group(),id="graph1"),
				html.P("Original Image", style={"margin":"0","padding-bottom":"10px"})
			], style = {"display": "block", "justify-content": "center", "align-items": "center", "padding":"0 20px 0 20px"}),
			html.Div([
				dcc.Graph(get_group1(),id="graph1"),
				html.P("Binary Image", style={"margin":"0","padding-bottom":"10px"})
			], style = {"display": "block", "justify-content": "center", "align-items": "center", "padding":"0 20px 0 20px"}),
		], style = {"display": "flex", "justify-content": "center", "align-items": "center", "text-align":"center"})
	],style = {"color":"black", "padding" : "20px 0 20px 0", "color" : "whitesmoke"},id='plots'),
	html.Div([
		html.H1(children="About Project", style={"text-align":"center"}),
		html.P(children=text1),
		html.P(children=text2),
		html.P(children=text3),
		html.P(children=text4),
		html.P(children=text5),
		html.P(children=text6),
	],style = {"color":"white", "padding":"10px 50px 10px 50px"},id="about-project"),
	html.Div([
		html.Div([
			html.Div([html.A("About Us",href="#")], style={"padding":"0 10px 0 10px","align-items": "center","font-size": "15px", "font-weight" :"600"}),
		],style={"padding": "10px 50px 10px 0px"})
	],className="foo",id="bottom")
])

#def parse_contents(contents, filename):
	#print(contents)

if __name__ == '__main__':
	app.run_server(debug=False)