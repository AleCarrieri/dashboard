import dash
import dash_bootstrap_components as dbc   # pip install dash-bootstrap-components
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np

from datetime import date
import plotly.express as px
#______________________________________________________________________________________
#### Import 2017 data
df_2017 = pd.read_csv('IST_Central_2017.csv')
#delete unused features
df_2017=df_2017.drop(columns=['Date_start','DayOfWeek','DayName','Month','MonthName','Official_Holiday','Holiday_NoActivity','Holiday'])
df_2017['Date'] = pd.to_datetime(df_2017['Date'])

#### Import 2018 data
df_2018 = pd.read_csv('IST_Central_2018.csv')
#delete unused features
df_2018=df_2018.drop(columns=['Date_start','DayOfWeek','DayName','Month','MonthName','Official_Holiday','Holiday_NoActivity','Holiday'])
df_2018['Date'] = pd.to_datetime(df_2018['Date'])

#Import data for forecast models
df3=pd.read_csv('IST_Central_Data20172018.csv')
df3['Date'] = pd.to_datetime(df3['Date_start'])
df3['year'] = df3['Date'].dt.year
available_years = df3['year'].unique()

#______________________________________________________________________________________
##Feature selection
options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))
from dash_extensions import Lottie         # pip install dash-extensions

#______________________________________________________________________________________
#Regression
from sklearn.model_selection import train_test_split
from sklearn import  metrics

data_1718 = pd.read_csv('IST_Central_DataModel.csv')
data_1718['Date_start'] =  pd.to_datetime(data_1718['Date_start'],format='%Y-%m-%d %H:%M:%S')
data_1718 = data_1718.set_index ('Date_start')


##Split training and test data
#sklearn methods do not work with dataframes, it is necessary to create arrays
X=data_1718.values
Y=X[:,0]
X=X[:,[1,2,3,4,5,6]]

##Split Data into training and test data

#by default, it chooses randomly 75% of the data for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X,Y)

#1) Linear regression
from sklearn import  linear_model

# Create linear regression object
modelLR_regr = linear_model.LinearRegression()
# Train the model using the training sets
modelLR_regr.fit(X_train,y_train)
# Make predictions using the testing set
y_pred_LR = modelLR_regr.predict(X_test)
#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR)
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
#____________________________________________________________________________________________________________________
#2) Support Vector Regressor

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_SVR = sc_X.fit_transform(X_train)
y_train_SVR = sc_y.fit_transform(y_train.reshape(-1,1))


modelSVR_regr = SVR(kernel='rbf')
modelSVR_regr.fit(X_train_SVR,y_train_SVR.ravel())

y_pred_SVR = modelSVR_regr.predict(sc_X.fit_transform(X_test))
y_test_SVR=sc_y.fit_transform(y_test.reshape(-1,1))
y_pred_SVR2=sc_y.inverse_transform(y_pred_SVR)


MAE_SVR=metrics.mean_absolute_error(y_test,y_pred_SVR2)
MSE_SVR=metrics.mean_squared_error(y_test,y_pred_SVR2)
RMSE_SVR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_SVR2))
cvRMSE_SVR=RMSE_SVR/np.mean(y_test)


#____________________________________________________________________________________________________________________
#3) Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

# Create Regression Decision Tree object
DT_regr_model = DecisionTreeRegressor()

# Train the model using the training sets
DT_regr_model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_DT = DT_regr_model.predict(X_test)

#Evaluate errors
MAE_DT=metrics.mean_absolute_error(y_test,y_pred_DT)
MSE_DT=metrics.mean_squared_error(y_test,y_pred_DT)
RMSE_DT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y_test)


#____________________________________________________________________________________________________________________
#4) Random forest
#Uniformized data
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Random forest uniformized data
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 100,
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 10,
              'max_leaf_nodes': None}

RF_model = RandomForestRegressor(**parameters)
RF_model.fit(X_train_scaled, y_train.reshape(-1,1).ravel())
y_pred_RF = RF_model.predict(X_test_scaled)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF)
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)



#____________________________________________________________________________________________________________________
#5) Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

GB_model = GradientBoostingRegressor()
GB_model.fit(X_train, y_train)
y_pred_GB =GB_model.predict(X_test)

#Evaluate errors
MAE_GB=metrics.mean_absolute_error(y_test,y_pred_GB)
MSE_GB=metrics.mean_squared_error(y_test,y_pred_GB)
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y_test)

#____________________________________________________________________________________________________________________
#6) Gradient Boosting
# from xgboost import XGBRegressor
# params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
#          'learning_rate': 0.01, 'loss': 'ls'}
# GB_model = GradientBoostingRegressor(**params)
#
# XGB_model = XGBRegressor()
# XGB_model.fit(X_train, y_train)
# y_pred_XGB =XGB_model.predict(X_test)
#
# #Evaluate errors
# MAE_XGB=metrics.mean_absolute_error(y_test,y_pred_XGB)
# MSE_XGB=metrics.mean_squared_error(y_test,y_pred_XGB)
# RMSE_XGB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_XGB))
# cvRMSE_XGB=RMSE_XGB/np.mean(y_test)

#____________________________________________________________________________________________________________________
#7) Bootstrapping

from sklearn.ensemble import BaggingRegressor

BT_model = BaggingRegressor()
BT_model.fit(X_train, y_train)
y_pred_BT =BT_model.predict(X_test)


#Evaluate errors
MAE_BT=metrics.mean_absolute_error(y_test,y_pred_BT)
MSE_BT=metrics.mean_squared_error(y_test,y_pred_BT)
RMSE_BT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y_test)

#____________________________________________________________________________________________________________________
#8) Neural Network

#from sklearn.neural_network import MLPRegressor

#NN_model = MLPRegressor(hidden_layer_sizes=(10,10,10,10))
#NN_model.fit(X_train,y_train)
#y_pred_NN = NN_model.predict(X_test)


#Evaluate errors
#MAE_NN=metrics.mean_absolute_error(y_test,y_pred_NN)
#MSE_NN=metrics.mean_squared_error(y_test,y_pred_NN)
#RMSE_NN= np.sqrt(metrics.mean_squared_error(y_test,y_pred_NN))
#cvRMSE_NN=RMSE_NN/np.mean(y_test)
#_____________________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________________
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.YETI])
server = app.server

# the style arguments for the sidebar. I use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.Div(html.Img(src=app.get_asset_url('IST.png'), style={'height':'100%', 'width':'100%'})),
        html.Hr(),
        html.H3("Alameda Campus"),
        html.H3(["Central"], style={'textAlign': 'center','font-style': 'italic','font-weight': 'bold'}),
        html.Hr(),
        html.P(
            "Energy Services", className="lead", style={'textAlign': 'center','font-weight': 'bold'}
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Exploratory Data Analysis ", href="/page-1", active="exact"),
                dbc.NavLink("Typical load profiles", href="/page-2", active="exact"),
                dbc.NavLink("Feature Selection", href="/page-3", active="exact"),
                dbc.NavLink("Energy Forecasting", href="/page-4", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Hr(),
        html.P("Alessandro Carrieri  100843", className="lead", style={'textAlign': 'center'}),
        html.Hr()
    ],
    style=SIDEBAR_STYLE,
)
content = html.Div(id="page-content", style=CONTENT_STYLE)


app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


card_Hour = dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="90%", height="200px", url="assets/Hour.json")),
                dbc.CardBody([
                    html.H4('Hour'),
                ], style={'textAlign':'center'})
            ]),
        ], width='2')
card_DayOfWeek = dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="100%", height="200px", url="assets/DayOfWeek.json")),
                dbc.CardBody([
                    html.H4('DayOfWeek'),
                ], style={'textAlign':'center'})
            ]),
        ], width='2')
card_Month = dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="90%", height="200px", url="assets/Month.json")),
                dbc.CardBody([
                    html.H4('Month'),
                ], style={'textAlign':'center'})
            ]),
        ], width='2')
card_Holiday = dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="100%", height="200px", url="assets/Holiday.json")),
                dbc.CardBody([
                    html.H4('Holiday'),
                ], style={'textAlign':'center'})
            ]),
        ], width='2')
card_Power = dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="80%", height="200px", url="assets/Power-1.json")),
                dbc.CardBody([
                    html.H4('Power-1'),
                ], style={'textAlign':'center'})
            ]),
        ], width='2')
card_Temperature = dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="50%", height="200px", url="assets/Temperature.json")),
                dbc.CardBody([
                    html.H4('Temperature'),
                ], style={'textAlign':'center'})
            ]),
        ], width='2')
card_HR = dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="60%", height="200px", url="assets/HR.json")),
                dbc.CardBody([
                    html.H4('HR'),
                ], style={'textAlign':'center'})
            ]),
        ], width='2')
card_WindSpeed = dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="90%", height="200px", url="assets/WindSpeed.json")),
                dbc.CardBody([
                    html.H4('WindSpeed'),
                ], style={'textAlign':'center'})
            ]),
        ], width=2)
card_WindGust = dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="90%", height="200px", url="assets/WindGust.json")),
                dbc.CardBody([
                    html.H4('WindGust'),
                ], style={'textAlign':'center'})
            ]),
        ], width=2)
card_P = dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="100%", height="200px", url="assets/P.json")),
                dbc.CardBody([
                    html.H4('P'),
                ], style={'textAlign':'center'})
            ]),
        ], width=2)
card_SolarRad = dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="80%", height="200px", url="assets/SolarRad.json")),
                dbc.CardBody([
                    html.H4('Solar Radiation'),
                ], style={'textAlign':'center'})
            ]),
        ], width=2)
card_rain = dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="80%", height="200px", url="assets/61287-raining-clouds.json")),
                dbc.CardBody([
                    html.H4('rain_mm/h'),
                ], style={'textAlign':'center'})
            ]),
        ], width=2)
card_rain_day = dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="100%", height="200px", url="assets/rain_day.json")),
                dbc.CardBody([
                    html.H4('rain_day'),
                ], style={'textAlign':'center'})
            ]),
        ], width=2)

table_header = [html.Thead(html.Tr([html.Th("Feature selected")]))]
row1 = html.Tr([html.Td("Hour")])
row2 = html.Tr([html.Td("DayOfWeek")])
row3 = html.Tr([html.Td("Month")])
row4 = html.Tr([html.Td("Holiday")])
row5 = html.Tr([html.Td("Power-1")])
row6 = html.Tr([html.Td("Temperature")])
row7 = html.Tr([html.Td("HR")])
row8 = html.Tr([html.Td("WindSpeed")])
row9 = html.Tr([html.Td("WindGust")])
row10 = html.Tr([html.Td("P")])
row11 = html.Tr([html.Td("SolarRad")])
row12 = html.Tr([html.Td("rain mm/h")])
row13 = html.Tr([html.Td("rain_day")])

table_body = [html.Tbody([row1, row2, row3, row4,row5, row6, row7,row8, row9, row10, row11, row12, row13])]
table_kBest = [html.Tbody([row5, row1, row11, row2])]
table_Wrapper = [html.Tbody([row4])]
table_Ensemble = [html.Tbody([row5, row1, row11, row6])]

#callback for Sidebar-define what's in Home page/Exploratory Data Analysis/Typical load profiles/Energy Forecasting
@app.callback(
    dash.dependencies.Output('page-content', 'children'),
    [dash.dependencies.Input('url', 'pathname')])
def render_page_content(pathname):
    if pathname == "/":
        return html.Div([
            html.Div([dbc.Jumbotron(
                [
                    html.H1(["Building Central"],style={'color': 'white','textAlign': 'center'},className="display-3"),
                    html.P(["Discover the Building"],style={'color': 'white','textAlign': 'center'},className="lead"),
                    html.Hr(className="my-2"),
                ]
            ,style={'background-image': 'url(assets/fundos-zoom-01.jpg)'})]
        ),
            dcc.Tabs(
                id="tabs-with-classes",
                value='tab-1',
                parent_className='custom-tabs',
                className='custom-tabs-container',
                children=[
                    dcc.Tab(
                        label='Localization',
                        value='tab-1',
                        className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        label='Enter in the Building',
                        value='tab-2',
                        className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                ]),
            html.Div(id='tabs-content-classes')
   ])
    elif pathname == "/page-1":
        return html.Div([
    html.Div([dbc.Jumbotron(
    [
        html.H1(["Visual representation of data"], style={'color': 'white', 'textAlign': 'center'}, className="display-3"),
        html.P(["Extract knowledge from energy data"], style={'color': 'white', 'textAlign': 'center'},
               className="lead"),
        html.Hr(className="my-2"),
        html.P(["Select a year and discover Electricity Consumption in the building."], style={'color': 'white', 'textAlign': 'center'},
               className="lead"),
        html.P(["Yellow points are referred to Holidays data ."], style={'color': 'white', 'textAlign': 'center'},
               className="lead"),
    ]
,style={'background-image': 'url(assets/fundos-zoom-01.jpg)'})]),
    html.Div([
        dcc.Dropdown(
            id='menu',
            options=[{'label': i, 'value': i} for i in available_years],
            value=2017
        ),
    ]),
    html.Div([
        dcc.Graph(id='yearly-data'),
    ]),
    html.Div([
        html.A(id="BoxPlotbyDay"),
        html.A(id="BoxPlotbyDay&Hour"),
        html.A(id="Carpet")
    ])

        ]),
    elif pathname == "/page-2":
          return html.Div([
              dbc.Jumbotron(
                [
                    html.H1(["Cluster analysis"], style={'color': 'white', 'textAlign': 'center'},
                            className="display-3"),
                    html.P(["Extract knowledge from energy data"], style={'color': 'white', 'textAlign': 'center'},
                           className="lead"),
                    html.Hr(className="my-2"),
                    html.P(["Pattern recognition in Energy consumption time series through Unsupervised learning."],
                           style={'color': 'white', 'textAlign': 'center'},
                           className="lead"),
                ]
                , style={'background-image': 'url(assets/fundos-zoom-01.jpg)'}),

              html.Div([
                  html.H2(["2017"], style={'color': 'black', 'textAlign': 'center'},
                          className="display-2"),
                  dcc.DatePickerRange(
                      id='my-date-picker-range1',
                      min_date_allowed=date(2017, 1, 1),
                      max_date_allowed=date(2017, 12, 31),
                      display_format='DD-MMM-YYYY',
                      clearable=True,
                      with_portal=False,
                      first_day_of_week=1
                  ),
                  dcc.Graph(id='daily_graph2017'),

                  html.Img(src=app.get_asset_url('Clusters2017.png')),
                  html.Hr(),
                  html.Hr(),
              ]),

              html.Div([
                  html.H2(["2018"], style={'color': 'black', 'textAlign': 'center'},
                          className="display-2"),
                  dcc.DatePickerRange(
                      id='my-date-picker-range2',
                      min_date_allowed=date(2018, 1, 1),
                      max_date_allowed=date(2018, 12, 31),
                      display_format='DD-MMM-YYYY',
                      clearable=True,
                      with_portal=False,
                      first_day_of_week=1
                  ),
                  dcc.Graph(id='daily_graph2018'),

                  html.Img(src=app.get_asset_url('Clusters2018.png'))
              ])

]
          ),
    elif pathname == "/page-3":
        return html.Div([
    dbc.Jumbotron(
        [
            html.H1(["Feature selection"], style={'color': 'white', 'textAlign': 'center'},
                    className="display-3"),
            html.P(["Relevant variables used to develop forecasting models"], style={'color': 'white', 'textAlign': 'center'},
                   className="lead"),
            html.Hr(className="my-2"),
            html.P(["View the results of the methods used to select features."],
                   style={'color': 'white', 'textAlign': 'center'},
                   className="lead"),
        ]
        , style={'background-image': 'url(assets/fundos-zoom-01.jpg)'}),
    html.Div([
        dbc.CardGroup(
            [card_Hour, card_DayOfWeek, card_Month, card_Holiday,card_Power, card_Temperature]),
        html.Br(),
        dbc.CardGroup(
            [card_HR, card_WindSpeed,card_WindGust, card_P, card_SolarRad, card_rain]),
        html.Br(),
        dbc.CardGroup([card_rain_day])
    ]),
html.Br(),


    dcc.Tabs(
        id="tabs-with-classes-3",
        value='tab-1',
        parent_className='custom-tabs',
        className='custom-tabs-container',
        children=[
            dcc.Tab(
                label='Filter method (kBest) ',
                value='tab-1',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                label='Wrapper method (RFE)',
                value='tab-2',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                label='Ensemble method',
                value='tab-3', className='custom-tab',
                selected_className='custom-tab--selected'
            )
        ]),

    html.Div(id='tabs-content-classes-3')
]),
    elif pathname == "/page-4":
        return html.Div([
            html.Div([dbc.Jumbotron(
                [
                    html.H1(["Regression"], style={'color': 'white', 'textAlign': 'center'},
                            className="display-3"),
                    html.P(["Extract knowledge from energy data"],
                           style={'color': 'white', 'textAlign': 'center'}, className="lead"),
                    html.Hr(className="my-2"),
                    html.P(["Compare results from Supervised machine learning methods."],
                           style={'color': 'white', 'textAlign': 'center'},
                           className="lead"),
                ]
            ,style={'background-image': 'url(assets/fundos-zoom-01.jpg)'})]),
            dcc.Tabs(
                id="tabs-with-classes-4",
                value='tab-1',
                parent_className='custom-tabs',
                className='custom-tabs-container',
                children=[
                    dcc.Tab(
                        label='Linear regression',
                        value='tab-1',
                        className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        label='Support Vector Regressor',
                        value='tab-2',
                        className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        label='Decision Tree Regressor',
                        value='tab-3', className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        label='Random forest',
                        value='tab-4', className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        label='Gradient Boosting',
                        value='tab-5', className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        label='XGBRegressor',
                        value='tab-6', className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        label='Bootstrapping',
                        value='tab-7', className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        label='Artificial Neural Network',
                        value='tab-8', className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                ]),
            html.Div(id='tabs-content-classes-4')
   ])

    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )



#callback for Home Page - Tabs
@app.callback(
    dash.dependencies.Output('tabs-content-classes', 'children'),
    [dash.dependencies.Input('tabs-with-classes', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            dbc.Row([
                dbc.Col(html.Iframe(
                    src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d665.7392181931064!2d-9.138397832612565!3d38.731916277830706!2m3!1f0!2f39.2763231660565!3f0!3m2!1i1024!2i768!4f35!3m3!1m2!1s0xd1933a2302b2157%3A0x11c13201e2cf8229!2sPavilh%C3%A3o%20Central%2C%20Av.%20Rovisco%20Pais%201%2C%201049-001%20Lisboa%2C%20Portogallo!5e1!3m2!1sit!2sit!4v1620238269696!5m2!1sit!2sit/index.html",
                    style={"height": "1060px", "width": "1000px"}), ),
                dbc.Col(html.Iframe(
                    src="https://3dwarehouse.sketchup.com/model/e8beccd1fe605d566054aafaa5dde650/Instituto-Superior-T%C3%A9cnico-Pavilh%C3%A3o-Central?hl=it.html",
                    style={"height": "850px", "width": "850px"})),
            ])

        ])
    elif tab == 'tab-2':
        return html.Div([
            dbc.Row([
                dbc.Col(html.Iframe(
                    src="http://in-learning.ist.utl.pt/embed/IST/01-Central/01.html",
                    style={"height": "1060px", "width": "1000px"}), ),
                dbc.Col(html.Iframe(
                    src="https://3dwarehouse.sketchup.com/model/e8beccd1fe605d566054aafaa5dde650/Instituto-Superior-T%C3%A9cnico-Pavilh%C3%A3o-Central?hl=it.html",
                    style={"height": "850px", "width": "850px"})),
            ])

        ])


#callback for updating line plot in Exploratory Data Analysis
@app.callback(
    dash.dependencies.Output('yearly-data', 'figure'),
    [dash.dependencies.Input('menu', 'value')])

def update_graph(value):
    dff = df3[df3['year'] == value]
    return create_graph(dff)
def create_graph(dff):
    return {
        'data': [
            {'x': dff.Date, 'y': dff.Power_kW, 'type': 'line', 'name': 'Power_kW'},

        ],
        'layout': {
            'title': 'IST Lisboa yearly  Consumption'
        }
    }

#html.Embed(src=app.get_asset_url("Grapg.html"), style={"height": "700px", "width": "800px"})
#callback for updating Boxplot by Day of Week in Exploratory Data Analysis
@app.callback(
    dash.dependencies.Output('BoxPlotbyDay', 'children'),
    [dash.dependencies.Input('menu', 'value')])
def updategraphs(value):
    if value==2017:
        fig1=html.Img(src=app.get_asset_url('BoxPlotbyDay2017.png'))
        return fig1
    elif value==2018:
        fig2 = html.Img(src=app.get_asset_url('BoxPlotbyDay2018.png'))
        return fig2


#callback for updating Boxplot by Hour (DayofWeek) in Exploratory Data Analysis
@app.callback(
    dash.dependencies.Output('BoxPlotbyDay&Hour', 'children'),
    [dash.dependencies.Input('menu', 'value')])
def updategraphs(value):
    if value==2017:
        fig3=html.Img(src=app.get_asset_url('BoxPlotbyDay&Hour2017.png'))
        return fig3
    elif value==2018:
        fig4 = html.Img(src=app.get_asset_url('BoxPlotbyDay&Hour2018.png'))
        return fig4

#callback for updating Carpet Plot in Exploratory Data Analysis
@app.callback(
    dash.dependencies.Output('Carpet', 'children'),
    [dash.dependencies.Input('menu', 'value')])
def updategraphs(value):
    if value==2017:
        fig5=html.Img(src=app.get_asset_url('Carpet2017.png'))
        return fig5
    elif value==2018:
        fig6 = html.Img(src=app.get_asset_url('Carpet2018.png'))
        return fig6


#callback for Cluster Analysis 2017
@app.callback(
    dash.dependencies.Output('daily_graph2017', 'figure'),
    [dash.dependencies.Input('my-date-picker-range1', 'start_date'),
     dash.dependencies.Input('my-date-picker-range1', 'end_date')]
)
def update_output(start_date, end_date):
    dff = df_2017.loc[(df_2017['Date'] > start_date) & (df_2017['Date'] < end_date)]
    fig = px.line(dff, x="Hour", y=["Power_kW"], line_group="Date", height=900, template="simple_white")
    return fig



#callback for Cluster Analysis 2018
@app.callback(
    dash.dependencies.Output('daily_graph2018', 'figure'),
    [dash.dependencies.Input('my-date-picker-range2', 'start_date'),
     dash.dependencies.Input('my-date-picker-range2', 'end_date')]
)
def update_output(start_date, end_date):
    dff = df_2018.loc[(df_2018['Date'] > start_date) & (df_2018['Date'] < end_date)]
    fig = px.line(dff, x="Hour", y="Power_kW", line_group="Date", height=900, template="simple_white")
    return fig




#callback for Feature selection - Tabs
@app.callback(
     dash.dependencies.Output('tabs-content-classes-3', 'children'),
    [dash.dependencies.Input('tabs-with-classes-3', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        table = dbc.Table(
            # using the same table as in the above example
            table_header + table_kBest,
            bordered=True,
            dark=False,
            hover=True,
            responsive=True,
            striped=True,
        )
        return table
    elif tab == 'tab-2':
        table = dbc.Table(
            # using the same table as in the above example
            table_header + table_Wrapper,
            bordered=True,
            dark=False,
            hover=True,
            responsive=True,
            striped=True,
        )
        return table
    elif tab == 'tab-3':
        table = dbc.Table(
            # using the same table as in the above example
            table_header + table_Ensemble,
            bordered=True,
            dark=False,
            hover=True,
            responsive=True,
            striped=True,
        )
        return table



#callback for Forecasting - Tabs
@app.callback(
     dash.dependencies.Output('tabs-content-classes-4', 'children'),
    [dash.dependencies.Input('tabs-with-classes-4', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        df = pd.DataFrame(
            {
                "MAE": [round(MAE_LR,2)],
                "MSE": [round(MSE_LR,2)],
                "RMSE": [round(RMSE_LR,2)],
                "cvRMSE": [round(cvRMSE_LR,3)]
            })
        fig = px.scatter(x=y_test, y=y_pred_LR, template="simple_white",labels={'x': 'Real [kW]', 'y': 'Predicted [kW]'})
        return html.Div([
            dcc.Graph(
                id='yearly-data',
                figure={
                    'data': [
                        {'y': y_test, 'type': 'line', 'name': 'Real'},
                        {'y': y_pred_LR, 'type': 'line', 'name': 'Predicted'},

                    ],
                    'layout': {
                        'title': 'Real vs Predicted with LR electrical load [kW] at IST Central'
                    }
                }
            ),
            dcc.Graph(figure=fig),
            dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size='md'),
        ])
    elif tab == 'tab-2':
        df = pd.DataFrame(
            {
                "MAE": [round(MAE_SVR, 2)],
                "MSE": [round(MSE_SVR, 2)],
                "RMSE": [round(RMSE_SVR, 2)],
                "cvRMSE": [round(cvRMSE_SVR, 3)]
            })
        fig = px.scatter(x=y_test, y=y_pred_SVR2, template="simple_white",
                         labels={'x': 'Real [kW]', 'y': 'Predicted [kW]'})
        return html.Div([
            dcc.Graph(
                id='yearly-data',
                figure={
                    'data': [
                        {'y': y_test, 'type': 'line', 'name': 'Real'},
                        {'y': y_pred_SVR2, 'type': 'line', 'name': 'Predicted'},

                    ],
                    'layout': {
                        'title': 'Real vs Predicted with SVR  electrical load [kW] at IST Central'
                    }
                }
            ),
            dcc.Graph(figure=fig),
            dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size='md'),
        ])
    elif tab == 'tab-3':
        df = pd.DataFrame(
            {
                "MAE": [round(MAE_DT, 2)],
                "MSE": [round(MSE_DT, 2)],
                "RMSE": [round(RMSE_DT, 2)],
                "cvRMSE": [round(cvRMSE_DT, 3)]
            })
        fig = px.scatter(x=y_test, y=y_pred_DT, template="simple_white",
                         labels={'x': 'Real [kW]', 'y': 'Predicted [kW]'})
        return html.Div([
            dcc.Graph(
                id='yearly-data',
                figure={
                    'data': [
                        {'y': y_test, 'type': 'line', 'name': 'Real'},
                        {'y': y_pred_DT, 'type': 'line', 'name': 'Predicted'},

                    ],
                    'layout': {
                        'title': 'Real vs Predicted with DT  electrical load [kW] at IST Central'
                    }
                }
            ),
            dcc.Graph(figure=fig),
            dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size='md'),
        ])
    elif tab == 'tab-4':
        df = pd.DataFrame(
            {
                "MAE": [round(MAE_RF, 2)],
                "MSE": [round(MSE_RF, 2)],
                "RMSE": [round(RMSE_RF, 2)],
                "cvRMSE": [round(cvRMSE_RF, 3)]
            })
        fig = px.scatter(x=y_test, y=y_pred_RF, template="simple_white",
                         labels={'x': 'Real [kW]', 'y': 'Predicted [kW]'})
        return html.Div([
            dcc.Graph(
                id='yearly-data',
                figure={
                    'data': [
                        {'y': y_test, 'type': 'line', 'name': 'Real'},
                        {'y': y_pred_RF, 'type': 'line', 'name': 'Predicted'},

                    ],
                    'layout': {
                        'title': 'Real vs Predicted with RF  electrical load [kW] at IST Central'
                    }
                }
            ),
            dcc.Graph(figure=fig),
            dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size='md'),
        ])
    elif tab == 'tab-5':
        df = pd.DataFrame(
            {
                "MAE": [round(MAE_GB, 2)],
                "MSE": [round(MSE_GB, 2)],
                "RMSE": [round(RMSE_GB, 2)],
                "cvRMSE": [round(cvRMSE_GB, 3)]
            })
        fig = px.scatter(x=y_test, y=y_pred_GB, template="simple_white",
                         labels={'x': 'Real [kW]', 'y': 'Predicted [kW]'})
        return html.Div([
            dcc.Graph(
                id='yearly-data',
                figure={
                    'data': [
                        {'y': y_test, 'type': 'line', 'name': 'Real'},
                        {'y': y_pred_GB, 'type': 'line', 'name': 'Predicted'},

                    ],
                    'layout': {
                        'title': 'Real vs Predicted with GB  electrical load [kW] at IST Central'
                    }
                }
            ),
            dcc.Graph(figure=fig),
            dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size='md'),
        ])
    elif tab == 'tab-6':
        df = pd.DataFrame(
            {
                "MAE": [round(MAE_XGB, 2)],
                "MSE": [round(MSE_XGB, 2)],
                "RMSE": [round(RMSE_XGB, 2)],
                "cvRMSE": [round(cvRMSE_XGB, 3)]
            })
        fig = px.scatter(x=y_test, y=y_pred_XGB, template="simple_white",
                         labels={'x': 'Real [kW]', 'y': 'Predicted [kW]'})
        return html.Div([
            dcc.Graph(
                id='yearly-data',
                figure={
                    'data': [
                        {'y': y_test, 'type': 'line', 'name': 'Real'},
                        {'y': y_pred_XGB, 'type': 'line', 'name': 'Predicted'},

                    ],
                    'layout': {
                        'title': 'Real vs Predicted with XGB  electrical load [kW] at IST Central'
                    }
                }
            ),
            dcc.Graph(figure=fig),
            dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size='md'),
        ])
    elif tab == 'tab-7':
        df = pd.DataFrame(
            {
                "MAE": [round(MAE_BT, 2)],
                "MSE": [round(MSE_BT, 2)],
                "RMSE": [round(RMSE_BT, 2)],
                "cvRMSE": [round(cvRMSE_BT, 3)]
            })
        fig = px.scatter(x=y_test, y=y_pred_BT, template="simple_white",
                         labels={'x': 'Real [kW]', 'y': 'Predicted [kW]'})
        return html.Div([
            dcc.Graph(
                id='yearly-data',
                figure={
                    'data': [
                        {'y': y_test, 'type': 'line', 'name': 'Real'},
                        {'y': y_pred_BT, 'type': 'line', 'name': 'Predicted'},

                    ],
                    'layout': {
                        'title': 'Real vs Predicted with BT  electrical load [kW] at IST Central'
                    }
                }
            ),
            dcc.Graph(figure=fig),
            dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size='md'),
        ])
    elif tab == 'tab-8':
        df = pd.DataFrame(
            {
                "MAE": [round(MAE_NN, 2)],
                "MSE": [round(MSE_NN, 2)],
                "RMSE": [round(RMSE_NN, 2)],
                "cvRMSE": [round(cvRMSE_NN, 3)]
            })
        fig = px.scatter(x=y_test, y=y_pred_NN, template="simple_white",
                         labels={'x': 'Real [kW]', 'y': 'Predicted [kW]'})
        return html.Div([
            dcc.Graph(
                id='yearly-data',
                figure={
                    'data': [
                        {'y': y_test, 'type': 'line', 'name': 'Real'},
                        {'y': y_pred_NN, 'type': 'line', 'name': 'Predicted'},

                    ],
                    'layout': {
                        'title': 'Real vs Predicted with NN  electrical load [kW] at IST Central'
                    }
                }
            ),
            dcc.Graph(figure=fig),
            dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size='md'),
        ])



if __name__ == '__main__':
    app.run_server(debug=False)

