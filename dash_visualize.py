
import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
import os

import numpy as np
import pandas as pd
from sklearn import datasets

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVC
from sklearn.svm import LinearSVR

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.decomposition import PCA     ## principal component analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import seaborn as sns

from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LassoCV

import base64 
import time

st.write("Machine Learning, Is concerned with using algorithms that automatically improve through iteration to produce a model that can predict outcomes of new data. \n The following sections present a visual representation of the machine learning process from enhancing data to applying various ML algorithms to the data and using these models to make predictions.")
st.write("Please select one of these sections, each provides an interactive explanation to one stage of the ML process.")
sections=st.selectbox("#choose sections",['Data Wrangling','ML Algorithms','Application in Stock Prices'])

if sections=='ML Algorithms':
    
    st.write("This is an interactive program that allows you to select datasets and parameters in a given model and view processed data interactively with the ability to change the model parameters in real-time.")

    vised=st.selectbox("choose ",['supervised','unsupervised'],1)

    #*******************************supervised*************************#

    if vised=="supervised":
        method=st.selectbox("choose the methode",['Classification','Regression'])

        if vised=='supervised' and method=='Classification':
            dataset_name=st.sidebar.selectbox("Dataset",("iris","Breast Cancer",'wine'))
            classifier_name=st.sidebar.selectbox("Classifier",("KNN","SVM"," Decision_Tree"))

            #this function for which the user picked 
            def get_dataset_c(dataset_name):

                if dataset_name == "iris":
                    data=datasets.load_iris()
                elif dataset_name == "Breast Cancer":
                    data=datasets.load_breast_cancer()
                else:
                    data=datasets.load_wine()

                #split data into two input and target
                X=data.data
                y=data.target
                return X,y
        

            def data_frame(dataset_name):
                '''Imports and returns dataset frame'''
                
                if dataset_name == "iris":
                    iris = datasets.load_iris()
                    data = pd.DataFrame(iris.data, columns = iris.feature_names)
                    table = data
                    
                elif dataset_name == "Breast Cancer":
                    cancer = datasets.load_breast_cancer()
                    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
                    df['target'] = cancer.target
                    table = df
                elif dataset_name == "wine":
                    wine = datasets.load_wine()
                    wine_df = pd.DataFrame(wine.data, columns = wine.feature_names)
                    table = wine_df
                return table

            X,y=get_dataset_c(dataset_name)
            table = data_frame(dataset_name)
            st.write("__Summarize Of Data__ ",pd.DataFrame(table).head())
            st.write(" __The Shape Of Data__ ",X.shape)
            st.write(" __Descriptive Statistics for Each Column Of Data__ ",table.describe())
            
            
            #add the parameter for the classifier algos like K and C 
            #**----  add parameters by the same way ----**

            def add_parameters_c(clf_name):
                
                #dictionary of parameters i will need in the classifer algorhithm
                params=dict()

                if clf_name == "KNN":
                    K=st.sidebar.slider("K",1,10)
                    w=st.sidebar.selectbox("weights",['distance','uniform'])
                    params["K"]=K
                    params["w"]=w
                elif clf_name =="SVM":
                    C=st.sidebar.slider("C",0.01,10.0)
                    params["C"]=C
                #random forest 
                else:
                    max_depth=st.sidebar.slider("max_depth",2,15)
                    random_state=st.sidebar.slider("random_state",1,100)
                    params["max_depth"]=max_depth
                    params["random_state"]=random_state
                return params

            #save the params as a varible 
            params=add_parameters_c(classifier_name)

            #get the classifier 

            def get_classifier(clf_name,params):
                if clf_name == "KNN":
                    clf = KNeighborsClassifier(n_neighbors=params["K"],weights=params["w"])
                elif clf_name =="SVM":
                    clf=SVC(C=params["C"])

                #random forest 
                else:
                    clf = tree.DecisionTreeClassifier()

                return clf
                    
            #save classifier name as a varible to use it after
            clf= get_classifier(classifier_name,params)


            #classification 

            #split the data into testing and training
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
            #training
            clf.fit(X_train,y_train)
            #predection
            y_pred=clf.predict(X_test)
            st.write(pd.DataFrame({
                'Values': y_test,
                'Predictions': y_pred,
                'Difference': abs(y_pred - y_test) 
                }))

            #accuaracy
            acc=accuracy_score(y_test,y_pred)


            st.write(f"__Mean absolute percentage error : {acc}__")

            pca = PCA(2)
            X_projected = pca.fit_transform(X)

            x1 = X_projected[:, 0]
            x2 = X_projected[:, 1]

            if dataset_name == 'iris':
                    
                pca = PCA(n_components=2)
                X_r = pca.fit(X).transform(X)

                lda = LinearDiscriminantAnalysis(n_components=2)
                X_r2 = lda.fit(X, y).transform(X)
                
                iris = datasets.load_iris()
                target_names=iris.target_names

                

                fig = plt.figure()
                colors = ['gold', 'teal', 'indigo']
                lw = 2

                for color, i, target_name in zip(colors, [0, 1, 2], target_names):
                    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8,
                                label=target_name)
                plt.legend(loc='best', shadow=False, scatterpoints=1)
                plt.title('PCA of IRIS dataset')
                fig2 = plt.figure()
                for color, i, target_name in zip(colors, [0, 1, 2], target_names):
                    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                                label=target_name)
                plt.legend(loc='best', shadow=False, scatterpoints=1)
                plt.title('LDA of IRIS dataset')

                st.pyplot(fig)
                st.pyplot(fig2)
                
                df_heatmap = pd.DataFrame(pca.components_, columns = table.columns)
                heatmap = plt.figure(figsize=(15, 8))
                sns.heatmap(df_heatmap, cmap='Blues')
                plt.title('Principal Components correlation with the features')
                plt.xlabel('Features')
                plt.ylabel('Principal Components')
                st.pyplot(heatmap)

                    
            elif dataset_name == 'Breast Cancer':
                
                data = pd.read_csv('C:\\Users\\20114\\data.csv')
                standardized = StandardScaler()
                standardized.fit(table)
                X = standardized.transform(table)
                
                pca = PCA(n_components=3)
                pca.fit(X)
                x_pca = pca.transform(X)
                
                def diag(x):
                    if x =='M':
                        return 1
                    else:
                        return 0
                df_diag= data['diagnosis'].apply(diag)
                
                fig = plt.figure(figsize=(15, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x_pca[:,0], x_pca[:,1], x_pca[:,2], c=df_diag,s=60)
                ax.legend(['Malign'])
                ax.set_xlabel('First Principal Component')
                ax.set_ylabel('Second Principal Component')
                ax.set_zlabel('Third Principal Component')
                ax.view_init(30, 120)
                st.pyplot(fig)
                
                ax = plt.figure(figsize=(12,8))
                sns.scatterplot(x_pca[:,0], x_pca[:,2],hue=data['diagnosis'], palette ="viridis"  )
                plt.xlabel('First Principal Component')
                plt.ylabel('Third Principal Component')
                st.pyplot(ax)
                
                ax = plt.figure(figsize=(12,8))
                sns.scatterplot(x_pca[:,1], x_pca[:,2],hue=data['diagnosis'], palette ="viridis" )
                plt.xlabel('Second Principal Component')
                plt.ylabel('Third Principal Component')
                st.pyplot(ax)
                
                ax = plt.figure(figsize=(12,8))
                sns.scatterplot(x_pca[:,0], x_pca[:,1],hue=data['diagnosis'], palette ="viridis" )
                plt.xlabel('First Principal Component')
                plt.ylabel('Second Principal Component')
                st.pyplot(ax)
                
                
                df_heatmap = pd.DataFrame(pca.components_, columns = table.columns)
                heatmap = plt.figure(figsize=(15, 8))
                sns.heatmap(df_heatmap, cmap='Blues')
                plt.title('Principal Components correlation with the features')
                plt.xlabel('Features')
                plt.ylabel('Principal Components')
                st.pyplot(heatmap)
                    
            else:
                
                k = list(y).copy()
                standardized = StandardScaler()
                standardized.fit(table)
                X = standardized.transform(table)
                
                pca = PCA(n_components=3)
                x_pca = pca.fit_transform(X)
            
                fig = plt.figure(figsize=(15, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x_pca[:,0], x_pca[:,1], x_pca[:,2], c=k,s=60)
                ax.set_xlabel('First Principal Component')
                ax.set_ylabel('Second Principal Component')
                ax.set_zlabel('Third Principal Component')
                ax.view_init(30, 120)
                st.pyplot(fig)
                
                ax1 = plt.figure(figsize=(12,8))
                sns.scatterplot(x_pca[:,0], x_pca[:,2],hue=k, palette ="viridis" )
                plt.xlabel('First Principal Component')
                plt.ylabel('Third Principal Component')
                st.pyplot(ax1)
                
                ax2 = plt.figure(figsize=(12,8))
                sns.scatterplot(x_pca[:,1], x_pca[:,2],hue = k, palette ="viridis" )
                plt.xlabel('Second Principal Component')
                plt.ylabel('Third Principal Component')
                st.pyplot(ax2)
                
                ax3 = plt.figure(figsize=(12,8))
                sns.scatterplot(x_pca[:,0], x_pca[:,1],hue = k, palette ="viridis" )
                plt.xlabel('First Principal Component')
                plt.ylabel('Second Principal Component')
                st.pyplot(ax3)
                
                
                df_heatmap = pd.DataFrame(pca.components_, columns = table.columns)
                heatmap = plt.figure(figsize=(15, 8))
                sns.heatmap(df_heatmap, cmap='Blues')
                plt.title('Principal Components correlation with the features')
                plt.xlabel('Features')
                plt.ylabel('Principal Components')
                st.pyplot(heatmap)

        if vised=='supervised' and method=='Regression':

            dataset_name=st.sidebar.selectbox("Dataset",['california_housing','boston_house','diabetes'])
            regressor_name=st.sidebar.selectbox("Regressor",("LinearRegressor","SVM",'KNeighborsRegressor'," Decision_Tree"))

            #this function for which the user picked 
            def get_dataset_r(dataset_name):
                '''Imports datasets, returns X data and y target'''
                if dataset_name == "california_housing":
                    data=datasets.fetch_california_housing()
                elif dataset_name == "boston_house":
                    data=datasets.load_boston()
                elif dataset_name == "diabetes":
                    data=datasets.load_diabetes()                  
                else:
                    st.sidebar.write("still working on it")
                    data=datasets.load_diabetes()
                
                #split data into two input and target
                X=data.data
                y=data.target
                
                return X,y

            def get_data(dataset_name):
                '''Imports and returns dataset frame'''
                
                if dataset_name == "california_housing":
                    data=datasets.fetch_california_housing(as_frame=True)
                    table = data.frame
                elif dataset_name == "boston_house":
                    boston=datasets.load_boston()
                    data= pd.DataFrame(boston.data,columns=boston.feature_names)
                    data['target'] = pd.Series(boston.target)
                    table = data
                elif dataset_name == "diabetes":
                    data=datasets.load_diabetes(as_frame=True)
                    table = data.frame
                return table
            
        
            def get_dataParameters(dataset_name):
                '''returns dataset frame'''
                if dataset_name == "california_housing":
                    data=datasets.fetch_california_housing()
                    
                elif dataset_name == "boston_house":
                    data=datasets.load_boston()
                    
                elif dataset_name == "diabetes":
                    data=datasets.load_diabetes()
                    
                parameter =data.feature_names
                return parameter
            
            #the shape of data 
            X,y=get_dataset_r(dataset_name)
            table = get_data(dataset_name)
            st.write("__Summary Of Data__ ",pd.DataFrame(table).head())
            st.write(" __The Shape Of Data__ ",X.shape)
            st.write(" __Descriptive Statistics for Each Column Of Data__ ",table.describe())
            #for i, v in enumerate(table.columns):
            #    st.write(i, v)
            #the classes or label 
            #add the parameter for the classifier algos like K and C 
            #**----  add parameters by the same way ----**
            
            def add_parameters_r(reg_name):
                """Lets user select the features, Outputs parameters dictionary """
                #dictionary of parameters i will need in the regression algorhithm
                params=dict()

                if reg_name == "KNeighborsRegressor":
                    n_neighbors=st.sidebar.slider("n_neighbors",1,15,5)
                    params["n_neighbors"]=n_neighbors
                
                elif reg_name == "LinearRegressor":
                    features = get_dataParameters(dataset_name)
                    selected=[]
                    st.sidebar.write("Select Features")
                    for x in features:
                        if st.sidebar.checkbox(x, key=x) == True:
                            selected.append(x)
                    params["LinearRegressionFeatures"] = selected

                elif reg_name =="SVM":
                    C=st.sidebar.slider("C",1.0,10.0)
                    params["C"]=C
                    
                #random forest 
                else:
                    max_depth=st.sidebar.slider("max_depth",2,15)
                    max_features=st.sidebar.selectbox("max_features",['auto','sqrt','log2'])
                    params["max_depth"]=max_depth
                    params["max_features"]=max_features
                
                return params

            #save the params as a varible 
            params=add_parameters_r(regressor_name)
            
            #get the regressor 

            def get_regressor(reg_name,params):
                if reg_name == "KNeighborsRegressor":
                    reg = KNeighborsRegressor(n_neighbors=params["n_neighbors"])          
                elif reg_name == "LinearRegressor":
                    reg= LinearRegression(fit_intercept=False)   
                elif reg_name =="SVM":
                    reg=LinearSVR(C=params["C"])
                #random forest 
                else:
                    reg = tree.DecisionTreeRegressor(max_depth=params["max_depth"],max_features=params["max_features"])            
                return reg
                    
            #save classifier name as a varible to use it after        
            reg=get_regressor(regressor_name,params)

            #classification 

            #split the data into testing and training
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=13)
            #training
            
            reg.fit(X_train,y_train)
            #predection
            y_pred=reg.predict(X_test)
            
            st.write(pd.DataFrame({
                'Values': y_test,
                'Predictions': y_pred,
                'Difference': abs(y_pred - y_test) 
                }))
            
            #accuaracy
            acc=mean_absolute_percentage_error(y_test,y_pred)


            st.write(f"__Mean absolute percentage error :{acc}__")

            #plot
            if dataset_name == "california_housing":
                
                st.write(" __1- House Value__ ")
                ### v1
                fig = plt.figure(figsize=(10,6))
                sns.distplot(table['MedHouseVal'],color = 'blue')
                st.pyplot(fig)
                
                ## v2
                st.write(" __2- Population Vs House Value__ ")
                fig2 = plt.figure(figsize=(10,6))

                plt.scatter(table['Population'],table['MedHouseVal'],c=table['MedHouseVal'],s=table['MedInc']*50)
                plt.colorbar()
                plt.title('population vs house value' )
                plt.xlabel('population')
                plt.ylabel('house value')
                st.pyplot(fig2)
                
                
                ## v3
                st.write('__3- House Price on basis of Geo-coordinates__')
                fig3 = plt.figure(figsize=(15,10))
                plt.scatter(table['Longitude'],table['Latitude'],c=table['MedHouseVal'],s=table['Population']/10,cmap='viridis')
                plt.colorbar()
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title('house price on basis of geo-coordinates')
                st.pyplot(fig3)
                
                ## v4
                rng = np.random.RandomState(0)
                n_samples = 500
                cov = [[3, 3],
                    [3, 4]]
                X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
                pca = PCA(n_components=2).fit(X)
                
                fig4 = plt.figure()
                plt.scatter(X[:, 0], X[:, 1], alpha=.3, label='samples')
                for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
                    comp = comp * var  # scale component by its variance explanation power
                    plt.plot([0, comp[0]], [0, comp[1]], label=f"Component {i}", linewidth=5,
                            color=f"C{i + 2}")
                plt.gca().set(aspect='equal',
                            title="2-dimensional dataset with principal components",
                            xlabel='first feature', ylabel='second feature')
                plt.legend()
                st.pyplot(fig4)
                
                
                y = X.dot(pca.components_[1]) + rng.normal(size=n_samples) / 2
                fig, axes = plt.subplots(1, 2, figsize=(10, 3))

                axes[0].scatter(X.dot(pca.components_[0]), y, alpha=.3)
                axes[0].set(xlabel='Projected data onto first PCA component', ylabel='y')
                axes[1].scatter(X.dot(pca.components_[1]), y, alpha=.3)
                axes[1].set(xlabel='Projected data onto second PCA component', ylabel='y')
                plt.tight_layout()
                st.pyplot(fig)
                
                ###v5
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)
                pcr = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())
                pcr.fit(X_train, y_train)
                pca = pcr.named_steps['pca']
                
                pls = PLSRegression(n_components=1)
                pls.fit(X_train, y_train)

                fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                axes[0].scatter(pca.transform(X_test), y_test, alpha=.3, label='ground truth')
                axes[0].scatter(pca.transform(X_test), pcr.predict(X_test), alpha=.3,
                                label='predictions')
                axes[0].set(xlabel='Projected data onto first PCA component',
                            ylabel='y', title='PCR / PCA')
                axes[0].legend()
                axes[1].scatter(pls.transform(X_test), y_test, alpha=.3, label='ground truth')
                axes[1].scatter(pls.transform(X_test), pls.predict(X_test), alpha=.3,
                                label='predictions')
                axes[1].set(xlabel='Projected data onto first PLS component',
                            ylabel='y', title='PLS')
                axes[1].legend()
                plt.tight_layout()
                st.pyplot(fig)
            elif dataset_name == 'boston_house':
                
                #v1
                table['MEDV'] = y #target column
                fig = plt.figure()
                sns.set(rc={'figure.figsize':(11.7,8.27)})
                sns.distplot(table['MEDV'], bins=30)
                st.pyplot(fig)
                
                
                #v2
                fig = plt.figure(figsize=(20, 5))

                features = ['LSTAT', 'RM']
                target = table['MEDV']

                for i, col in enumerate(features):
                    plt.subplot(1, len(features) , i+1)
                    x = table[col]
                    y = target
                    plt.scatter(x, y, marker='o')
                    plt.title(col)
                    plt.xlabel(col)
                    plt.ylabel('MEDV')
                st.pyplot(fig)
                
            else:
                
                ##test
                predicted = cross_val_predict(reg, X, y, cv=5)

                fig = fig, ax = plt.subplots()
                ax.scatter(y, predicted, edgecolors=(0, 0, 0))
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
                ax.set_xlabel('Measured')
                ax.set_ylabel('Predicted')
                st.pyplot(fig)
                
                #v2
                diabetes = datasets.load_diabetes() 
                lasso = LassoCV().fit(X, y)
                importance = np.abs(lasso.coef_)
                feature_names = np.array(diabetes.feature_names)
                fig = plt.figure()
                plt.bar(height=importance, x=feature_names)
                plt.title("Feature importances via coefficients")
                st.pyplot(fig)

    #*******************************unsupervised*************************#

    if vised=="unsupervised":
        st.info("we still working on it ! is will be availbal soon")

elif sections=='Application in Stock Prices':
    st.write('Using the techniques from the first two stages we will show a model that predicts stock prices based on historic data')
    fields=st.selectbox("Fieds",['finance','working on other'])
    if fields=="finance":
        START = st.sidebar.date_input("Start Date ")
        end=st.sidebar.date_input("end Date ")
        TODAY = date.today().strftime("%Y-%m-%d")

        st.title('Stock Forecast App')

        selected_stock = st.sidebar.selectbox('Select the company triker symbol',
            ['AAPL','MSFT','AMZN','GOOG','FB','TSLA','BABA','NVDA','PYPL','INTC','AMD'])

        n_years = st.slider('Years of prediction:', 1, 4)
        period = n_years * 365


        @st.cache
        def load_data(ticker):
            data = yf.download(ticker, START='2010-10-5', end=TODAY)
            data.reset_index(inplace=True)
            return data

            
        data_load_state = st.text('Loading data...')
        data = load_data(selected_stock)
        data_load_state.text('Loading data... done!')

        st.subheader('Raw data')
        st.write(data.tail())

        #close and volume

        # Open	High	Low	Close	Volume	Dividends	Stock Splits

        st.write("""## Closing Price""")
        st.line_chart(data.Close)

        st.write("""## Volume Price""")
        st.line_chart(data.Volume)



        # Predict forecast with Prophet.
        df_train = data[['Date','Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Show and plot forecast
        st.subheader('Forecast data')
        st.write(forecast.tail())
        st.write("Forecast components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)
           
elif sections=='Data Wrangling':
    
    st.write("Data wrangling is one of the steps taken to enhance data and prepare it for processing, In this stage you can import data and it'll be prepared to be used with ML algorithms")
    timestr = time.strftime("%Y%m%d-%H:%M:%S")
    
    
    st.title("Upload Your CSV File")
    
    
    class FileDownloader(object):
        """docstring for FileDownloader
        >>> download = FileDownloader(data,filename,file_ext).download()
        """
        def __init__(self, data,filename='edit_file',file_ext='txt'):
            super(FileDownloader, self).__init__()
            self.data = data
            self.filename = filename
            self.file_ext = file_ext

        def download(self):
            b64 = base64.b64encode(self.data.encode()).decode()
            new_filename = "{}_{}_.{}".format(self.filename,timestr,self.file_ext)
            st.markdown("#### Download File ###")
            href = f'<a href="data:file/txt;base64,{b64}" download="{new_filename}">Click Here!!</a>'
            st.markdown(href,unsafe_allow_html=True)



    data_file = ''


    data_file = st.file_uploader("Upload CSV",type=['csv'])

    if data_file is not None:
        st.title('Explore the Dataset')
        df = pd.read_csv(data_file) # encoding= 'unicode_escape'
        if st.checkbox('Show raw Data'):
            st.dataframe(df)
            df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
            
            st.markdown('### Analysing column distribution')
            all_columns_names = df.columns.tolist()
            selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
            if st.button("Generate Plot"):
                st.success("Generating Customizable Bar Plot for {}".format(selected_columns_names))
                cust_data = df[selected_columns_names]
                st.bar_chart(cust_data)

            if st.checkbox("Show Shape"):
                st.write(df.shape)
            if st.checkbox("Show Columns"):
                for i, v in enumerate(df.columns):
                    st.write(i, v)
            if st.checkbox("Summary"):
                st.write(df.describe())
            if st.checkbox("Show Selected Columns"):
                all_columns = df.columns.to_list()
                selected_columns = st.multiselect("Select Columns",all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)
            if st.checkbox('Unique Value'):
                st.write('the number of unique values in each column', df.nunique())
            if st.checkbox("Null Value"):
                st.write("__number of null value__",df.isnull().sum())
                if st.checkbox("Remove Null Value"):
                    df.fillna(df.mean(), inplace=True)
                    st.write('__chek after fill null value by mean__', df.isnull().sum())
            if st.checkbox("Duplicates Row"):
                st.write('__Check For Duplicates Row In The Data__',sum(df.duplicated()))
            if sum(df.duplicated()) != 0:
                if st.checkbox("Drop Duplicates Row"):
                    df.drop_duplicates(inplace=True)
                    st.write('__Check After Drop Duplicates __',sum(df.duplicated()))
            if st.button('Save your edit'):
                download = FileDownloader(df.to_csv(index = False),data_file.name,file_ext='csv').download()


