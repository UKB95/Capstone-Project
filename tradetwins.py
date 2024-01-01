##Importing libraries


import streamlit as st

import seaborn as sns
import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
from datetime import date
from dateutil.relativedelta import relativedelta
from streamlit_option_menu import option_menu

from sklearn.preprocessing import StandardScaler
from sklearn import metrics


#Import Model Packages 
from sklearn.cluster import KMeans, AgglomerativeClustering,AffinityPropagation, DBSCAN
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import cluster, covariance, manifold

import warnings
warnings.filterwarnings("ignore")

import requests
from streamlit_lottie import st_lottie



st.set_page_config(page_title = 'TradeTwins', page_icon = ":tada:", layout = "wide")

################################################################################################################################
#=======================================Some auxiliary functions================================================================

def find_cointegrated_pairs(data):
    n = data.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            result = coint(data[keys[i]], data[keys[j]])
            pvalue_matrix[i, j] = result[1]
            if result[1] < 0.05:
                pairs.append((keys[i], keys[j]))
    return pvalue_matrix, pairs   


def find_correlated_pairs(train_data):
    correlated_pairs = ()
    for i in range(len(train_data.columns)):
        for j in range(i+1,len(train_data.columns)):
            if (train_data[train_data.columns[i]].pct_change().corr(train_data[train_data.columns[j]].pct_change()))>0.5:
                   correlated_pairs = ((train_data.columns[i],train_data.columns[j]),)+correlated_pairs

    return train_data.pct_change().corr(method ='pearson'), correlated_pairs


def find_cluster_pairs(train_data):
    returns = train_data.pct_change().mean() * 252
    returns = pd.DataFrame(returns)
    returns.columns = ['Returns']
    returns['Volatility'] = train_data.pct_change().std() * np.sqrt(252)
    data=returns


    scaler = StandardScaler().fit(data)
    rescaledDataset = pd.DataFrame(scaler.fit_transform(data),columns = data.columns, index = data.index)
    X=rescaledDataset

    silhouette_score = []
    max_loop = 9
    for k in range(2, max_loop):
        kmeans = KMeans(n_clusters=k,  random_state=10, n_init=10)
        kmeans.fit(X)        
        silhouette_score.append(metrics.silhouette_score(X, kmeans.labels_, random_state=10))
    
    nclust = silhouette_score.index(max(silhouette_score))+2

    k_means = cluster.KMeans(n_clusters = nclust)
    k_means.fit(X)

    kmeans = metrics.silhouette_score(X, k_means.labels_, metric='euclidean')

    #Affinity propogation
    ap = AffinityPropagation()
    ap.fit(X)

    no_clusters = len(ap.cluster_centers_indices_)

    affinity = metrics.silhouette_score(X, ap.labels_, metric='euclidean')

    if kmeans>affinity:
        clustered_series_all = pd.Series(index=X.index, data=k_means.labels_.flatten())
        clustered_series = clustered_series_all[clustered_series_all != -1]
    else:
        clustered_series_all = pd.Series(index=X.index, data=ap.labels_.flatten())
        clustered_series = clustered_series_all[clustered_series_all != -1]


    df = pd.DataFrame()
    df['Cluster_no'] = pd.DataFrame(clustered_series)


    groups = df.groupby('Cluster_no')

    # Print groups
    for key, group in groups:
        print(f"Key: {key}, Group: {list(group)}")


    counts=clustered_series.value_counts()

    # let's visualize some clusters
    cluster_vis_list = list((counts).index)[::-1]
    

    pairs = ()
    for k in range(0,len(cluster_vis_list)):   
        globals()[f'cluster_pairs{k}']=()
        df = pd.DataFrame(groups.get_group(k))
        df.reset_index(inplace = True)
        for i in range(0,len(df)):
            for j in range(i+1,len(df)):
                globals()[f'cluster_pairs{k}'] = ((df['index'].loc[i],df['index'].iloc[j]),)+globals()[f'cluster_pairs{k}']
        pairs = pairs+ globals()[f'cluster_pairs{k}']

    print(len(pairs))
    return pairs, 1


def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()



def zscore(series):
    return (series - series.mean()) / np.std(series)

def rename_columns(col):
    if col.endswith('.NS'):
        return col[:-3]
    else:
        return col

#################################################################################################################################
#==========================================To extract data important for output================================================

def result_calculation(train_data, test_data,initial_investment, method):
    pairs_list = []
    investment_list = []
    final_protfolio_list = []
    profit_list = []
    roi_list = []



    initial_capital = initial_investment
    if method == 'Cointegration':
        pvalues, pairs = find_cointegrated_pairs(train_data)
    elif method == 'Correlation':
        pvalues, pairs = find_correlated_pairs(train_data)
    else:
        pairs, pvalues = find_cluster_pairs(train_data)

    for i in range(len(pairs)):
        asset1 = pairs[i][0]
        asset2 = pairs[i][1]
        
        # create a train dataframe of 2 assets
        while(True):
            train = pd.DataFrame()
            train['asset1'] = train_data[asset1]
            train['asset2'] = train_data[asset2]

            # Fitting OLS model
            train = pd.DataFrame()
            train['asset1'] = train_data[asset1]
            train['asset2'] = train_data[asset2]

            model=sm.OLS(train.asset2, train.asset1).fit()

            #Checking if parameters need changing
            if model.params[0]>1:
                #print('Assests should be interchanged')
                asset1 = pairs[i][1]
                asset2 = pairs[i][0]
                continue
            else:
                break
                
                
        residual = train.asset2 - model.params[0] * train.asset1 # residual of the OLS model
        

        
        
        # conduct Augmented Dickey-Fuller test
        adf = adfuller(residual, maxlag = 1)


        
        ######################Signal and profit loss generation######################
        
            # create a dataframe for trading signals
        signals = pd.DataFrame()
        signals['asset1'] = test_data[asset1] 
        signals['asset2'] = test_data[asset2]
        ratios = signals.asset1 / signals.asset2

        # calculate z-score and define upper and lower thresholds
        signals['z'] = zscore(ratios)
        signals['z upper limit'] = np.mean(signals['z']) + np.std(signals['z'])
        signals['z lower limit'] = np.mean(signals['z']) - np.std(signals['z'])

        # create signal - short if z-score is greater than upper limit else long
        signals['signals1'] = 0
        signals['signals1'] = np.select([signals['z'] > \
                                         signals['z upper limit'], signals['z'] < signals['z lower limit']], [-1, 1], default=0)


        # we take the first order difference to obtain portfolio position for stock
        signals['positions1'] = signals['signals1'].diff()
        signals['signals2'] = -signals['signals1']
        signals['positions2'] = signals['signals2'].diff()




        signals = signals.reset_index()

        
        
        # shares to buy for each position
        positions1 = initial_capital// max(signals['asset1'])
        positions2 = initial_capital// max(signals['asset2'])
        # and in the end we aggregate them into one portfolio
        portfolio = pd.DataFrame()
        portfolio['asset1'] = signals['asset1']
        portfolio['holdings1'] = signals['positions1'].cumsum() * signals['asset1'] * positions1
        portfolio['cash1'] = initial_capital - (signals['positions1'] * signals['asset1'] * positions1).cumsum()
        portfolio['total asset1'] = portfolio['holdings1'] + portfolio['cash1']
        portfolio['return1'] = portfolio['total asset1'].pct_change()
        portfolio['positions1'] = signals['positions1']

        #portfolio.head().append(portfolio.tail())

        #portfolio[portfolio.positions1 != 0].head()

        # pnl for the 2nd asset
        portfolio['asset2'] = signals['asset2']
        portfolio['holdings2'] = signals['positions2'].cumsum() * signals['asset2'] * positions2
        portfolio['cash2'] = initial_capital - (signals['positions2'] * signals['asset2'] * positions2).cumsum()
        portfolio['total asset2'] = portfolio['holdings2'] + portfolio['cash2']
        portfolio['return2'] = portfolio['total asset2'].pct_change()
        portfolio['positions2'] = signals['positions2']



        ############Looping portion############
        #portfolio.to_csv('portfolio.csv')


        # total pnl and z-score
        portfolio['z'] = signals['z']
        portfolio['total asset'] = portfolio['total asset1'] + portfolio['total asset2']
        portfolio['z upper limit'] = signals['z upper limit']
        portfolio['z lower limit'] = signals['z lower limit']
        portfolio = portfolio.dropna()

        # calculate final portfolio value
        final_portfolio = portfolio['total asset'].iloc[-1]
        #final_portfolio

        profit = final_portfolio - 2*initial_capital
        #profit

        roi = (profit / (2*initial_capital))*100 # Overall ROI of the pair trading strategy
        #roi
        
        ###Updating list
        pairs_list.append(pairs[i])
        investment_list.append(initial_capital)
        final_protfolio_list.append(final_portfolio)
        profit_list.append(profit)
        roi_list.append(roi)

    full_output ={'Stock Pairs':pairs_list,
                  'Initial Investment':investment_list,
                  'Final Value' : final_protfolio_list,
                  'Profit':profit_list,
                  'Return(%)': roi_list}
    df = pd.DataFrame.from_dict(full_output)
    df.sort_values(by = 'Return(%)', axis=0, ascending=False, inplace = True)
    df.reset_index(drop = True, inplace = True)
    df.index = df.index+1
    df_styled = df.style.applymap(lambda x: 'background-color: %s'% 'red' if x < 0 else 'background-color: %s'% 'green'  , subset =[ 'Return(%)'])

    df.to_csv('final_output_cointergrated_auto.csv')
    return df, df_styled, pairs, signals


def pvalue_plot(pvalue, train_data,method):
    
    if (method == 'Cointegration'):
        st.header("The p-value heatmap is given below")
        fig, ax = plt.subplots(figsize=(15,10))
        sns.heatmap(pvalue, xticklabels = train_data.columns,
                    yticklabels = train_data.columns, cmap = 'RdYlGn_r', annot = True, fmt=".2f",
                    mask = (pvalue >= 0.99))
                        
        ax.set_title('Assets Cointegration Matrix p-values Between Pairs')
        plt.savefig('pvalue.png')  
    else:
        st.header("Correlation heatmap")
        fig, ax = plt.subplots(figsize=(15,10))
        sns.heatmap(train_data.pct_change().corr(method ='pearson'), ax=ax, cmap = 'crest', annot=True, fmt=".2f") 
        ax.set_title('Correlation Matrix')
        plt.savefig("pvalue.png")

def data(test_end_date,test_start_date, train_end_date, train_start_date, ticker_list):

    
    # Downloading train data
    train_data= yf.download(ticker_list ,train_start_date,train_end_date, auto_adjust=True)['Close']
    train_data = train_data.rename(columns=rename_columns)
    # Downloading test data
    test_data = yf.download(ticker_list ,test_start_date,test_end_date, auto_adjust=True)['Close']
    test_data = test_data.rename(columns=rename_columns)

    while(max(train_data.isnull().sum())):
        if max(train_data.isnull().sum()) < 100:
            train_data.dropna(inplace = True)
        else:
            test_data.drop(train_data.isnull().sum().idxmax(), axis =1, inplace = True)
            train_data.drop(train_data.isnull().sum().idxmax(), axis =1, inplace = True)

    return train_data, test_data

def if_function(test_end_date,test_start_date,train_end_date,train_start_date,initial_investment, sector, method,selected):

    if sector== "Auto":
        ticker_list = ['M&M.NS','MARUTI.NS','TATAMOTORS.NS','EICHERMOT.NS','BAJAJ-AUTO.NS','HEROMOTOCO.NS','TIINDIA.NS','TVSMOTOR.NS','BHARATFORG.NS','ASHOKLEY.NS']
        mother_function(test_end_date,test_start_date,train_end_date,train_start_date,initial_investment, ticker_list, method,selected)
    elif sector == "Banking":
        ticker_list=['HDFCBANK.NS','ICICIBANK.NS','KOTAKBANK.NS','AXISBANK.NS','SBIN.NS','INDUSINDBK.NS','BANKBARODA.NS','AUBANK.NS','FEDERALBNK.NS']
        mother_function(test_end_date,test_start_date,train_end_date,train_start_date,initial_investment, ticker_list, method,selected)
    elif sector == "FMCG":
        ticker_list=['ITC.NS','HINDUNILVR.NS','NESTLEIND.NS','BRITANNIA.NS','TATACONSUM.NS','GODREJCP.NS','DABUR.NS','VBL.NS','MARICO.NS','MCDOWELL-N.NS'] 
        mother_function(test_end_date,test_start_date,train_end_date,train_start_date,initial_investment, ticker_list, method,selected)
    elif sector == "IT":
        ticker_list = ['TCS.NS','INFY.NS','HCLTECH.NS','WIPRO.NS','TECHM.NS','LTIM.NS','PERSISTENT.NS','MPHASIS.NS','COFORGE.NS','LTTS.NS'] 
        mother_function(test_end_date,test_start_date,train_end_date,train_start_date,initial_investment, ticker_list, method,selected)
    elif sector == 'Media':
        ticker_list=['ZEEL.NS','PVR.NS','TV18BRDCST.NS','SUNTV.NS','DISHTV.NS','NETWORK18.NS','NAVNETEDUL.NS','HATHWAY.NS','NDTV.NS', 'NAZARA.NS']
        mother_function(test_end_date,test_start_date,train_end_date,train_start_date,initial_investment, ticker_list, method,selected)
    elif sector == "Metal":
        ticker_list=['TATASTEEL.NS','ADANIENT.NS','HINDALCO.NS','JSWSTEEL.NS','VEDL.NS','JINDALSTEL.NS','APLAPOLLO.NS','SAIL.NS','HINDZINC.NS','NATIONALUM.NS'] 
        mother_function(test_end_date,test_start_date,train_end_date,train_start_date,initial_investment, ticker_list, method,selected)
    elif sector == "Oil & gas":
        ticker_list=['RELIANCE.NS','ONGC.NS','BPCL.NS','IOC.NS','GAIL.NS','PETRONET.NS','HINDPETRO.NS','IGL.NS','OIL.NS', 'ATGL.NS'] 
        mother_function(test_end_date,test_start_date,train_end_date,train_start_date,initial_investment, ticker_list, method,selected)
    elif sector == "Pharmaceutical":
        ticker_list=['SUNPHARMA.NS','CIPLA.NS','CIPLA.NS','DIVISLAB.NS','LUPIN.NS','ALKEM.NS','TORNTPHARM.NS','LAURUSLABS.NS','IPCALAB.NS','AUROPHARMA.NS'] 
        mother_function(test_end_date,test_start_date,train_end_date,train_start_date,initial_investment, ticker_list, method,selected)
    elif sector == "PSU":
        ticker_list=['SBIN.NS','BANKBARODA.NS','CANBK.NS','PNB.NS','UNIONBANK.NS','INDIANB.NS','BANKINDIA.NS','IOB.NS','MAHABANK.NS','CENTRALBK.NS'] 
        mother_function(test_end_date,test_start_date,train_end_date,train_start_date,initial_investment, ticker_list, method,selected)
    elif sector == "Private Bank Sector":
        ticker_list = ['M&M.NS','MARUTI.NS','TATAMOTORS.NS','EICHERMOT.NS','BAJAJ-AUTO.NS','HEROMOTOCO.NS','TIINDIA.NS','TVSMOTOR.NS','BHARATFORG.NS','ASHOKLEY.NS']
        mother_function(test_end_date,test_start_date,train_end_date,train_start_date,initial_investment, ticker_list, method,selected)
    elif sector == "Nifty50":
        ticker_list = ['LT.NS', 'HDFC.NS', 'HDFCBANK.NS', 'GRASIM.NS', 'ITC.NS', 'SUNPHARMA.NS', 'HINDUNILVR.NS', 'TATACONSUM.NS', 'TITAN.NS', 'ADANIPORTS.NS', 'TCS.NS',\
                        'HCLTECH.NS', 'ASIANPAINT.NS', 'COALINDIA.NS', 'BAJFINANCE.NS', 'INFY.NS', 'HDFCLIFE.NS', 'TATASTEEL.NS', 'BAJAJ-AUTO.NS', 'ULTRACEMCO.NS', 'JSWSTEEL.NS',\
                     'DRREDDY.NS', 'NESTLEIND.NS', 'WIPRO.NS', 'TATAMOTORS.NS', 'DIVISLAB.NS', 'CIPLA.NS', 'POWERGRID.NS', 'KOTAKBANK.NS', 'UPL.NS', 'BHARTIARTL.NS', 'HINDALCO.NS',\
                      'BRITANNIA.NS', 'SBILIFE.NS', 'TECHM.NS', 'ICICIBANK.NS', 'HEROMOTOCO.NS', 'BAJAJFINSV.NS', 'RELIANCE.NS', 'AXISBANK.NS', 'ONGC.NS', 'BPCL.NS', 'MARUTI.NS', \
                      'SBIN.NS', 'APOLLOHOSP.NS', 'NTPC.NS', 'ADANIENT.NS', 'M&M.NS', 'INDUSINDBK.NS', 'EICHERMOT.NS'] 
        mother_function(test_end_date,test_start_date,train_end_date,train_start_date,initial_investment, ticker_list, method,selected)


    
def mother_function(test_end_date,test_start_date,train_end_date,train_start_date,initial_investment, ticker_list, method,selected):
    
    
    train_data, test_data = data(test_end_date,test_start_date, train_end_date, train_start_date, ticker_list)

    

    if selected == 'Analysis':
        raw_df, df, pvalues, signals = result_calculation(train_data,test_data,initial_investment, method)
        streamlit_output(raw_df, df, pvalues, train_data, test_data, method)
    elif selected == "Highly recommended stocks":
        raw_df, df, pvalues, signals = result_calculation(train_data,test_data,initial_investment, method)
        raw_df1, df1, pvalues1, signals1 = result_calculation(train_data,test_data,initial_investment, method = 'Cointegration')
        int_df = pd.merge(raw_df, raw_df1, how ='inner')
        #int_df = int_df[int_df['Return(%)']>0]
        int_df.sort_values(by = 'Return(%)', axis=0, ascending=False, inplace = True)
        int_df.reset_index(drop = True, inplace = True)
        int_df.index = int_df.index+1
        df_styled = int_df.style.applymap(lambda x: 'background-color: %s'% 'red' if x < 0 else 'background-color: %s'% 'green'  , subset =[ 'Return(%)'])
        streamlit_output1(df_styled)
    

#=============================================Cointegration and correlation based approach=====================================

def streamlit_output(raw_df, df, pvalues, train_data, test_data, method):
    tab1, tab2= st.tabs(["Best Stock Pairs", "Result"])
                  
    with tab1:
        st.header("The stock pairs with positive return are given below") 
        st.table(raw_df[raw_df['Return(%)']>0]['Stock Pairs'])
    with tab2:
        st.header("The full result with return")
        st.table(df)
    
        


#=========================================Codes for Cluster based analysis=======================================================
  
def streamlit_output1(df_styled):

    st.table(df_styled)
    

#==========================================Codes for the explanation=============================================================

def streamlit_output2():

    col1, col2 = st.columns(2)
    with col1:
        st.image('tradetwins.jpg')
    with col2: 
        gif = load_lottie("https://assets2.lottiefiles.com/packages/lf20_bdsthrsj.json")
        st_lottie(gif, height = 400)
    
    
    
    st.title("Explanation")
    
   

    st.header("Correlation ")
    st.write("Correlation is the easiest approach to look for stock pairs that exhibit long\
              term similar behaviour, which can be exploited to get pairs that give good\
              profit for a given testing period. The Correlation matrix given is given below. The\
              stock pairs exhibiting pearson correlation coefficient of more than 0.5 are\
              selected as the correlated stock pairs. The correlated stock pairs obtained\
              for which the portfolio value is calculated ")
    st.image('auto_corrmap.png', caption = 'Correlation Heat Map', width = 1000)

    st.subheader('Cointegration')
    st.write("A pair of time series \
            variables are said to be cointegrated if they exhibit significant correlation in \
            a long-term time frame while no relationship may be apparent in the shortterm (Engle & Granger, 1987). To identify the pairs which are cointegrated, \
            the coint function defined in the stattools submodule under the statsmodels\
            module of Python is used. The null hypothesis of the coint test assumes that \
                a given pair, which is passed as the two parameters to the function, is not \
            cointegrated (MacKinnon, 2012). Hence, pairs with p-values below the \
            threshold of 0.05 are assumed to be cointegrated")
    st.image('auto_cointegrationmap.png', caption = 'Cointegration heat map', width = 1000)


    st.write('Fruther analysis of cointegrated pairs is to be done. This includes the closing price plot given below')

    st.subheader('Closing price plot')
    st.image('auto_cp_plot.png', caption = ' Time Series plot for the stock pair', width = 1000)
    st.write("Then OLS is performed which eventually leads to residual analysis. The OLS is done with the convention of higher stock as the predictor and lower stock as the target\
               The results for the same are given in the OLS output given below")
    st.image('auto_ols_results.png', caption = 'Python OLS output', width = 1000)
    st.subheader("Residual Plot to check for statinarity")
    st.write("The residual plot showcases whether the time series is stationary or not. This is also evident from the ADF test performed")
    st.image('auto_residual_2.png', caption = 'Python OLS output', width = 1000)
    st.subheader("Z_score plot")
    st.write("The Z-score plot shows the upper and the lower limit which in turn will help in identifying the long and short positions")
    st.image('auto_zscore_plot.png', caption = 'Python OLS output', width = 1000)
    st.subheader("Signal plot")
    st.write("The signal plot shows the trigger points for the pairs wherer the actions need to be taken. This will in turn lead to increase in portfolio value ")
    st.image('auto_signal_plot.png', caption = 'Python OLS output', width = 1000)
    st.subheader("Portfolio Evaluation Plot")
    st.image('auto_portfolio_plot.png', caption = 'Python OLS output', width = 1000)
    


    





#################################################################################################################################
#################################################################################################################################
#################################################################################################################################


######################################## Main function for deployment###########################################################
def main():
    # Set title of the app
    with st.sidebar:
        page = option_menu(
                "Navigate",
                ("Main", "Explanation"))
    if page == "Main":

        gif1 = load_lottie("https://assets8.lottiefiles.com/packages/lf20_xx9zron9.json")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image('tradetwins.jpg')
            st.header("Get the stock pairs in which you can invest and become the owner of your dream car!!! Or get bag full of money to stash in your basement.")
            st.subheader("Just browse the website and get Highly recommended stock pairs. Want to dare, then visit our analysis tab for more risk associated stock pairs. You a market nerd, then explore the explanation tab for detailed explanation ")
            
        with col2:
            st_lottie(gif1, height = 450)

        

        
        with st.container():
            selected = option_menu(menu_title = None, 
                                   options = [ "Highly recommended stocks","Analysis"],
                                   icons = ["calculator", "file-bar-graph"], 
                                   menu_icon ="cast",
                                   default_index = 0,
                                   orientation = "horizontal")
            if selected == "Analysis":
                st.title("Sectoral analysis with Cointegration/Correlation approach")


                with st.form(key = "form1"):
                    
                    
                    sector =  st.selectbox("Choose the sector", ["Auto", "Banking" ,"FMCG", "IT", "Media", "Metal", "Oil & gas", "Pharmaceutical", "PSU", "Private Bank Sector", "Nifty50"])
                    col1, col2 = st.columns(2)
                    with col1:
                        testing_period =  st.selectbox("Investment Period(Months)", [1,2,3,4,5,6,7,8,9,10,11,12])
                    with col2:
                        method = st.selectbox("Method", ["Cointegration", "Correlation"])
                    initial_investment= st.number_input("Initial Investment amount", min_value = 1000, max_value = 1000000, step = 500 )
                    col1, col2, col3 = st.columns([1,1,1])
                    with col2:
                        submit = st.form_submit_button("SUBMIT TO FIND THE BEST STOCK PAIRS TO INVEST", use_container_width = True)
                

                if submit:
                    test_end_date = date.today() + relativedelta(days=-1)
                    test_start_date = test_end_date + relativedelta(months=-(testing_period+1))
                    train_end_date = test_start_date +relativedelta(days=-1)
                    train_start_date = train_end_date +relativedelta(years=-4)
                    
                    # test_end_date = '2022-12-31'
                    # test_start_date = '2022-01-01'
                    # train_end_date = '2021-12-31'
                    # train_start_date = '2018-01-01'



                    
                    if_function(test_end_date,test_start_date,train_end_date,train_start_date,initial_investment, sector,method,selected)
                            
            if selected == "Highly recommended stocks":
                st.title("Sectoral analysis with Cluster assisted Cointegration approach")


                with st.form(key = "form2"):
                    
                    
                    sector =  st.selectbox("Choose the sector", ["Auto", "Banking" ,"FMCG", "IT", "Media", "Metal", "Oil & gas", "Pharmaceutical", "PSU", "Private Bank Sector", "Nifty50"])
                    col1, col2 = st.columns(2)
                    with col1:
                        testing_period =  st.selectbox("Investment Period(Months)", [1,2,3,4,5,6,7,8,9,10,11,12])
                    with col2:
                        initial_investment= st.number_input("Initial Investment amount", min_value = 1000, max_value = 1000000, step = 500 )
                    col1, col2, col3 = st.columns([1,1,1])
                    with col2:
                        submit = st.form_submit_button("SUBMIT TO FIND THE BEST STOCK PAIRS TO INVEST", use_container_width = True)

                if submit:
                    test_end_date = date.today() + relativedelta(days=-1)
                    test_start_date = test_end_date + relativedelta(months=-(testing_period+1))
                    train_end_date = test_start_date +relativedelta(days=-1)
                    train_start_date = train_end_date +relativedelta(years=-4)
                    method = 'Umesh'

                    # test_end_date = '2022-12-31'
                    # test_start_date = '2022-01-01'
                    # train_end_date = '2021-12-31'
                    # train_start_date = '2018-01-01'
                    if_function(test_end_date,test_start_date,train_end_date,train_start_date,initial_investment, sector, method,selected)


    if page == "Explanation":
        
        streamlit_output2()

    
if __name__ == "__main__":
    main()
