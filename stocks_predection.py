from bs4 import BeautifulSoup
import requests
import streamlit as st
import nltk
from datetime import date
import time
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import talib as ta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skewtest, kurtosistest, jarque_bera
from scipy.stats import skew, kurtosis
from plotly.subplots import make_subplots
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk.data



nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

try:
    import newspaper
    NEWSPAPER_AVAILABLE = True
except Exception:
    newspaper = None
    NEWSPAPER_AVAILABLE = False


def flatten_yf_data(data):
    """Flatten MultiIndex columns returned by yfinance >= 0.2."""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


st.set_page_config(layout="wide")
def load_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        st.markdown(f.read(), unsafe_allow_html=True)

load_html("styles.html")
st.markdown("<h1 style='text-align: center; color: white;'>FINANCIAL MARKET ANALYSES</h1>", unsafe_allow_html=True)



tabs = st.tabs(["Home", "Technical Analysis", "Fundamental Analysis", "Porfolio Optimization"])

with tabs[0]:
    col1, col2 = st.columns([3,2])

    with col1:
        st.markdown('<h2 style="color: white; text-align: right;">TOP TRENDING</h2>', unsafe_allow_html=True)
        url_hm = "https://finance.yahoo.com/most-active"
        try:
            _headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            page_hm = requests.get(url_hm, headers=_headers, timeout=10)
            tables_hm = pd.read_html(page_hm.text)
            if tables_hm:
                df_hm = tables_hm[0]
                df_hm.drop(df_hm.columns[2], axis=1, inplace=True)
                st.dataframe(df_hm, use_container_width=True)
            else:
                st.info("No data available from Yahoo Finance Most Active.")
        except Exception:
            st.info("Could not load Most Active data — Yahoo Finance may have changed its structure.")
    with col2:
        try:
            pct_col = next((c for c in df_hm.columns if "%" in str(c)), None)
            sym_col = next((c for c in df_hm.columns if "Symbol" in str(c)), None)
            name_col = next((c for c in df_hm.columns if "Name" in str(c)), None)
            if pct_col and sym_col:
                df_tm = df_hm.copy()
                df_tm[pct_col] = pd.to_numeric(
                    df_tm[pct_col].astype(str).str.replace('%', '').str.replace('+', ''), errors='coerce'
                )
                df_tm = df_tm.dropna(subset=[pct_col])
                hover_cols = [c for c in [name_col, "Volume"] if c and c in df_tm.columns]
                fig = px.treemap(
                    df_tm,
                    path=[sym_col],
                    values=df_tm[pct_col].abs(),
                    color=pct_col,
                    color_continuous_scale=["red", "lightgrey", "green"],
                    color_continuous_midpoint=0,
                    hover_data=hover_cols if hover_cols else None,
                )
                fig.update_layout(height=550, coloraxis_colorbar=dict(title="% Change"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Could not find % Change or Symbol column in the data.")
        except Exception:
            st.info("Treemap unavailable — load the Most Active table first.")
    st.write("____________")
    col1, col2 = st.columns(2)

    # Top gainers
    with col1:
        st.markdown('<h2 style="color: green; text-align: center;">TOP GAINERS</h2>', unsafe_allow_html=True)
        url = "https://finance.yahoo.com/gainers"
        try:
            _headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            page = requests.get(url, headers=_headers, timeout=10)
            tables_g = pd.read_html(page.text)
            if tables_g:
                df_win = tables_g[0]
                df_win.drop(df_win.columns[2], axis=1, inplace=True)
                
                styled_df = df_win.style.background_gradient(cmap="Greens",axis=None   )
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.info("No gainers data available.")
        except Exception:
            st.info("Could not load Top Gainers — Yahoo Finance may have changed its structure.")
    # Top losers
    with col2:
        st.markdown('<h2 style="color: red; text-align: center;">TOP LOSERS</h2>', unsafe_allow_html=True)
        url_los = "https://finance.yahoo.com/losers"
        try:
            _headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            page_los = requests.get(url_los, headers=_headers, timeout=10)
            tables_los = pd.read_html(page_los.text)
            if tables_los:
                df_los = tables_los[0]
                df_los.drop(df_los.columns[2], axis=1, inplace=True)

                styled_df = df_los.style.background_gradient(cmap="Reds",axis=None   )
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.info("No losers data available.")
        except Exception:
            st.info("Could not load Top Losers — Yahoo Finance may have changed its structure.")

    st.markdown('<h1 style="color: white; text-align: center;">LATEST NEWS</h1>', unsafe_allow_html=True)
    st.write("____________")

    col1,col2=st.columns(2)
    with col1:
       
        image_url = 'https://tse1.mm.bing.net/th/id/OIP.DV4Jc_YsKrP8UVzgsssBxgHaBD?cb=thfc1falcon2&rs=1&pid=ImgDetMain&o=7&rm=3'
        st.markdown(f'<div style="display: flex; justify-content: center; background-color: white; padding: 15px;"><img src="{image_url}" height="50"/></div>', unsafe_allow_html=True)
        st.write("__________________")

        url = 'https://finance.yahoo.com/topic/latest-news/'
        try:
            _headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=_headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                headlines = soup.find_all('h3', class_='Mb(5px)') or soup.find_all('h3')
                for headline in headlines[:15]:
                    st.markdown(f"<h4 style='text-align: left;'>• {headline.text}</h4>", unsafe_allow_html=True)
            else:
                st.info('Could not retrieve Yahoo Finance news.')
        except Exception:
            st.info('Could not load Yahoo Finance news.')

    with col2:
        image_url = 'https://d1rwhvwstyk9gu.cloudfront.net/2018/02/Google-Finance.png'
        st.markdown(f'<div style="display: flex; justify-content: center; background-color: white; padding: 15px;"><img src="{image_url}" height="50"/></div>', unsafe_allow_html=True)
        st.write("__________________")

        # Google Finance is JS-rendered — use Yahoo Finance Markets RSS instead
        try:
            _headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get("https://finance.yahoo.com/rss/topstories", headers=_headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                for item in items[:15]:
                    title = item.find('title')
                    link = item.find('link')
                    if title:
                        if link and link.text:
                            st.markdown(f"<h4 style='text-align: left;'>• <a href='{link.text}' target='_blank' style='color:inherit;text-decoration:none;'>{title.text}</a></h4>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h4 style='text-align: left;'>• {title.text}</h4>", unsafe_allow_html=True)
            else:
                st.info('Could not retrieve news feed.')
        except Exception:
            st.info('Could not load news feed.')


with tabs[1]:
    col1, col2, col3 = st.columns([3,3,1])
    with col1:
        s_type = st.selectbox("Select Equity type", ("Stocks", "Forex", "Commodities", "Cryptos"))
           
    with col2:
        if s_type == "Stocks":
            stk = ["AAPL", "MSFT", "AMZN", "GOOGL","PYPL","JNJ","META", "JPM","AMD","ADBE" ,"NVDA","GS","BABA" ,"V", "AXP", "DIS", "WMT", "HD", "NKE", "MCD", "XOM", "CVX", "KO", "PG", "VZ","TSLA","IBM","BA"]
            selected_stocks = st.selectbox("Select Stock", stk)
            with col3:
                curr = ('USD',)
                selected_curr = st.radio('Select currency', curr)
        elif s_type == "Forex":
            frx = ['USD/EUR', 'USD/JPY', 'USD/GBP', 'USD/CAD', 'USD/CHF', 'USD/CNY', 'USD/SEK', 'USD/NZD', 'USD/SGD', 'USD/HKD', 'USD/NOK', 'USD/KRW', 'USD/TRY', 'USD/RUB', 'USD/ZAR', 'USD/BRL', 'USD/DKK']
            selected_stocks = st.selectbox("Select Currency", frx)
        elif s_type == "Commodities":
            Commodities = {"GC=F": "Gold", "SI=F":'Silver', "CL=F":'Crude Oil WTI',"BZ=F":'Crude Oil Brent', "NG=F":'Natural Gas',"HG=F": 'Copper',"PL=F": 'Platinum', "PA=F":'Palladium', "ZC=F":'Corn', "ZS=F":'Soybean ', " ZW=F":'Wheat', "CT=F":'Cotton', "KC=F":'Coffee',  "CC=F":'Cocoa', "SB=F":'Sugar', "LBS=F":'Lumber ', "OJ=F":'Orange Juice'}
            selected_stocks = st.selectbox("Select Commodities", Commodities.keys(), format_func=lambda x: Commodities[x])
            with col3:
                curr = ("USD",)
                selected_curr = st.radio('Select currency', curr)
        elif s_type == "Cryptos":
            crypto = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'BNB', 'USDT', 'LINK', 'DOT', 'ADA', 'XLM', 'EOS', 'TRX', 'XMR', 'XTZ', 'DASH', 'DOGE']
            selected_stocks = st.selectbox("Select Crypto", crypto)
            with col3:
                curr = ('USD', 'EUR')
                selected_curr = st.radio('Select currency', curr)

    perriod = {"1d": "1 Day", "5d": "5 Days", "1mo": "1 Month", "3mo": "3 Months", "6mo": "6 Months","YTD": "Year to date" ,"1y": "1 Year", "2y": "2 Years", "5y": "5 Years","10y": "10 years"}
    interval_short = {"1m": "1 Minute", "2m": "2 Minutes", "5m": "5 Minutes", "15m": "15 Minutes", "30m": "30 Minutes", "1h": "1 Hour"}
    interval_long = {"1d": "1 Day", "5d": "5 Days", "1wk": "1 Week", "1mo": "1 Month"}

    col1, col2 = st.columns([1, 1])

    with col1:
        select_period = st.select_slider("Select period", options=perriod.keys(), format_func=lambda x: perriod[x])

    with col2:
        if select_period in ["1d", "5d"]:
            select_interval = st.selectbox("Select interval", options=interval_short.keys(), format_func=lambda x: interval_short[x])
        elif select_period in ["1mo", "3mo", "6mo","YTD" ,"1y", "2y", "5y", "10y"]:
            select_interval = st.selectbox("Select interval", options=interval_long.keys(), format_func=lambda x: interval_long[x])

    if select_interval in ["1m", "2m", "5m", "15m", "30m", "1h"]:
        dt = "Datetime"
    elif select_interval in ["1d", "5d", "1wk", "1mo"]:
        dt = "Date"

    st.write("____________")

    if s_type=="Stocks":
        ticker = yf.Ticker(selected_stocks)
        details = ticker.info

        col1, col2  = st.columns([1, 1])
        with col1:
            st.markdown(f"<center><h2>  {details['longName']}</h2></center>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<center><h3>Sector : {details['sector']}</h3></center>", unsafe_allow_html=True)
   
    st.write("____________")

    col1, col2  = st.columns([1, 1])

    with col1:
        if s_type == "Cryptos":
            def load_data(ticker, selected_curr, select_period, select_interval):
                ticker_with_curr = ticker + '-' + selected_curr
                data = yf.download(ticker_with_curr, period=select_period, interval=select_interval, auto_adjust=True)
                data.reset_index(inplace=True)
                return flatten_yf_data(data)
            data = load_data(selected_stocks, selected_curr, select_period, select_interval)

        elif s_type == "Forex":
            def load_data(selected_stocks, select_period, select_interval):
                ticker1, ticker2 = selected_stocks.split('/')
                data = yf.download(ticker1, period=select_period, interval=select_interval, auto_adjust=True)
                data.reset_index(inplace=True)
                return flatten_yf_data(data)
            data = load_data(selected_stocks, select_period, select_interval)

        elif s_type == "Commodities":
            def load_data_c(ticker, select_period, select_interval):
                data = yf.download(ticker, period=select_period, interval=select_interval, auto_adjust=True)
                data.reset_index(inplace=True)
                return flatten_yf_data(data)
            data = load_data_c(selected_stocks, select_period, select_interval)
        else:
            def load_data(ticker, select_period, select_interval):
                data = yf.download(ticker, period=select_period, interval=select_interval, auto_adjust=True)
                data.reset_index(inplace=True)
                return flatten_yf_data(data)
            data = load_data(selected_stocks, select_period, select_interval)

        st.markdown(f"<h3 style='text-align: center;'>Stock Prices Data for {selected_stocks}</h3>", unsafe_allow_html=True)

        st.dataframe(data.tail(7))

    with col2:
        st.markdown(f"<h3 style='text-align: center;'>Evolution of {selected_stocks}</h3>", unsafe_allow_html=True)
        
        first_open = data.loc[0, 'Open']
        last_open = data.loc[len(data) - 1, 'Open']
        rate_of_change = ((last_open - first_open) / first_open) * 100  
        first_open_v = f"{first_open:.3f} $"
        last_open_v = f"{last_open:.3f} $"
        rate_of_change_v = f"{rate_of_change:.3f}"

        # Show the first and last open values with rate of change
        if last_open < first_open:
            
            st.write(f"<span style='color:white; font-size: xx-large; font-weight: bold; text-align: center; display: block;'>{first_open_v}</span>", unsafe_allow_html=True)
            st.write(f"<span style='color:red; font-size: x-large; font-weight: bold; text-align: center; display: block;'>{rate_of_change_v}%</span>", unsafe_allow_html=True)
            st.write(f"<span style='color:red; font-size: xx-large; font-weight: bold; text-align: center; display: block;'>{last_open_v}</span>", unsafe_allow_html=True)
        else:
            st.write(f"<span style='color:white; font-size: xx-large; font-weight: bold; text-align: center; display: block;'>{first_open_v}</span>", unsafe_allow_html=True)
            st.write(f"<span style='color:green; font-size: x-large; font-weight: bold; text-align: center; display: block;'>{rate_of_change_v}%</span>", unsafe_allow_html=True)
            st.write(f"<span style='color:green; font-size: xx-large; font-weight: bold; text-align: center; display: block;'>{last_open_v}</span>", unsafe_allow_html=True)


    st.markdown('<h2 style="color: white; text-align: center;">Time Series Analysis </h2>', unsafe_allow_html=True)
    # data = data[data[dt].dt.dayofweek < 5]  # Garder seulement les jours de semaine

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[dt],y=data["Open"],name="stock_open"))
    fig.add_trace(go.Scatter(x=data[dt],y=data["Close"],name="stock_close"))
    fig.layout.update(xaxis_rangeslider_visible=True,legend=dict(orientation="h"))
    st.plotly_chart(fig,use_container_width=True,height=900)

    def plot_stock_data():
        st.markdown('<h2 style="color: white; text-align: center;">Candel-Stick  Pattern</h2>', unsafe_allow_html=True)

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=data[dt],
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                    increasing=dict(line=dict(color='#58FA58')),
                    decreasing=dict(line=dict(color='#FA5858'))
                )
            ]
        )

        # doji = ta.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])
        # gravestone_doji = ta.CDLGRAVESTONEDOJI(data['Open'], data['High'], data['Low'], data['Close'])
        # dragonfly_doji = ta.CDLDRAGONFLYDOJI(data['Open'], data['High'], data['Low'], data['Close'])
        # hammer = ta.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
        # shooting_star = ta.CDLSHOOTINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
        # engulfing = ta.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])
        # morning_star = ta.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
        # evening_star = ta.CDLEVENINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
        # dark_cloud_cover = ta.CDLDARKCLOUDCOVER(data['Open'], data['High'], data['Low'], data['Close'])
        # piercing = ta.CDLPIERCING(data['Open'], data['High'], data['Low'], data['Close'])
        # three_white_soldiers = ta.CDL3WHITESOLDIERS(data['Open'], data['High'], data['Low'], data['Close'])
        # three_black_crows = ta.CDL3BLACKCROWS(data['Open'], data['High'], data['Low'], data['Close'])
        # harami = ta.CDLHARAMI(data['Open'], data['High'], data['Low'], data['Close'])
        # bullish_belt_hold = ta.CDLBELTHOLD(data['Open'], data['High'], data['Low'], data['Close'])
        # bearish_belt_hold = ta.CDLBELTHOLD(data['Open'], data['High'], data['Low'], data['Close'])
        # marubozu = ta.CDLMARUBOZU(data['Open'], data['High'], data['Low'], data['Close'])
        # kicker = ta.CDLKICKING(data['Open'], data['High'], data['Low'], data['Close'])
        # abandoned_baby = ta.CDLABANDONEDBABY(data['Open'], data['High'], data['Low'], data['Close'])
        # harami_cross = ta.CDLHARAMICROSS(data['Open'], data['High'], data['Low'], data['Close'])
        # upside_gap_two_crows = ta.CDLUPSIDEGAP2CROWS(data['Open'], data['High'], data['Low'], data['Close'])
        # three_inside_up = ta.CDL3INSIDE(data['Open'], data['High'], data['Low'], data['Close'])
        # three_inside_down = ta.CDL3INSIDE(data['Open'], data['High'], data['Low'], data['Close'])
        # rising_three_methods = ta.CDLRISEFALL3METHODS(data['Open'], data['High'], data['Low'], data['Close'])
        # falling_three_methods = ta.CDLRISEFALL3METHODS(data['Open'], data['High'], data['Low'], data['Close'])

        
        
        doji = ta.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])
        gravestone_doji = ta.CDLGRAVESTONEDOJI(data['Open'], data['High'], data['Low'], data['Close'])
        dragonfly_doji = ta.CDLDRAGONFLYDOJI(data['Open'], data['High'], data['Low'], data['Close'])
        hanging_man = ta.CDLHANGINGMAN(data['Open'], data['High'], data['Low'], data['Close'])
        piercing_pattern = ta.CDLPIERCING(data['Open'], data['High'], data['Low'], data['Close'])
        abandoned_baby = ta.CDLABANDONEDBABY(data['Open'], data['High'], data['Low'], data['Close'])
        morning_star = ta.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
        evening_star = ta.CDLEVENINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
        hammer = ta.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
        shooting_star = ta.CDLSHOOTINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
        engulfing = ta.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])
        dark_cloud_cover = ta.CDLDARKCLOUDCOVER(data['Open'], data['High'], data['Low'], data['Close'])
        three_white_soldiers = ta.CDL3WHITESOLDIERS(data['Open'], data['High'], data['Low'], data['Close'])
        three_black_crows = ta.CDL3BLACKCROWS(data['Open'], data['High'], data['Low'], data['Close'])
        three_inside_up = ta.CDL3INSIDE(data['Open'], data['High'], data['Low'], data['Close'])
        three_inside_down = ta.CDL3INSIDE(data['Open'], data['High'], data['Low'], data['Close'])
        marubozu = ta.CDLMARUBOZU(data['Open'], data['High'], data['Low'], data['Close'])
        kicker = ta.CDLKICKING(data['Open'], data['High'], data['Low'], data['Close'])
        harami = ta.CDLHARAMI(data['Open'], data['High'], data['Low'], data['Close'])

                    
        candlestick_patterns = st.multiselect("Select Candlestick Patterns", ["Doji", "Gravestone Doji", "Dragonfly Doji", "Hammer", "Shooting Star", "Morning Star", "Evening Star",
                                                                                "Engulfing", "Hanging Man", "Piercing", "Abandoned Baby",
                                                                                "Dark Cloud Cover", "Three White Soldiers", "Three Black Crows", "Three Inside Up",
                                                                                "Three Inside Down", "Marubozu", "Kicker", "Harami"])

        # Plot selected candlestick patterns
        for pattern in candlestick_patterns:
            if pattern == "Doji":
                fig.add_trace(go.Scatter(x=data[dt][doji != 0], y=data['Close'][doji != 0], mode='markers', name='Doji', marker=dict(symbol='circle-open',size=30,opacity=0.8,line=dict(width=2))))
            elif pattern == "Hammer":
                fig.add_trace(go.Scatter(x=data[dt][hammer != 0], y=data['Close'][hammer != 0], mode='markers', name='Hammer', marker=dict(symbol='circle-open',size=30,opacity=0.8,line=dict(width=2))))
            elif pattern == "Shooting Star":
                fig.add_trace(go.Scatter(x=data[dt][shooting_star != 0], y=data['Close'][shooting_star != 0], mode='markers', name='Shooting Star', marker=dict(symbol='circle-open',size=40,opacity=0.8,line=dict(width=2))))
            elif pattern == "Engulfing":
                fig.add_trace(go.Scatter(x=data[dt][engulfing != 0], y=data['Close'][engulfing != 0], mode='markers', name='Engulfing', marker=dict(symbol='circle-open',size=40,opacity=0.8,line=dict(width=2))))
            elif pattern == "Morning Star":
                fig.add_trace(go.Scatter(x=data[dt][morning_star != 0], y=data['Close'][morning_star != 0], mode='markers', name='Morning Star', marker=dict(symbol='circle-open',size=40,opacity=0.8,line=dict(width=2))))
            elif pattern == "Evening Star":
                fig.add_trace(go.Scatter(x=data[dt][evening_star != 0], y=data['Close'][evening_star != 0], mode='markers', name='Evening Star', marker=dict(symbol='circle-open',size=40,opacity=0.8,line=dict(width=2))))
            elif pattern == "Hanging Man":
                fig.add_trace(go.Scatter(x=data[dt][hanging_man != 0], y=data['Close'][hanging_man != 0], mode='markers', name='Hanging Man', marker=dict(symbol='circle-open',size=30,opacity=0.8,line=dict(width=2))))
            elif pattern == "Piercing":
                fig.add_trace(go.Scatter(x=data[dt][piercing_pattern != 0], y=data['Close'][piercing_pattern != 0], mode='markers', name='Piercing', marker=dict(symbol='circle-open',size=40,opacity=0.8,line=dict(width=2))))
            elif pattern == "Abandoned Baby":
                fig.add_trace(go.Scatter(x=data[dt][abandoned_baby != 0], y=data['Close'][abandoned_baby != 0], mode='markers', name='Abandoned Baby', marker=dict(symbol='circle-open',size=34,opacity=0.8,line=dict(width=2))))
            elif pattern == "Dark Cloud Cover":
                fig.add_trace(go.Scatter(x=data[dt][dark_cloud_cover != 0], y=data['Close'][dark_cloud_cover != 0], mode='markers', name='Dark Cloud Cover', marker=dict(symbol='circle-open',size=40,opacity=0.8,line=dict(width=2))))
            elif pattern == "Three White Soldiers":
                fig.add_trace(go.Scatter(x=data[dt][three_white_soldiers != 0], y=data['Close'][three_white_soldiers != 0], mode='markers', name='Three White Soldiers', marker=dict(symbol='circle-open',size=40,opacity=0.8,line=dict(width=2))))
            elif pattern == "Three Black Crows":
                fig.add_trace(go.Scatter(x=data[dt][three_black_crows != 0], y=data['Close'][three_black_crows != 0], mode='markers', name='Three Black Crows', marker=dict(symbol='circle-open',size=40,opacity=0.8,line=dict(width=2))))
            elif pattern == "Three Inside Up":
                fig.add_trace(go.Scatter(x=data[dt][three_inside_up != 0], y=data['Close'][three_inside_up != 0], mode='markers', name='Three Inside Up', marker=dict(symbol='circle-open',size=40,opacity=0.8,line=dict(width=2))))
            elif pattern == "Three Inside Down":
                fig.add_trace(go.Scatter(x=data[dt][three_inside_down != 0], y=data['Close'][three_inside_down != 0], mode='markers', name='Three Inside Down', marker=dict(symbol='circle-open',size=40,opacity=0.8,line=dict(width=2))))
            elif pattern == "Marubozu":
                fig.add_trace(go.Scatter(x=data[dt][marubozu != 0], y=data['Close'][marubozu != 0], mode='markers', name='Marubozu', marker=dict(symbol='circle-open',size=30,opacity=0.8,line=dict(width=2))))
            elif pattern == "Kicker":
                fig.add_trace(go.Scatter(x=data[dt][kicker != 0], y=data['Close'][kicker != 0], mode='markers', name='Kicker', marker=dict(symbol='circle-open',size=40,opacity=0.8,line=dict(width=2))))
            elif pattern == "Harami":
                fig.add_trace(go.Scatter(x=data[dt][harami != 0], y=data['Close'][harami != 0], mode='markers', name='Harami', marker=dict(symbol='circle-open',size=40,opacity=0.8,line=dict(width=2))))
            elif pattern == "Gravestone Doji":
                fig.add_trace(go.Scatter(x=data[dt][gravestone_doji != 0], y=data['Close'][gravestone_doji != 0], mode='markers', name='Gravestone Doji', marker=dict(symbol='circle-open',size=30,opacity=0.8,line=dict(width=2))))
            elif pattern == "Dragonfly Doji":
                fig.add_trace(go.Scatter(x=data[dt][dragonfly_doji != 0], y=data['Close'][dragonfly_doji != 0], mode='markers', name='Dragonfly Doji', marker=dict(symbol='circle-open',size=30,opacity=0.8,line=dict(width=2))))
        # Update the layout
        fig.update_layout(xaxis_rangeslider_visible=True,height=700,  legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    plot_stock_data()



    ####################################################

    def plot_RSI():
        st.markdown('<h2 style="color: white; text-align: center;">Relative Strength Index</h2>', unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2)

        # Add stock value subplot
        fig.add_trace(go.Scatter(x=data[dt], y=data["Close"], name="stock value"), row=1, col=1)

        # Add RSI subplot
        fig.add_trace(go.Scatter(x=data[dt], y=ta.RSI(data["Close"]), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line=dict(color='red'), row=2, col=1, annotation_text="Overbought (70)", annotation_position="top right")
        fig.add_hline(y=30, line=dict(color='green'), row=2, col=1, annotation_text="Oversold (30)", annotation_position="bottom right")

        # Update the layout
        fig.update_layout(xaxis_rangeslider_visible=True,
            height=600,legend=dict(orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)  # Use use_container_width=True parameter here

    plot_RSI()
    ###########################

    ######################

    def plot_Mean_Average():
        st.markdown('<h2 style="color: white; text-align: center;">Moving Averages</h2>', unsafe_allow_html=True)

        moving_average_types = st.multiselect("Select Moving Average Types", ["SMA", "EMA", "WMA", "DEMA", "TEMA"], default=["SMA"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data[dt], y=data["Close"], name="stock value" , line=dict(width=2)))


        for moving_average_type in moving_average_types:
            if moving_average_type == "SMA":
                fig.add_trace(go.Scatter(x=data[dt], y=ta.SMA(data["Close"]), name="SMA", line=dict(color="red")))
            elif moving_average_type == "EMA":
                fig.add_trace(go.Scatter(x=data[dt], y=ta.EMA(data["Close"]), name="EMA",line=dict(color="orange")))
            elif moving_average_type == "WMA":
                fig.add_trace(go.Scatter(x=data[dt], y=ta.WMA(data["Close"]), name="WMA",line=dict(color="green")))
            elif moving_average_type == "DEMA":
                fig.add_trace(go.Scatter(x=data[dt], y=ta.DEMA(data["Close"]), name="DEMA",line=dict(color="purple")))
            elif moving_average_type == "TEMA":
                fig.add_trace(go.Scatter(x=data[dt], y=ta.TEMA(data["Close"]), name="TEMA",line=dict(color="yellow")))                    
            # Update the layout
        fig.update_layout(
            width=1000,  # Specify the width of the plot
            height=600,  # Specify the height of the plot)
            xaxis_rangeslider_visible=True,legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    plot_Mean_Average()


    def MACD():
        st.markdown('<h2 style="color: white; text-align: center;">MACD</h2>', unsafe_allow_html=True)
        st.markdown('<h6 style="color: white; text-align: center;">Moving Averages Convergence Divergence</h2>', unsafe_allow_html=True)

        macd, macd_signal, macd_hist = ta.MACD(data['Close'])
        colors = ['green' if value > 0 else 'red' for value in macd_hist]
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2, subplot_titles=("Close Price", "MACD"))

        # Add Close price trace
        fig.add_trace(go.Scatter(x=data[dt], y=data['Close'], mode='lines', name='Close Price'), row=1, col=1)

        # Add MACD and Signal Line traces
        fig.add_trace(go.Scatter(x=data[dt], y=macd, mode='lines', name='MACD', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=data[dt], y=macd_signal,mode='lines', name='Signal Line', line=dict(color='orange')), row=2, col=1)

        # Add MACD Histogram bars
        fig.add_trace(go.Bar(x=data[dt], y=macd_hist, name='MACD Histogram',marker=dict(color=colors)), row=2, col=1)
        fig.add_hline(y=0, line=dict(color='white'), row=2, col=1)


        fig.update_layout(height=800, width=1000,xaxis_rangeslider_visible=True,legend=dict(orientation="h"))

        st.plotly_chart(fig, use_container_width=True)

    MACD()

    def plot_BBANDS(data):
        st.markdown('<h2 style="color: white; text-align: center;">Bollinger Bands</h2>', unsafe_allow_html=True)

        upper_band, middle_band, lower_band = ta.BBANDS(data["Close"])
        
        fig = go.Figure()
        fig = go.Figure(
            data=[go.Candlestick(x=data[dt],open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                    increasing=dict(line=dict(color='#58FA58')),
                    decreasing=dict(line=dict(color='#FA5858'))
                )])
        fig.add_trace(go.Scatter(x=data[dt], y=upper_band, name="Upper Bollinger Band"))
        fig.add_trace(go.Scatter(x=data[dt], y=middle_band, name="Middle Bollinger Band", line=dict(color='green')))
        fig.add_trace(go.Scatter(x=data[dt], y=lower_band, name="Lower Bollinger Band"))
        fig.update_layout(height=800, width=1000,xaxis_rangeslider_visible=True,legend=dict(orientation="h"))
        fig.add_trace(go.Scatter(x=data[dt], y=data["Close"], name="stock value", line=dict(color='yellow')))


        st.plotly_chart(fig, use_container_width=True)
    plot_BBANDS(data)
#####################################################################################################################################

    st.markdown('<h1 style="color: White; text-align: center;">FORECASTING</h2>', unsafe_allow_html=True)

    n_years = st.slider("years of prediction:",1,4)
    period = n_years*365
    if hasattr(data[dt].dt, 'tz') and data[dt].dt.tz is not None:
        data[dt] = data[dt].dt.tz_localize(None)

    ###############forecasting with profet################
    df_train = data [[dt,'Close']]
    df_train = df_train.rename(columns={dt:"ds", "Close":"y"})

    m=Prophet()
    m.fit(df_train)
    future=m.make_future_dataframe(periods=period)
    forecast=m.predict(future)

    st.markdown('<h3 style="color: White; text-align: center;">Forecast Data</h2>', unsafe_allow_html=True)
    st.write(forecast.tail())
    ##########
    fig_1=plot_plotly(m,forecast)
    st.plotly_chart(fig_1, use_container_width=True)
    ####
    y_true = df_train['y'].values
    y_pred = forecast['yhat'].values[:len(y_true)]
    ##############robustesse du model ######################
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        st.write("Mean Squared Error (MSE):", mse)
    with col2:
        st.write("Root Mean Squared Error (RMSE):", rmse)
    with col3:
        st.write("Mean Absolute Error (MAE):", mae)
    with col4:
        st.write("Mean Absolute Percentage Error (MAPE):", mape)
    #############################
    if select_period in ["2y", "5y","10y"]:
        st.markdown('<h3 style="color: White; text-align: center;">Forecast Components</h2>', unsafe_allow_html=True)
        components_fig = m.plot_components(forecast)
        st.pyplot(components_fig, use_container_width=True)

    



       
    with tabs[2]:
        col1, col2 = st.columns([1,1])


        with col1:
            s_type = st.selectbox("Select Equity type", ("Stocks", "Forex", "Commodities", "Cryptos"), key="asset_type")
                   
        with col2:
            if s_type == "Stocks":
                stk = ["AAPL", "MSFT", "AMZN", "GOOGL","PYPL","JNJ","META", "JPM","AMD","ADBE" ,"NVDA","GS","BABA" ,"V", "AXP", "DIS", "WMT", "HD", "NKE", "MCD", "XOM", "CVX", "KO", "PG", "VZ","TSLA","IBM","BA"]
                selec_stock = st.selectbox("Select Stock", stk, key="stock_selection")
            elif s_type == "Forex":
                frx = ["EURUSD=X","EURCHF=X", 'USDJPY=X', 'GBPUSD=X',"EURCAD=X","EURGBP=X", 'USDCAD=X', 'USDCHF=X', 'USDCNY=X', 'USDSEK=X', 'USDNZD=X', 'USDSGD=X', 'USDHKD=X', 'USDNOK=X', 'USDKRW=X', 'USDTRY=X', 'USDRUB=X', 'USDZAR=X', 'USDBRL=X', 'USDDKK=X']
                selec_stock = st.selectbox("Select Currency", frx, key="currency_selection")
            elif s_type == "Commodities":
                Commodities= ['GC=F', "SI=F", "CL=F","BZ=F", "NG=F","HG=F","PL=F", "PA=F", "ZC=F", "ZS=F", " ZW=F", "CT=F", "KC=F",  "CC=F", "SB=F", "LBS=F", "OJ=F"]
                selec_stock = st.selectbox("Select Commodities", Commodities, key="commodities_selection")

            elif s_type == "Cryptos":
                crypto = ["BTC-USD", 'ETH-USD', 'XRP-USD', 'BTC-USD', 'BCH-USD', 'BNB-USD', 'USDT-USD', 'LINK-USD', 'DOT-USD', 'ADA-USD', 'XLM-USD', 'EOS-USD', 'TRX-USD', 'XMR-USD', 'XTZ-USD', 'DASH-USD', 'DOGE-USD']
                selec_stock = st.selectbox("Select Crypto", crypto, key="crypto_selection")

        if s_type == "Stocks":
            ticker = yf.Ticker(selec_stock)
            details = ticker.info

            df_dt = pd.DataFrame([details])
            df_details = df_dt


            _cols_to_drop = ["industryKey","industryDisp","sectorKey","sectorDisp","longBusinessSummary","companyOfficers","uuid"]
            x = df_details.drop(columns=[c for c in _cols_to_drop if c in df_details.columns])
            x = x.reindex(columns=['symbol'] + [col for col in x.columns if col != 'symbol'])

            xx=(x.head(1))
            st.markdown('<h2 style="color: White; text-align: center;">Performance Analyses </h2>', unsafe_allow_html=True)

            # st.dataframe(xx)
            st.table(xx)

            with st.container(border=True):
                st.markdown(f"<center><h2><font color='gray'>Recommendation:  <br></font> {details['recommendationMean']}</h2></center>", unsafe_allow_html=True)


            col1, col2, col3 ,col4= st.columns([1,1,1,1])

            with col1:
                with st.container(border=True):
                    st.markdown('<h3 style="color: White; text-align: center;">• Performance •</h3>', unsafe_allow_html=True)
                    try:
                        st.markdown(f"<center><h4><font color='gray'>Name:<br></font> {details['longName']}</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>Sector:<br></font> {details['sector']}</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>Market Capitalization:<br></font> {details['marketCap']}</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>Revenue:<br></font> {details['totalRevenue']}</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>Enterprise Value:<br></font> {details['enterpriseValue']}</h4></center>", unsafe_allow_html=True)
                    except :

                        print("")

            with col2:
                with st.container(border=True):
                    st.markdown('<h3 style="color: White; text-align: center;">• Performance Index •</h3>', unsafe_allow_html=True)

                    try:
                        st.markdown(f"<center><h4><font color='gray'>Earnings Growth:<br></font> {details['earningsGrowth']}</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>Revenue Growth:<br></font> {details['revenueGrowth']}</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>Gross Margins:<br></font> {details['grossMargins']}</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>EBITDA Margins:<br></font> {details['ebitdaMargins']}</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>Operating Margins:<br></font> {details['operatingMargins']}</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>Profit Margin:<br></font> {details['profitMargins']}</h4></center>", unsafe_allow_html=True)
                    except :
                        print("")


            with col3:
                with st.container(border=True):
                    st.markdown('<h3 style="color: White; text-align: center;">• Risk Score •</h3>', unsafe_allow_html=True)
                    try:
                        st.markdown(f"<center><h4><font color='gray'>Audit Risk:<br></font> {details['auditRisk']}</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>Board Risk:<br></font> {details['boardRisk']}</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>Compensation Risk:<br></font> {details['compensationRisk']}</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>Shareholder Rights Risk:<br></font> {details['shareHolderRightsRisk']}</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>Overall Risk:<br></font> {details['overallRisk']}</h4></center>", unsafe_allow_html=True)
                    except :
                        print("")
                   
            with col4:
                with st.container(border=True):
                    st.markdown('<h3 style="color: White; text-align: center;">• Dividend •</h3>', unsafe_allow_html=True)
                    try:
                        st.markdown(f"<center><h4><font color='gray'>Dividend Rate:<br></font> {details['dividendRate'] }</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>Dividend Yield:<br></font> {details['dividendYield'] }</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>AVG Dividend Yield :<br></font> {details['fiveYearAvgDividendYield']}</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>Last Dividend Value:<br></font> {details['lastDividendValue']}</h4></center>", unsafe_allow_html=True)
                        st.markdown(f"<center><h4><font color='gray'>Payout Ratio:<br></font> {details['payoutRatio']}</h4></center>", unsafe_allow_html=True)
                    except :
                        print("")


        def analyze_sentiment(text):
            vader = SentimentIntensityAnalyzer()
            scores = vader.polarity_scores(text)

            # NLTK sentiment categories (fine-grained analysis)
            sentiment = max(scores, key=scores.get)
            confidence = scores[sentiment]

            # Classify sentiment based on compound score (adjusted from Response A)
            if scores['compound'] >= 0.05:  # Positive
                return "Positive", confidence
            elif scores['compound'] <= -0.05:  # Negative
                return "Negative", confidence
            else:  # Neutral
                return "Neutral", confidence

        def main():
            st.markdown('<h1 style="color: White; text-align: center;">Sentiment Analysis of News</h1>', unsafe_allow_html=True)

            # Use yfinance .news API — avoids 429 rate-limit from scraping the quote page
            ticker_obj = yf.Ticker(selec_stock)
            news_items = ticker_obj.news
            headlines = [item.get('content', {}).get('title', '') or item.get('title', '')
                         for item in (news_items or []) if item]
            headlines = [h for h in headlines if h]

            if not headlines:
                st.info("No news found for this ticker.")
                return

            pos_count = 0
            neg_count = 0
            neu_count = 0
            total_confidence_pos = 0
            total_confidence_neg = 0

            col1, col2, col3 = st.columns([1.8, 0.2, 1])
            with col1:
                with st.container(border=True):
                    st.markdown('<h2 style="color: White; text-align: center;"> Article Sentiment Analysis</h2>', unsafe_allow_html=True)
                    st.write("_________")

                    for article_text in headlines:
                        sentiment, confidence = analyze_sentiment(article_text)
                        st.markdown(f"<h3 style='text-align: left;'>•  {article_text}</h3>", unsafe_allow_html=True)

                        if sentiment == 'Positive':
                            pos_count += 1
                            total_confidence_pos += confidence
                            st.write(f'<span style="color:green">  Sentiment: {sentiment} ({confidence:+.2f} confidence)</span>', unsafe_allow_html=True)
                        elif sentiment == 'Negative':
                            neg_count += 1
                            total_confidence_neg += confidence
                            st.write(f'<span style="color:red">  Sentiment: {sentiment} ({-confidence:.2f} confidence)</span>', unsafe_allow_html=True)
                        else:
                            neu_count += 1
                            st.write(f'<span style="color:grey">  Sentiment: {sentiment} (0.00 confidence)</span>', unsafe_allow_html=True)

                total_confidence_diff = total_confidence_pos - total_confidence_neg
            with col2:
                pass


            with col3:
               
                st.markdown('<h2 style="color: White; text-align: center;"> Article Sentiment Analysis Count</h2>', unsafe_allow_html=True)
                st.write("_________")

                with st.container(border=True):

                    st.markdown(f"<h3 style='color:green; text-align:center;'>Positive: </h3>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='color:green; text-align:center;'>{pos_count}</h1>", unsafe_allow_html=True)
                   
                with st.container(border=True):
                    st.markdown(f"<h3 style='color:grey; text-align:center;'>Neutral: </h3>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='color:grey; text-align:center;'>{neu_count}</h1>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(f"<h3 style='color:red; text-align:center;'>Negative: </h3>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='color:red; text-align:center;'>{neg_count}</h1>", unsafe_allow_html=True)
                st.write("_________")


                tot = pos_count + neu_count + neg_count
                if tot != 0:
                    pos_percent = pos_count / tot * 100
                    neu_percent = neu_count / tot * 100
                    neg_percent = neg_count / tot * 100
                else:
                    pos_percent = 0
                    neu_percent = 0
                    neg_percent = 0
           
                data = {'Sentiment': ['Positive', 'Neutral', 'Negative'],'Pourcentage': [pos_percent, neu_percent, neg_percent] }
                df = pd.DataFrame(data)

                colors = {'Negative': '#FF204E', 'Positive': '#0A6847', 'Neutral': '#808080'}

                fig = go.Figure(go.Pie(labels=df['Sentiment'],values=df['Pourcentage'],marker=dict(colors=[colors[sentiment] for sentiment in df['Sentiment']], line=dict(color='#000000', width=1)),textinfo='percent+label',hole=0.3))

                fig.update_layout(showlegend=True,height=550 )
                fig.update_traces(textposition='inside', insidetextfont=dict(size=23, color='white'))  # Increase text size

                st.plotly_chart(fig)



                color = 'green' if total_confidence_diff > 0 else 'red'
                formatted_diff = f'<span style="color:{color}">{total_confidence_diff:.4f}</span>'

                st.markdown('<h2 style="color: White; text-align: center;"> News Analyses Total Rate:</h2>', unsafe_allow_html=True)

                st.markdown (f'<h1 style="text-align:center;"> {formatted_diff}</h1>', unsafe_allow_html=True)

               
        main()


        st.markdown('<h2 style="color: White; text-align: center;"> Article Summarizer</h2>', unsafe_allow_html=True)

        url = st.text_input('', placeholder='Paste the URL of the article amd press Enter')

        def fetch_article_text(url):
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            title = None
            if soup.title and soup.title.string:
                title = soup.title.string
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                title = og_title.get('content')

            image_url = None
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                image_url = og_image.get('content')

            paragraphs = [p.get_text(separator=' ', strip=True) for p in soup.find_all('p')]
            text = '\n\n'.join([p for p in paragraphs if len(p) > 50])
            if not text:
                raise ValueError('Unable to extract article body from URL.')
            return title, image_url, text

        def simple_summary(text, sentence_count=5):
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            if len(sentences) <= sentence_count:
                return text
            return ' '.join(sentences[:sentence_count])

        if url:
            try:
                if NEWSPAPER_AVAILABLE:
                    article = newspaper.Article(url)
                    article.download()
                    article.parse()
                    img = article.top_image
                    title = article.title
                    authors = article.authors
                    article.nlp()
                    keywords = article.keywords
                    full_text = article.text
                    summary = article.summary
                else:
                    title, img, full_text = fetch_article_text(url)
                    authors = []
                    keywords = []
                    summary = simple_summary(full_text)

                if img:
                    st.image(img)
                if title:
                    st.subheader(title)
                if authors:
                    st.text(','.join(authors))

                st.subheader('Keywords:')
                st.write(', '.join(keywords) if keywords else 'No keywords available.')

                tab1, tab2 = st.tabs(["Full Text", "Summary"])
                with tab1:
                    txt = full_text.replace('Advertisement', '')
                    st.write(txt)
                with tab2:
                    st.subheader('Summary')
                    st.write(summary.replace('Advertisement', ''))
            except Exception as exc:
                st.error(f"Sorry, something went wrong: {exc}")




with tabs[3]:



    # Set app title
    st.markdown('<h1 style="color: white; text-align: center;">Portfolio Optimization</h1>', unsafe_allow_html=True)
    st.write("__________________")


    # Stock selection with two select boxes
    available_stocks = ["AAPL", "AMZN", "TSLA", "GOOGL","META", "MSFT", "GS", "V", "AXP", "DIS", "WMT", "HD", "NKE", "MCD", "XOM", "CVX", "KO", "PG", "VZ", "JPM"]
    col1, col2, col3 = st.columns([2, 3,3])

    with col1:
        portfolio = st.radio('Select Portfolio Number of Stocks to Compare', ("2", "3", "4"), index=2)

    with col2:
        selected_stock1 = st.selectbox('Select Stock 1', available_stocks)
        selected_stock2 = st.selectbox('Select Stock 2', [stock for stock in available_stocks if stock != selected_stock1])

    if portfolio == "3":
        with col3:
            selected_stock3 = st.selectbox('Select Stock 3', [stock for stock in available_stocks if stock not in [selected_stock1, selected_stock2]])

    if portfolio == "4":
        with col3:
            selected_stock3 = st.selectbox('Select Stock 3', [stock for stock in available_stocks if stock not in [selected_stock1, selected_stock2]])
        with col3:
            selected_stock4 = st.selectbox('Select Stock 4', [stock for stock in available_stocks if stock not in [selected_stock1, selected_stock2, selected_stock3]])


    # Period and interval selection
    period = {"1d": "1 Day", "5d": "5 Days", "1mo": "1 Month", "3mo": "3 Months", "6mo": "6 Months", "YTD": "Year to date", "1y": "1 Year", "2y": "2 Years", "5y": "5 Years", "10y": "10 years"}
    interval_short = {"1m": "1 Minute", "2m": "2 Minutes", "5m": "5 Minutes", "15m": "15 Minutes", "30m": "30 Minutes", "1h": "1 Hour"}
    interval_long = {"1d": "1 Day", "5d": "5 Days", "1wk": "1 Week", "1mo": "1 Month"}

    col1, col2 = st.columns([1, 1])

    with col1:
        select_period = st.select_slider("Select period", options=list(period.keys()), format_func=lambda x: period[x], value="YTD")
    with col2:
        if select_period in ["1d", "5d"]:
            select_interval = st.selectbox("Select interval", options=interval_short.keys(), format_func=lambda x: interval_short[x])
        elif select_period in ["1mo", "3mo", "6mo", "YTD", "1y", "2y", "5y", "10y"]:
            select_interval = st.selectbox("Select interval", options=interval_long.keys(), format_func=lambda x: interval_long[x], key="unique_key_for_this_selectbox")

    if select_interval in ["1m", "2m", "5m", "15m", "30m", "1h"]:
        dt = "Datetime"
    elif select_interval in ["1d", "5d", "1wk", "1mo"]:
        dt = "Date"

    # Fetch historical data for selected stocks
    def get_data(symbol, period, interval):
        data = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        return flatten_yf_data(data)

    stock1_data = get_data(selected_stock1, select_period, select_interval)
    stock2_data = get_data(selected_stock2, select_period, select_interval)
    if portfolio == "3":
        stock3_data = get_data(selected_stock3, select_period, select_interval)
    if portfolio == "4":
        stock3_data = get_data(selected_stock3, select_period, select_interval)
        stock4_data = get_data(selected_stock4, select_period, select_interval)

    ticker1 = yf.Ticker(selected_stock1)
    ticker2 = yf.Ticker(selected_stock2)
    if portfolio == "3":
        ticker3 = yf.Ticker(selected_stock3)
    if portfolio == "4":
        ticker3 = yf.Ticker(selected_stock3)
        ticker4 = yf.Ticker(selected_stock4)


    details1 = ticker1.info
    details2 = ticker2.info
    if portfolio == "3":
        details3 = ticker3.info
    if portfolio == "4":
        details3 = ticker3.info
        details4 = ticker4.info

    col1, col2, col3, col4 = st.columns([1, 1,1,1])
    with col1 :
        st.markdown(f"<center><h2>  {details1['longName']}</h2></center>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<center><h2>  {details2['longName']}</h2></center>", unsafe_allow_html=True)
    if portfolio == "3":
        with col3:
            st.markdown(f"<center><h2>  {details3['longName']}</h2></center>", unsafe_allow_html=True)
    if portfolio == "4":
        with col3:
            st.markdown(f"<center><h2>  {details3['longName']}</h2></center>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<center><h2>  {details4['longName']}</h2></center>", unsafe_allow_html=True)


    # Plotting time series data
    st.markdown('<h2 style="color: white; text-align: center;">Time Series Comparison </h2>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock1_data.index, y=stock1_data['Close'], name=selected_stock1,line=dict(color='blue')))

    if 'stock2_data' in locals():
        fig.add_trace(go.Scatter(x=stock2_data.index, y=stock2_data['Close'], name=selected_stock2,line=dict(color='green')))

    if 'stock3_data' in locals():
        fig.add_trace(go.Scatter(x=stock3_data.index, y=stock3_data['Close'], name=selected_stock3,line=dict(color='yellow')))

    if 'stock4_data' in locals():
        fig.add_trace(go.Scatter(x=stock4_data.index, y=stock4_data['Close'], name=selected_stock4,line=dict(color='red')))

    # Update layout
    fig.update_layout(xaxis_title=dt, yaxis_title='Stock Price',height=600, legend=dict(orientation="h"))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("<h1 style='text-align: center; color: white;'>DESCRIPTIVE STATISTICS</h1>", unsafe_allow_html=True)
    col1, col2,= st.columns([3, 3])
    with col1:
        descriptive_tables = []
        if 'stock1_data' in locals():
            descriptive_tables.append(stock1_data['Close'].describe().rename(selected_stock1))
        if 'stock2_data' in locals():
            descriptive_tables.append(stock2_data['Close'].describe().rename(selected_stock2))
        if 'stock3_data' in locals():
            descriptive_tables.append(stock3_data['Close'].describe().rename(selected_stock3))
        if 'stock4_data' in locals():
            descriptive_tables.append(stock4_data['Close'].describe().rename(selected_stock4))
        descriptive_table = pd.concat(descriptive_tables, axis=1)
        trans_table = descriptive_table.T
        st.table(trans_table)
    with col2:
        trans_table['Coefficient of variation'] = trans_table['std'] / trans_table['mean']
        st.table(trans_table[['Coefficient of variation']])

    fig_1 = go.Figure()
    fig_2 = go.Figure()
    fig_3 = go.Figure()
    fig_4 = go.Figure()

    stock1_data['Close_pct_change'] = stock1_data['Close'].pct_change() * 100
    stock2_data['Close_pct_change'] = stock2_data['Close'].pct_change() * 100
    st.markdown('<h3 style="color: white; text-align: center;">Stock Prices Changes Over Time</h3>', unsafe_allow_html=True)

    col1,col2=st.columns(2)
    with col1:
        st.markdown(f"<h3 style='text-align: center;'>{selected_stock1}</h3>", unsafe_allow_html=True)
        fig_1.add_trace(go.Scatter(x=stock1_data.index, y=stock1_data['Close_pct_change'], name=selected_stock1, line=dict(color='blue', width=2)))
        fig_1.update_layout( yaxis_title='Percentage change')
        st.plotly_chart(fig_1,use_container_width=True)
       
        st.markdown(f"<h3 style='text-align: center;'>{selected_stock2}</h3>", unsafe_allow_html=True)
        fig_2.add_trace(go.Scatter(x=stock2_data.index, y=stock2_data['Close_pct_change'], name=selected_stock2 ,line=dict(color='green', width=2)))
        fig_2.update_layout(yaxis_title='Percentage change')
        st.plotly_chart(fig_2,use_container_width=True)
    with col2:
        if 'stock3_data' in locals():
            stock3_data['Close_pct_change'] = stock3_data['Close'].pct_change() * 100
            st.markdown(f"<h3 style='text-align: center;'>{selected_stock3}</h3>", unsafe_allow_html=True)
            fig_3.add_trace(go.Scatter(x=stock3_data.index, y=stock3_data['Close_pct_change'], name=selected_stock3, line=dict(color='yellow', width=2)))
            fig_3.update_layout(yaxis_title='Percentage change')
            st.plotly_chart(fig_3,use_container_width=True)
        if 'stock4_data' in locals():
            stock4_data['Close_pct_change'] = stock4_data['Close'].pct_change() * 100
            st.markdown(f"<h3 style='text-align: center;'>{selected_stock4}</h3>", unsafe_allow_html=True)

            fig_4.add_trace(go.Scatter(x=stock4_data.index, y=stock4_data['Close_pct_change'], name=selected_stock4, line=dict(color='red', width=2)))
            fig_4.update_layout(yaxis_title='Percentage change')

            st.plotly_chart(fig_4,use_container_width=True)

    st.markdown('<h3 style="color: white; text-align: center;">Box Plots</h3>', unsafe_allow_html=True)
    col1, col2 ,col3,col4= st.columns(4)
    with col1:
        st.markdown(f"<h3 style='text-align: center;'>{selected_stock1}</h3>", unsafe_allow_html=True)
        fig_box_1 = px.box(stock1_data, x=[selected_stock1]*len(stock1_data['Close']), y=stock1_data['Close'], points="all", color_discrete_sequence=['blue'])
        st.plotly_chart(fig_box_1, use_container_width=True)
    with col2:
        st.markdown(f"<h3 style='text-align: center;'>{selected_stock2}</h3>", unsafe_allow_html=True)
        fig_box_2 = px.box(stock2_data, x=[selected_stock2]*len(stock2_data['Close']), y=stock2_data['Close'], points="all", color_discrete_sequence=['green'])
        st.plotly_chart(fig_box_2, use_container_width=True)
    with col3:
        if 'stock3_data' in locals():
            stock3_data['Close_pct_change'] = stock3_data['Close'].pct_change() * 100
            st.markdown(f"<h3 style='text-align: center;'>{selected_stock3}</h3>", unsafe_allow_html=True)
            fig_box_3 = px.box(stock3_data, x=[selected_stock3]*len(stock3_data['Close']), y=stock3_data['Close'], points="all", color_discrete_sequence=['yellow'])
            st.plotly_chart(fig_box_3, use_container_width=True)
    with col4:
        if 'stock4_data' in locals():
            stock4_data['Close_pct_change'] = stock4_data['Close'].pct_change() * 100
            st.markdown(f"<h3 style='text-align: center;'>{selected_stock4}</h3>", unsafe_allow_html=True)
            fig_box_4 = px.box(stock4_data, x=[selected_stock4]*len(stock4_data['Close']), y=stock4_data['Close'], points="all", color_discrete_sequence=['red'])
            st.plotly_chart(fig_box_4, use_container_width=True)

    st.markdown('<h3 style="color: white; text-align: center;">Box log Plots per percentage</h3>', unsafe_allow_html=True)
    col1, col2 ,col3,col4= st.columns(4)
    with col1:
        st.markdown(f"<h3 style='text-align: center;'>{selected_stock1}</h3>", unsafe_allow_html=True)
        fig_box_1 = px.box(stock1_data, x=[selected_stock1]*len(stock1_data['Close']), y=np.log(1 + stock1_data['Close'].pct_change()), points="all", color_discrete_sequence=['blue'])
        st.plotly_chart(fig_box_1, use_container_width=True)
    with col2:
        st.markdown(f"<h3 style='text-align: center;'>{selected_stock2}</h3>", unsafe_allow_html=True)
        fig_box_2 = px.box(stock2_data, x=[selected_stock2]*len(stock2_data['Close']), y=np.log(1 + stock2_data['Close'].pct_change()), points="all", color_discrete_sequence=['green'])
        st.plotly_chart(fig_box_2, use_container_width=True)
    with col3:
        if 'stock3_data' in locals():
            stock3_data['Close_pct_change'] = stock3_data['Close'].pct_change() * 100
            st.markdown(f"<h3 style='text-align: center;'>{selected_stock3}</h3>", unsafe_allow_html=True)
            fig_box_3 = px.box(stock3_data, x=[selected_stock3]*len(stock3_data['Close']), y=np.log(1 + stock3_data['Close'].pct_change()), points="all", color_discrete_sequence=['yellow'])
            st.plotly_chart(fig_box_3, use_container_width=True)
    with col4:
        if 'stock4_data' in locals():
            stock4_data['Close_pct_change'] = stock4_data['Close'].pct_change() * 100
            st.markdown(f"<h3 style='text-align: center;'>{selected_stock4}</h3>", unsafe_allow_html=True)
            fig_box_4 = px.box(stock4_data, x=[selected_stock4]*len(stock4_data['Close']), y=np.log(1 + stock4_data['Close'].pct_change()), points="all", color_discrete_sequence=['red'])
            st.plotly_chart(fig_box_4, use_container_width=True)
    col1 ,col2=st.columns(2)
    with col1:
        st.markdown('<h3 style="color: white; text-align: center;">Box Plots Comparison</h3>', unsafe_allow_html=True)      
        fig_combined_box = go.Figure()
        fig_combined_box.add_trace(go.Box(x=[selected_stock1]*len(stock1_data['Close']),y=stock1_data['Close'],name=selected_stock1,marker_color='blue'))

        fig_combined_box.add_trace(go.Box(x=[selected_stock2]*len(stock2_data['Close']),y=stock2_data['Close'],name=selected_stock2,marker_color='green'))

        if 'stock3_data' in locals():
            stock3_data['Close_pct_change'] = stock3_data['Close'].pct_change() * 100
            fig_combined_box.add_trace(go.Box(x=[selected_stock3]*len(stock3_data['Close']),y=stock3_data['Close'],name=selected_stock3,marker_color='yellow'))

        if 'stock4_data' in locals():
            stock4_data['Close_pct_change'] = stock4_data['Close'].pct_change() * 100
            fig_combined_box.add_trace(go.Box(x=[selected_stock4]*len(stock4_data['Close']),y=stock4_data['Close'],name=selected_stock4,marker_color='red'))

        fig_combined_box.update_layout(xaxis_title='Stocks', yaxis_title='Closing Price')
        st.plotly_chart(fig_combined_box, use_container_width=True)
    with col2:
        st.markdown('<h3 style="color: white; text-align: center;">Box log Plots per percentage Comparison</h3>', unsafe_allow_html=True)

        comb_box_2 = go.Figure()
        # Add traces for each selected stock
        comb_box_2.add_trace(go.Box(x=[selected_stock1]*len(stock1_data['Close']), y=np.log(1 + stock1_data['Close'].pct_change()), name=selected_stock1, marker_color='blue'))
        comb_box_2.add_trace(go.Box(x=[selected_stock2]*len(stock2_data['Close']),  y=np.log(1 + stock2_data['Close'].pct_change()), name=selected_stock2, marker_color='green'))

        if 'stock3_data' in locals():
            stock3_data['Close_pct_change'] = stock3_data['Close'].pct_change() * 100
            comb_box_2.add_trace(go.Box(x=[selected_stock3]*len(stock3_data['Close']),  y=np.log(1 + stock3_data['Close'].pct_change()), name=selected_stock3, marker_color='yellow'))

        if 'stock4_data' in locals():
            stock4_data['Close_pct_change'] = stock4_data['Close'].pct_change() * 100
            comb_box_2.add_trace(go.Box(x=[selected_stock4]*len(stock4_data['Close']), y=np.log(1 + stock4_data['Close'].pct_change()), name=selected_stock4, marker_color='red'))

        comb_box_2.update_layout(xaxis_title='Stocks', yaxis_title='Closing Price')
        st.plotly_chart(comb_box_2, use_container_width=True)
    # Create a correlation matrix

   
    st.markdown('<h3 style="color: white; text-align: center;">Histogram Plots</h3>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<h3 style='text-align: center;'>{selected_stock1}</h3>", unsafe_allow_html=True)
        fig_hist_1 = px.histogram(stock1_data, x=np.log(1 + stock1_data['Close'].pct_change()), color_discrete_sequence=['blue'])
        st.plotly_chart(fig_hist_1, use_container_width=True)
    with col2:
        st.markdown(f"<h3 style='text-align: center;'>{selected_stock2}</h3>", unsafe_allow_html=True)
        fig_hist_2 = px.histogram(stock2_data, x=np.log(1 + stock2_data['Close'].pct_change()), color_discrete_sequence=['green'])
        st.plotly_chart(fig_hist_2, use_container_width=True)
    with col3:
        if 'stock3_data' in locals():
            stock3_data['Close_pct_change'] = stock3_data['Close'].pct_change() * 100
            st.markdown(f"<h3 style='text-align: center;'>{selected_stock3}</h3>", unsafe_allow_html=True)
            fig_hist_3 = px.histogram(stock3_data, x=np.log(1 + stock3_data['Close'].pct_change()), color_discrete_sequence=['yellow'])
            st.plotly_chart(fig_hist_3, use_container_width=True)
    with col4:
        if 'stock4_data' in locals():
            stock4_data['Close_pct_change'] = stock4_data['Close'].pct_change() * 100
            st.markdown(f"<h3 style='text-align: center;'>{selected_stock4}</h3>", unsafe_allow_html=True)
            fig_hist_4 = px.histogram(stock4_data, x=np.log(1 + stock4_data['Close'].pct_change()), color_discrete_sequence=['red'])
            st.plotly_chart(fig_hist_4, use_container_width=True)
    st.markdown('<h2 style="color: white; text-align: center;">Correlation </h2>', unsafe_allow_html=True)

    if 'stock4_data' in locals():
        all_stock_data = pd.concat([
            stock1_data['Close_pct_change'].rename(selected_stock1),
            stock2_data['Close_pct_change'].rename(selected_stock2),
            stock3_data['Close_pct_change'].rename(selected_stock3),
            stock4_data['Close_pct_change'].rename(selected_stock4)], axis=1)
    elif 'stock3_data' in locals():
        all_stock_data = pd.concat([
            stock1_data['Close_pct_change'].rename(selected_stock1),
            stock2_data['Close_pct_change'].rename(selected_stock2),
            stock3_data['Close_pct_change'].rename(selected_stock3)], axis=1)
    else:
        all_stock_data = pd.concat([
            stock1_data['Close_pct_change'].rename(selected_stock1),
            stock2_data['Close_pct_change'].rename(selected_stock2)], axis=1)

   
   
   
   
    correlation_matrix = all_stock_data.corr()


    col1, col2 = st.columns(2)

    with col1:
           
        st.markdown('<h3 style="color: white; text-align: center;">Pair plot </h3>', unsafe_allow_html=True)
       
        pairplot = px.scatter_matrix(all_stock_data)
        pairplot.update_yaxes(autorange="reversed")  
        st.plotly_chart(pairplot)

        correlation_matrix = all_stock_data.corr()
        # st.write(all_stock_data)
       
    with col2:
        st.markdown('<h3 style="color: white; text-align: center;">Matrix </h3>', unsafe_allow_html=True)
        fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values,x=correlation_matrix.columns,y=correlation_matrix.index,colorscale='Viridis'))
        fig.update_layout(xaxis_title='Stocks', yaxis_title='Stocks', yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig)


    def portfolio_optimization(all_stock_data):
        np.random.seed(0)
        nb_portfolio = 10000
        all_weights = np.zeros((nb_portfolio, len(all_stock_data.columns)))
        ret_arr = np.zeros(nb_portfolio)
        vol_arr = np.zeros(nb_portfolio)
        sharpe_arr = np.zeros(nb_portfolio)
        log_return = np.log(1 + all_stock_data.pct_change())

        for ind in range(nb_portfolio):
            weights = np.array(np.random.random(len(all_stock_data.columns)))
            weights = weights / np.sum(weights)
            all_weights[ind, :] = weights.round(3)
            ret_arr[ind] = np.sum((log_return.mean() * weights) * 252)
            vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_return.cov() * 252, weights)))
            sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]

        simulations_data = [ret_arr, vol_arr, sharpe_arr, all_weights]
        simulations_df = pd.DataFrame(data=simulations_data).T
        simulations_df.columns = ['Returns', 'Volatility', 'Sharpe Ratio', 'Portfolio Weights']
        simulations_df = simulations_df.infer_objects()
        return simulations_df

    # all_stock_data = all_stock_data
    st.markdown('<h2 style="color: white; text-align: center;">SIMULATIONS RESULT</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        simulations_df = portfolio_optimization(all_stock_data)
        st.write(simulations_df)

    with col2:
        mean_return = simulations_df['Returns'].mean()
        mean_volatility = simulations_df['Volatility'].mean()
        mean_sharperatio = simulations_df['Sharpe Ratio'].mean()
        mean_weights = simulations_df['Portfolio Weights'].mean()

        mean_values = pd.DataFrame({'Metrics': ['Returns', 'Volatility', 'Sharpe Ratio'],'Mean Values': [mean_return, mean_volatility, mean_sharperatio]})
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=mean_values['Mean Values'],theta=mean_values['Metrics'],fill='toself',name='Mean Metrics'))

        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, max(mean_values['Mean Values'])])))

        st.plotly_chart(fig,use_container_width=True)    
    with col3:

        sim_cor = simulations_df[['Returns', 'Volatility', 'Sharpe Ratio']].corr()
        st.table(sim_cor)

        input_number = st.number_input("Initial investment", min_value=0.0, step=1.0)

        if st.button('Calculate'):
            gains = input_number * mean_return / 100
            total_amount = gains + input_number

            # Displaying the gains and total investment after gaining
            st.write(f"Portfolio Gains: {round(gains, 3)}")
            st.write(f"Total investment after gaining: {round(total_amount, 3)}")


       
    # Calculate maximum Sharpe ratio and optimal allocation
    max_sharpe_idx = simulations_df['Sharpe Ratio'].idxmax()
    max_sharpe_ratio = simulations_df.loc[max_sharpe_idx]
    alloc_max_sr = max_sharpe_ratio['Portfolio Weights']
    max_sr_stocks = all_stock_data.columns[np.nonzero(alloc_max_sr)]

    # Calculate minimum volatility and optimal allocation
    min_volatility_idx = simulations_df['Volatility'].idxmin()
    min_volatility = simulations_df.loc[min_volatility_idx]
    alloc_min_vol = min_volatility['Portfolio Weights']
    min_vol_stocks = all_stock_data.columns[np.nonzero(alloc_min_vol)]


    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('MAX SHARPE RATIO')
        # st.write(max_sharpe_ratio[['Returns', 'Volatility', 'Sharpe Ratio']])
        selected_columns = max_sharpe_ratio[['Returns', 'Volatility', 'Sharpe Ratio']]
        selected_columns_df = pd.DataFrame(selected_columns)
        selected_columns_df_tr = selected_columns_df.transpose()
        st.table(selected_columns_df)

    with col2:

        data = [go.Scatterpolar(r=selected_columns_df_tr['Returns'], theta=['Returns'] * len(selected_columns_df_tr),fill='tonext',name='Returns',marker=dict(size=10)),
                go.Scatterpolar(r=selected_columns_df_tr['Volatility'],theta=['Volatility'] * len(selected_columns_df_tr),fill='tonext',name='Volatility',marker=dict(size=10)),
                go.Scatterpolar(r=selected_columns_df_tr['Sharpe Ratio'],theta=['Sharpe Ratio'] * len(selected_columns_df_tr),fill='tonext',name='Sharpe Ratio',marker=dict(size=15))]
               

        layout = go.Layout(polar=dict(radialaxis=dict(visible=True)),legend=dict(visible=False))

        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig,use_container_width=True)

           
    with col3:
        st.write("Allocation optimal:")
        data_all = {"Stocks": max_sr_stocks, "Weights": alloc_max_sr[np.nonzero(alloc_max_sr)]}
        st.table(data_all)

    col1, col2,col3  = st.columns(3)
    with col1:
        st.write('MIN VOLATILITY')
        # st.write(min_volatility[['Returns', 'Volatility', 'Sharpe Ratio']])
        selected_columns = min_volatility[['Returns', 'Volatility', 'Sharpe Ratio']]
        selected_columns_df = pd.DataFrame(selected_columns)
        st.table(selected_columns_df)
        selected_columns_df_tr = selected_columns_df.transpose()
    with col2:

        data = [go.Scatterpolar(r=selected_columns_df_tr['Returns'], theta=['Returns'] * len(selected_columns_df_tr),fill='tonext',name='Returns',marker=dict(size=10)),
                go.Scatterpolar(r=selected_columns_df_tr['Volatility'],theta=['Volatility'] * len(selected_columns_df_tr),fill='tonext',name='Volatility',marker=dict(size=15)),
                go.Scatterpolar(r=selected_columns_df_tr['Sharpe Ratio'],theta=['Sharpe Ratio'] * len(selected_columns_df_tr),fill='tonext',name='Sharpe Ratio',marker=dict(size=10))]

        layout = go.Layout(polar=dict(radialaxis=dict(visible=True)),legend=dict(visible=False))

        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig,use_container_width=True)
    with col3:
        st.write("Allocation optimal:")
        data_op = {"Stocks": min_vol_stocks, "Weights": alloc_min_vol[np.nonzero(alloc_min_vol)]}
        st.table(data_op)



    fig = px.scatter(simulations_df,x='Volatility',y='Returns',color='Sharpe Ratio',color_continuous_scale='plasma',labels={'Volatility':'Volatility','Returns':'Return'})
    fig.update_layout(coloraxis_colorbar=dict(title='Sharpe Ratio'),xaxis_title='Volatility',yaxis_title='Return',title='Sharpe Ratio Analysis')

    max_sr_ret = simulations_df['Returns'][simulations_df['Sharpe Ratio'].idxmax()]
    max_sr_vol = simulations_df['Volatility'][simulations_df['Sharpe Ratio'].idxmax()]
    min_vol_vol = simulations_df['Volatility'][simulations_df['Volatility'].idxmin()]
    min_vol_ret = simulations_df['Returns'][simulations_df['Volatility'].idxmin()]

    fig.add_trace(go.Scatter(x=[max_sr_vol], y=[max_sr_ret], mode='markers', marker=dict(color='white', size=30),name='Maximum Sharpe ratio'))
    fig.add_trace(go.Scatter(x=[min_vol_vol], y=[min_vol_ret], mode='markers', marker=dict(color='green', size=30),name='Minimum volatility'))

    fig.update_layout(height=900, legend=dict(orientation="h"))

    st.plotly_chart(fig, use_container_width=True)