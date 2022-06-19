import pandas as pd
import yfinance as yf
import streamlit as st
import PIL.Image
import datetime
from datetime import datetime,date, time, timedelta

from prophet import Prophet
import matplotlib.pyplot as plt
import base64

from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric

st.title("Lotus Finans AI")
st.write("###### Finansal piyasaların yapay zeka analizleri sizi başarıya bir adım daha yaklaştıracaktır.")

fp = open("newlogo.png","rb")
logo = PIL.Image.open(fp)
st.sidebar.image(logo,use_column_width=True)

st.sidebar.title("Filteler")

islemturu=st.sidebar.radio("İşlem Türü",["Kriptopara","Emtia","Hisse Senedi"])

if islemturu == "Kriptopara":
    kriptosec=st.sidebar.selectbox("Kripto Para Cinsi", ["Btc","ETH","XRP","DODGE","AVAX","BNB"])
    kriptosec=kriptosec+"-USD"
    sembol=kriptosec

elif islemturu == "Emtia":
    emitasec = st.sidebar.selectbox("Emitalar", ["XAU"])
    emitas={
        "XAU":"GC=F",
    }

    emtiasec=emitas[emitasec]
    sembol=emtiasec

else:
    borsasec=st.sidebar.selectbox("Hisse Senetleri",["ASELSAN","THY","GARANTİ","AKBANK","BJK","DEVA","MNDRS","MRSHL","TTRAK","CCOLA","VESTL",])
    senetler={
        "ASELSAN":"ASELS.IS",
        "THY":"THYAO.IS",
        "GARANTİ":"GARAN.IS",
        "AKBANK":"AKBNK.IS",
        "BJK":"BJKAS.IS",
        "DEVA":"DEVA.IS",
        "MNDRS":"MNDRS.IS",
        "MRSHL":"MRSHL.IS",
        "TTRAK":"TTRAK.IS",
        "CCOLA":"CCOLA.IS",
        "VESTL":"VESTL.IS",
    }

    hissesec=senetler[borsasec]
    sembol=hissesec

zaralik=range(1,721)
slider=st.sidebar.select_slider("Zaman Aralığı",options=zaralik,value=30)

bugun=datetime.today()
aralik=timedelta(days=slider)

st.sidebar.write("## Tarih Aralığı")
baslangic=st.sidebar.date_input("Baslangıç Tarihi",bugun-aralik)
bitis=st.sidebar.date_input("Bitis Tarihi",value=bugun)

st.sidebar.write("## Makine Öğrenmesi Tahmini")

prophet=st.sidebar.checkbox("Facebook Prophet")


if prophet:
    fbaralik=range(1,1441)
    fbperiyot=st.sidebar.select_slider("Periyot", options=zaralik, value=30)
    components=st.sidebar.checkbox("Components")


if prophet:
    cvsec=st.sidebar.checkbox("CV")
    if cvsec:
        st.sidebar.write("## Metrik Seçiniz")
        metric=st.sidebar.radio("Metrik",["rmse","mse","mape","mdape"])
        
        st.sidebar.write("## Parametre Seçiniz")
        inaralik=range(1,1441)
        cvin=st.sidebar.select_slider("Initial", options=inaralik, value=120)
        peraralik=range(1,1441)
        cvper=st.sidebar.select_slider("CV Periyot", options=peraralik, value=120)
        horaralik=range(1,1441)
        cvhor=st.sidebar.select_slider("Horizon", options=horaralik, value=60)
        

        


def grafikgetir(sembol,baslangic,bitis):
        data=yf.Ticker(sembol)
        global df
        df=data.history(period="1d",start=baslangic,end=bitis)
        st.line_chart(df["Close"])

        if prophet:
            fb = df.reset_index()
            fb = fb[["Date", "Close"]]
            fb.columns = ["ds", "y"]
            global model
            model = Prophet()
            
            model.fit(fb)
            future = model.make_future_dataframe(periods=fbperiyot)
            predict = model.predict(future)
            grap=model.plot(predict)
            st.write(grap)
         
            
            if components:
                grap2=model.plot_components(predict)
                st.write(grap2)
            
        else:
            pass


def cvgrafik(model,initial,period,horizon,metric):
    initial = str(initial)+" days"
    period = str(period)+" days"
    horizon = str(horizon)+" days"
    cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
    grap3 = plot_cross_validation_metric(cv,metric=metric)
    st.write(grap3)
    

grafikgetir(sembol,baslangic,bitis)

if prophet:
    if cvsec:
        cvgrafik(model,cvin,cvper,cvhor,metric)


#pandas kütüphanemizdeki methodumuz ile tabloyu csv formatına çevirdim, dosya olarak çekme işlemimiz.
def indir(df):
    csv=df.to_csv()
    b64=base64.b64encode(csv.encode()).decode() 
    href=f'<a href="data:file/csv;base64,{b64}">CSV Formatında İndir</a>'
    return href


st.markdown(indir(df),unsafe_allow_html=True)


#finansal indikatörler
def SMA(data,period=30,column='Close'):
    return data[column].rolling(window=period).mean()


def EMA(data,period=21,column='Close'):
    return data[column].ewm(span=period,adjust=False).mean()


def MACD(data,period_long=26,period_short=12,period_signal=9,column='Close'):
    ShortEMA=EMA(data,period_short,column=column)
    LongEMA=EMA(data,period_long,column=column)
    data["MACD"]=ShortEMA-LongEMA
    data["Signal_Line"]=EMA(data,period_signal,column="MACD")
    
    return data


def RSI(data,period=14,column='Close'):
    delta=data[column].diff(1)
    delta=delta[1:]
    up=delta.copy()
    down=delta.copy()
    up[up<0]=0
    down[down>0]=0
    data["up"]=up
    data["down"]=down
    AVG_Gain=SMA(data,period,column='up')
    AVG_Loss=abs(SMA(data,period,column='down'))
    RS=AVG_Gain/AVG_Loss
    RSI=100.0-(100.0/(1.0+RS))
    data["RSI"]=RSI
    return data


st.sidebar.write("## Finansal İndikatörler")
fi=st.sidebar.checkbox("Finansal İndikatörler")

def filer():
    if fi:
        fimacd=st.sidebar.checkbox("MACD")
        firsi=st.sidebar.checkbox("RSI")
        fisl=st.sidebar.checkbox("Signal Line")
        
        if fimacd:
            macd=MACD(df)
            st.line_chart(macd["MACD"])
        if firsi:
            rsi=RSI(df)
            st.line_chart(rsi["RSI"])
        if fisl:
            macd=MACD(df)
            st.line_chart(macd["Signal_Line"])
filer()
    
    
    
    
    
    