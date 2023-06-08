import logging
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error as mae

st.title("Dự báo số lượng sản phẩm bán ra")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Read in the data
INPUT_DIR = r'D:\Dss\data'
with open('sales.pkl', 'rb') as f:
    stv = pickle.load(f)
    d_cols = [c for c in stv.columns if 'd_' in c]

store_list = stv['store_id'].unique()
chosen_store = st.selectbox('Chọn cửa hàng', store_list)
stv = stv.query(f'store_id=="{chosen_store}"')
chosen_product = st.selectbox('Chọn sản phẩm', stv['item_id'])
dataset = stv.query(f'item_id=="{chosen_product}"')
train_dataset = dataset[d_cols[-300:-30]].squeeze()
val_dataset = dataset[d_cols[-30:]].squeeze()
tab0, tab1, tab2, tab3 = st.tabs(["Baseline", "Moving average", "SARIMAX", "So sánh"])
with tab0:
    try:
        predictions = []
        for i in range(len(val_dataset)):
            if i == 0:
                predictions.append(train_dataset[-1])
            else:
                predictions.append(val_dataset[i - 1])
        pred_1 = predictions
        error_base = mae(val_dataset, pred_1)
        fig0 = make_subplots(rows=1, cols=1)

        fig0.add_trace(
            go.Scatter(x=np.arange(70), mode='lines', y=train_dataset, marker=dict(color="dodgerblue"),
                       name="Train"),
            row=1, col=1
        )

        fig0.add_trace(
            go.Scatter(x=np.arange(70, 100), y=val_dataset, mode='lines', marker=dict(color="darkorange"),
                       name="Val"),
            row=1, col=1
        )

        fig0.add_trace(
            go.Scatter(x=np.arange(70, 100), y=pred_1, mode='lines', marker=dict(color="seagreen"),
                       name="Pred"),
            row=1, col=1
        )

        fig0.update_layout(height=400, width=800)
        st.plotly_chart(fig0)
    except:
        pass
with tab1:
    try:
        predictions = []
        for i in range(len(val_dataset)):
            if i == 0:
                predictions.append(np.mean(train_dataset[-30:]))
            if i < 31 and i > 0:
                predictions.append(0.5 * (np.mean(train_dataset[-30 + i:]) + np.mean(predictions[:i])))
            if i > 31:
                predictions.append(np.mean([predictions[:i]]))

        pred_1 = predictions
        error_avg = mae(val_dataset, pred_1)
        fig1 = make_subplots(rows=1, cols=1)

        fig1.add_trace(
            go.Scatter(x=np.arange(70), mode='lines', y=train_dataset, marker=dict(color="dodgerblue"),
                       name="Train"),
            row=1, col=1
        )

        fig1.add_trace(
            go.Scatter(x=np.arange(70, 100), y=val_dataset, mode='lines', marker=dict(color="darkorange"),
                       name="Val"),
            row=1, col=1
        )

        fig1.add_trace(
            go.Scatter(x=np.arange(70, 100), y=pred_1, mode='lines', marker=dict(color="seagreen"),
                       name="Pred"),
            row=1, col=1
        )

        fig1.update_layout(height=400, width=800)

        st.plotly_chart(fig1)
    except:
        pass

with tab2:
    try:
        import statsmodels.api as sm

        model = auto_arima(train_dataset, seasonal=True, m=7, start_p=0, start_q=0)
        predictions = []
        # fit = sm.tsa.statespace.SARIMAX(train_dataset, seasonal_order=(0, 1, 1, 7)).fit()
        # predictions.append(fit.forecast(30))
        # predictions = np.array(predictions).T
        predictions = model.predict(30)
        st.write(model)
        pred_1 = predictions
        error_sarimax = mae(val_dataset, pred_1)
        fig2 = make_subplots(rows=1, cols=1)
        fig2.add_trace(
            go.Scatter(x=np.arange(71), mode='lines', y=train_dataset, marker=dict(color="dodgerblue"),
                       name="Train"),
            row=1, col=1
        )

        fig2.add_trace(
            go.Scatter(x=np.arange(70, 100), y=val_dataset, mode='lines', marker=dict(color="darkorange"),
                       name="Val"),
            row=1, col=1
        )

        fig2.add_trace(
            go.Scatter(x=np.arange(70, 100), y=pred_1, mode='lines', marker=dict(color="seagreen"),
                       name="Pred"),
            row=1, col=1
        )
        fig2.update_layout(height=400, width=800)
        st.plotly_chart(fig2)

    except:
        pass
with tab3:
    try:
        fig3 = make_subplots(rows=3, cols=1)
        fig3.add_trace(fig0)
        fig3.add_trace(fig1)
        fig3.add_trace(fig2)
        fig3.update_layout(height=1200, width=800)
        st.plotly_chart(fig3)
        error = [error_base, error_avg, error_sarimax]
        names = ["Baseline", "Moving average", "SARIMAX"]
        df = pd.DataFrame(np.transpose([error, names]))
        df.columns = ["RMSE Loss", "Model"]
        fig3=px.bar(df, y="MAE", x="Mô hình", title="MAE của các mô hình")
        st.plotly_chart(fig3)
    except Exception as e:
        logging.info(e)
        pass
