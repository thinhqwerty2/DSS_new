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
from sklearn.metrics import mean_absolute_percentage_error as mpae

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
chosen_product = st.selectbox('Chọn sản phẩm', stv['item_id'],index=2314)
dataset = stv.query(f'item_id=="{chosen_product}"')
train_dataset = dataset[d_cols[-1900:-30]].squeeze()
val_dataset = dataset[d_cols[-30:]].squeeze()
tab0, tab1, tab2, tab3 = st.tabs(["Baseline", "Moving average", "ARIMA", "So sánh"])
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
        mpae_base = mpae(val_dataset,pred_1)

        fig0 = make_subplots(rows=1, cols=1)

        fig0.add_trace(
            go.Scatter(x=np.arange(70), mode='lines', y=train_dataset[-70:], marker=dict(color="dodgerblue"),
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
        st.write(pred_1)
        error_avg = mae(val_dataset, pred_1)
        mpae_avg = mpae(val_dataset,pred_1)
        fig1 = make_subplots(rows=1, cols=1)

        fig1.add_trace(
            go.Scatter(x=np.arange(70), mode='lines', y=train_dataset[-70:], marker=dict(color="dodgerblue"),
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

        from statsmodels.tsa.stattools import adfuller
        from statsmodels.graphics.tsaplots import plot_acf
        from statsmodels.graphics.tsaplots import plot_pacf


        def diff_series_and_check(new_train_data,count = 0):
            # Assuming 'data' is your time series data
            result = adfuller(new_train_data)

            # Extract the test statistics and p-value
            p_value = result[1]
            st.write('ADF Statistic: %f' % result[0])
            st.write('p-value: %f' % result[1])
            st.write('Critical Values:')
            for key, value in result[4].items():
                st.write('\t%s: %.3f' % (key, value))
            # Compare the p-value to the significance level

            if p_value < 0.05:
                st.write("The time series is stationary.")
                return [new_train_data,count]
            else:
                st.write("The time series is non-stationary.")
                new_train_data = new_train_data.diff().dropna()
                rs=diff_series_and_check(new_train_data,count+1)
            return rs
        rs=diff_series_and_check(train_dataset)
        new_train_dataset = rs[0]
        diff = rs[1]
        st.write("Số lần sai phân",rs[1])



        import pandas as pd
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_pacf(new_train_dataset,alpha=None))
            p_param = int(st.text_input("Tham số p",value='1',key='p_param'))

        with col2:
            st.pyplot(plot_acf(new_train_dataset,alpha=None))
            q_param = int(st.text_input("Tham số q",value='1',key='q_param'))
        st.write(p_param,q_param)

        import statsmodels.api as sm

        predictions = []
        fit = sm.tsa.statespace.SARIMAX(train_dataset,order=(p_param,diff,q_param)).fit()
        # seasonal_order = (0, 1, 1, 7)
        predictions.append(fit.forecast(30))
        predictions = np.array(predictions).reshape((-1, 30))

        # model = auto_arima(train_dataset, seasonal=False)
        # predictions1 = model.predict(30)
        # pred_2=predictions1

        # st.write(model)
        pred_1 = predictions[0]
        error_arima = mae(val_dataset, pred_1)
        mpae_arima = mpae(val_dataset,pred_1)
    except Exception as e:
        st.write(e)
        pass
    finally:
        fig2 = make_subplots(rows=1, cols=1)
        fig2.add_trace(
            go.Scatter(x=np.arange(71), mode='lines', y=train_dataset[-70:], marker=dict(color="dodgerblue"),
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
        # fig2.add_trace(
        #     go.Scatter(x=np.arange(70, 100), y=pred_2, mode='lines',
        #                name="Pred"),
        #     row=1, col=1
        # )
        fig2.update_layout(height=400, width=800)
        st.plotly_chart(fig2)

with tab3:
    try:

        st.plotly_chart(fig0)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
        # error = [error_base, error_avg, error_arima]
        error = [mpae_base,mpae_avg,mpae_arima]
        names = ["Baseline", "Moving average", "ARIMA"]
        df = pd.DataFrame(np.transpose([error, names]))
        df.columns = ["MPAE", "Mô hình"]
        fig3 = px.bar(df, y="MPAE", x="Mô hình",color="Mô hình", title="MPAE của các mô hình")
        st.plotly_chart(fig3)

        # mpae_=[mpae_base,mpae_avg,mpae_arima]
        # st.write(error)


    except Exception as e:
        st.write(e)
        pass
