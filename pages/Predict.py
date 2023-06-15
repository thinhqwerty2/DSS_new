import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit
import streamlit as st
from plotly.subplots import make_subplots
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

st.title("Dự báo số lượng sản phẩm bán ra")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Read in the data
INPUT_DIR = r'D:\Dss\data'
with open('sales.pkl', 'rb') as f:
    stv = pickle.load(f)
    d_cols = [c for c in stv.columns if 'd_' in c]

store_list = stv['store_id'].unique()
chosen_store = st.selectbox('Chọn cửa hàng', store_list,index=2)
stv = stv.query(f'store_id=="{chosen_store}"')
chosen_product = st.selectbox('Chọn sản phẩm', stv['item_id'], index=2314)
dataset = stv.query(f'item_id=="{chosen_product}"')
train_dataset = dataset[d_cols[-365*2:-30]].squeeze()
val_dataset = dataset[d_cols[-30:]].squeeze()
tab0, tab1, tab2, tab3 = st.tabs(["Baseline", "Moving average", "ARIMA", "Đánh giá"])
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
        mape_base = mape(val_dataset, pred_1)

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
        width = st.slider(
            "Chọn độ rộng cửa sổ trượt", min_value=1, max_value=100, value=30, key='ma')
        predictions = []
        for i in range(len(val_dataset)):
            if i == 0:
                predictions.append(np.mean(train_dataset[-width:]))
            if i < width and i > 0:
                predictions.append(1 / width * (np.sum(train_dataset[-width + i:]) + np.sum(predictions[:i])))
            if i >= width:
                predictions.append(np.mean([predictions[-width:]]))

        pred_1 = predictions
        error_avg = mae(val_dataset, pred_1)
        mape_avg = mape(val_dataset, pred_1)
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
    except Exception as e:
        st.write(e)
        pass

with tab2:
    try:

        from statsmodels.tsa.stattools import adfuller
        from statsmodels.graphics.tsaplots import plot_acf
        from statsmodels.graphics.tsaplots import plot_pacf

        fig = make_subplots(1, 1)
        st.plotly_chart(fig.add_trace(
            go.Scatter(y=train_dataset, mode='lines', ),
            row=1, col=1
        ))


        def diff_series_and_check(new_train_data, count=0):
            # Assuming 'data' is your time series data
            result = adfuller(new_train_data)

            # Extract the test statistics and p-value

            p_value = result[1]
            # st.write('ADF Statistic: %f' % result[0])
            # st.write('p-value:' , result[1])
            # st.write('Critical Values:')
            # for key, value in result[4].items():
            #     st.write('\t%s: %.3f' % (key, value))
            # Compare the p-value to the significance level

            if p_value < 0.05:
                st.write(f"Chuỗi thời gian dừng với {count} lần sai phân")
                return [new_train_data, count]
            else:
                new_train_data = new_train_data.diff().dropna()
                rs = diff_series_and_check(new_train_data, count + 1)
            return rs


        def diff_seasonal_and_check(new_train_data, lag=1, count=0):
            # Assuming 'data' is your time series data
            count += 1
            new_train_data = new_train_data.diff(lag).dropna()

            result = adfuller(new_train_data)

            # Extract the test statistics and p-value

            p_value = result[1]
            # st.write('ADF Statistic: %f' % result[0])
            # st.write('p-value:', result[1])
            # st.write('Critical Values:')
            # for key, value in result[4].items():
            #     st.write('\t%s: %.3f' % (key, value))
            # Compare the p-value to the significance level

            if p_value < 0.05:
                return [new_train_data, count]
            else:
                rs = diff_series_and_check(new_train_data, lag, count)
            return rs


        rs = diff_series_and_check(train_dataset)
        new_train_dataset = rs[0]
        diff = rs[1]
        # st.write("Số lần sai phân d = ", rs[1])

        import pandas as pd
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        lag = st.slider(
            "Chọn trễ", min_value=1, max_value=100, value=30, key='lag1')
        col1, col2 = st.columns(2)
        with col2:
            st.pyplot(plot_pacf(new_train_dataset, alpha=0.05, lags=lag))

        with col1:
            st.pyplot(plot_acf(new_train_dataset, alpha=0.05, lags=lag))

        # fig = make_subplots(1,1)
        # fig.add_trace(
        #     go.Scatter( mode='lines', y=train_dataset.diff(7).dropna(), marker=dict(color="dodgerblue"),
        #                name="Train"),
        #     row=1, col=1
        # )

        P_param = D_param = Q_param = 0
        s_param = 7
        p_col, d_col, q_col, s_col = st.columns(4)
        with s_col:
            seasonal = st.checkbox('Sử dụng SARIMA', value=False)
            if seasonal:
                s_param = int(st.text_input("Tham số s", value='7', key='S_param'))
        with p_col:
            p_param = int(st.text_input("Tham số p", value='1', key='p_param'))
            if seasonal:
                P_param = int(st.text_input("Tham số P", value='0', key='P_param'))

        if seasonal:
            rs = diff_seasonal_and_check(train_dataset, lag=s_param)
            new_train_dataset = rs[0]
            Diff = rs[1]


        with d_col:
            d_param = int(st.text_input("Tham số d", value=diff, key='d_param'))
            if seasonal:
                D_param = int(st.text_input("Tham số D", value=Diff, key='D_param'))
        with q_col:
            q_param = int(st.text_input("Tham số q", value='1', key='q_param'))
            if seasonal:
                Q_param = int(st.text_input("Tham số Q", value='0', key='Q_param'))

        try:
            if seasonal:
                with col2:
                    st.write(f'D = {rs[1]}')
                    st.pyplot(plot_pacf(new_train_dataset, alpha=0.05, lags=lag, title=f'Diff({s_param}) pacf'))

                with col1:
                    st.write(f"Số lần sai phân với chu kỳ {s_param} là ", )
                    # lag2 = st.slider(
                    #     "Chọn khoảng thời gian", min_value=1, max_value=100, value=50,key='lag2')
                    st.pyplot(plot_acf(new_train_dataset, alpha=0.05, lags=lag, title=f'Diff({s_param}) acf'))
        except:
            pass

        # st.plotly_chart(fig)

        import statsmodels.api as sm

        predictions = []
        fit = sm.tsa.statespace.SARIMAX(train_dataset, order=(p_param, d_param, q_param)
                                        , seasonal_order=(P_param, D_param, Q_param, s_param)).fit()
        # seasonal_order = (0, 1, 1, 7)
        predictions.append(fit.forecast(30))
        predictions = np.array(predictions).reshape((-1, 30))

        model = auto_arima(train_dataset, seasonal=seasonal, m=s_param)
        predictions1 = model.predict(30)
        pred_2 = predictions1

        st.write(model)
        pred_1 = predictions[0]
        error_arima = mae(val_dataset, pred_1)
        error_arima1 = mae(val_dataset, pred_2)

        mape_arima = mape(val_dataset, pred_1)
        mape_arima1 = mape(val_dataset, pred_2)
    except Exception as e:
        st.write(f'p và q phải nhỏ hơn s = {s_param}')
        pass
    finally:
        try:
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
                           name="Pred ARIMA"),
                row=1, col=1
            )
            fig2.add_trace(
                go.Scatter(x=np.arange(70, 100), y=pred_2, mode='lines',
                           name="Pred AutoARIMA"),
                row=1, col=1
            )
            fig2.update_layout(height=400, width=800)
            st.plotly_chart(fig2)


            dl_data=pd.Series(pred_1)
            dl_data.name = f'{chosen_product}_{chosen_store}'

            data = dl_data.to_csv(index=False).encode()
            st.download_button(label="Tải xuống file dự báo", data=data, file_name='predict_30day.csv',
                               mime='text/csv', )
        except Exception as e:
            # st.write(e)
            pass

with tab3:
    try:

        st.plotly_chart(fig0)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
        error_type = st.radio('Loại sai số', ('MAE', 'MAPE'))
        if error_type == 'MAE':
            error = [error_base, error_avg, error_arima, error_arima1]
        if error_type == 'MAPE':
            error = [mape_base, mape_avg, mape_arima, mape_arima1]
        names = ["Baseline", "Moving average", "ARIMA", "AUTO-ARIMA"]
        df = pd.DataFrame(np.transpose([error, names]))
        df.columns = [f"{error_type}", "Mô hình"]
        fig3 = px.bar(df, y=f"{error_type}", x="Mô hình", color="Mô hình", title=f"{error_type} của các mô hình")
        st.plotly_chart(fig3)



    except Exception as e:
        # st.write(e)
        pass
