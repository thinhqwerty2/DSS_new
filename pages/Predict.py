import pickle

import numpy as np
import pandas as pd
import streamlit as st
from pmdarima import auto_arima

import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("Dự báo số lượng sản phẩm bán ra")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Read in the data
INPUT_DIR = r'D:\Dss\data'
with open('sales.pkl', 'rb') as f:
    stv = pickle.load(f)
    d_cols = [c for c in stv.columns if 'd_' in c]


store_list = stv['store_id'].unique()
chosen_store = st.selectbox('Chọn cửa hàng',store_list)
stv=stv.query(f'store_id=="{chosen_store}"')
chosen_product = st.selectbox('Chọn sản phẩm',stv['item_id'])
dataset=stv.query(f'item_id=="{chosen_product}"')
train_dataset = dataset[d_cols[-100:-30]].squeeze()
val_dataset = dataset[d_cols[-30:]].squeeze()
tab0, tab1, tab2, tab3 = st.tabs(["BaseLine", "Moving average", "SARIMAX", "So sánh"])
with tab0:
    try:
        predictions = []
        for i in range(len(val_dataset)):
            if i == 0:
                predictions.append(train_dataset[-1])
            else:
                predictions.append(val_dataset[i - 1])
        # predictions = np.transpose(np.array([row.tolist() for row in predictions]))
        # error_naive = np.linalg.norm(predictions[:3] - val_dataset.values[:3]) / len(predictions)
        pred_1 = predictions

        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(
            go.Scatter(x=np.arange(70), mode='lines', y=train_dataset, marker=dict(color="dodgerblue"),
                       name="Train"),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=np.arange(70, 100), y=val_dataset, mode='lines', marker=dict(color="darkorange"),
                       name="Val"),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=np.arange(70, 100), y=pred_1, mode='lines', marker=dict(color="seagreen"),
                       name="Pred"),
            row=1, col=1
        )


        fig.update_layout(height=400, width=800, title_text="Baseline")
        st.plotly_chart(fig)
    except:
        pass
with tab1:
    try:
        predictions = []
        for i in range(len(val_dataset.columns)):
            if i == 0:
                predictions.append(np.mean(train_dataset[train_dataset.columns[-30:]].values, axis=1))
            if i < 31 and i > 0:
                predictions.append(0.5 * (np.mean(train_dataset[train_dataset.columns[-30 + i:]].values, axis=1) + np.mean(predictions[:i], axis=0)))
            if i > 31:
                predictions.append(np.mean([predictions[:i]], axis=1))

        predictions = np.transpose(np.array([row.tolist() for row in predictions]))
        error_avg = np.linalg.norm(predictions[:3] - val_dataset.values[:3]) / len(predictions[0])


        pred_1 = predictions[0]
        pred_2 = predictions[1]
        pred_3 = predictions[2]

        fig = make_subplots(rows=3, cols=1)

        pred_1 = predictions[0]
        pred_2 = predictions[1]
        pred_3 = predictions[2]

        fig = make_subplots(rows=3, cols=1)

        fig.add_trace(
            go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[0].values, marker=dict(color="dodgerblue"),
                       name="Train"),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[0].values, mode='lines', marker=dict(color="darkorange"),
                       name="Val"),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=np.arange(70, 100), y=pred_1, mode='lines', marker=dict(color="seagreen"),
                       name="Pred"),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[1].values, marker=dict(color="dodgerblue"),
                       showlegend=False),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[1].values, mode='lines', marker=dict(color="darkorange"),
                       showlegend=False),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=np.arange(70, 100), y=pred_2, mode='lines', marker=dict(color="seagreen"), showlegend=False,
                       name="Denoised signal"),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[2].values, marker=dict(color="dodgerblue"),
                       showlegend=False),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[2].values, mode='lines', marker=dict(color="darkorange"),
                       showlegend=False),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(x=np.arange(70, 100), y=pred_3, mode='lines', marker=dict(color="seagreen"), showlegend=False,
                       name="Denoised signal"),
            row=3, col=1
        )

        fig.update_layout(height=1200, width=800, title_text="Moving average")

        st.plotly_chart(fig)

        # ss = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
        #
        # thirty_day_avg_map = stv.set_index('id')[d_cols[-30:]].mean(axis=1).to_dict()
        # fcols = [f for f in ss.columns if 'F' in f]
        # for f in fcols:
        #     ss[f] = ss['id'].map(thirty_day_avg_map).fillna(0)
        #
        # data = ss.to_csv().encode()
        # st.download_button(label="Tải xuống", data=data, file_name='predict_30day.csv',
        #                    mime='text/csv', )
    except:
        pass
# %% md
# ARIMA

# %%
# from statsmodels.tsa.stattools import adfuller
# # Assuming 'data' is your time series data
# result = adfuller(train_dataset.loc[0])
#
# # Extract the test statistics and p-value
# test_statistic = result[0]
# p_value = result[1]
#
# # Compare the p-value to the significance level
# if p_value < 0.05:
#     print("The time series is stationary.")
# else:
#     print("The time series is non-stationary.")
# %%

with tab2:
    try:

        import statsmodels.api as sm

        predictions = []
        for row in train_dataset[train_dataset.columns[-30:]].values[:100]:
            st.write(row)
            model = auto_arima(row,seasonal=True)
            # fit = sm.tsa.statespace.SARIMAX(row, seasonal_order=(0, 1, 1, 7)).fit()
            predictions.append(model.predict(30))
        predictions = np.array(predictions).reshape((-1, 30))
        # %%
        pred_1 = predictions[0]
        pred_2 = predictions[1]
        pred_3 = predictions[2]

        fig = make_subplots(rows=3, cols=1)
        fig.add_trace(
            go.Scatter(x=np.arange(71), mode='lines', y=train_dataset.loc[0].values, marker=dict(color="dodgerblue"),
                       name="Ori"),
            row=1, col=1
        )

        # fig.add_trace(
        #     go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[0].values, mode='lines', marker=dict(color="darkorange"),
        #                name="Val"),
        #     row=1, col=1
        # )

        fig.add_trace(
            go.Scatter(x=np.arange(70, 100), y=pred_1, mode='lines', marker=dict(color="seagreen"),
                       name="Pred"),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=np.arange(71), mode='lines', y=train_dataset.loc[1].values, marker=dict(color="dodgerblue"),
                       showlegend=False),
            row=2, col=1
        )

        # fig.add_trace(
        #     go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[1].values, mode='lines', marker=dict(color="darkorange"), showlegend=False),
        #     row=2, col=1
        # )

        fig.add_trace(
            go.Scatter(x=np.arange(70, 100), y=pred_2, mode='lines', marker=dict(color="seagreen"), showlegend=False),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=np.arange(71), mode='lines', y=train_dataset.loc[2].values, marker=dict(color="dodgerblue"),
                       showlegend=False),
            row=3, col=1
        )

        # fig.add_trace(
        #     go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[2].values, mode='lines', marker=dict(color="darkorange"), showlegend=False),
        #     row=3, col=1
        # )

        fig.add_trace(
            go.Scatter(x=np.arange(70, 100), y=pred_3, mode='lines', marker=dict(color="seagreen"), showlegend=False),
            row=3, col=1
        )
        fig.update_layout(height=1200, width=800, title_text="SARIMAX")
        st.plotly_chart(fig)
    except:
        pass
