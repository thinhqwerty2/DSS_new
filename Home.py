import pickle
from datetime import datetime
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta
from matplotlib.patches import Polygon

st.set_option('deprecation.showPyplotGlobalUse', False)

pd.set_option('display.max_columns', 50)
plt.style.use('bmh')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

st.markdown("# Dữ liệu 🎉")
st.sidebar.markdown("# Dữ liệu 🎉")
tab1, tab2 = st.tabs(["Upload dữ liệu", "Thông tin dữ liệu"])
stv = None
sellp = None
cal = None
with tab1:
    st.subheader("Tải lên dữ liệu bán hàng")
    sales = st.file_uploader("Choose a CSV file", key='sales')

    st.subheader("Tải lên dữ liệu về giá")
    sell_prices = st.file_uploader("Choose a CSV file", key='sell_prices')

    st.subheader("Tải lên dữ liệu về ngày tháng")
    calendar = st.file_uploader("Choose a CSV file", key='calendar')

    if sales is not None:
        stv = pd.read_csv(sales)
        with open('sales.pkl', 'wb') as f:
            pickle.dump(stv, f)
    else:
        with open('sales.pkl', 'rb') as f:
            stv = pickle.load(f)

    if sell_prices is not None:
        sellp = pd.read_csv(sell_prices)
        with open('sell_prices.pkl', 'wb') as f:
            pickle.dump(sellp, f)
    else:
        with open('sell_prices.pkl', 'rb') as f:
            sellp = pickle.load(f)

    if calendar is not None:
        cal = pd.read_csv(calendar)
        with open('calendar.pkl', 'wb') as f:
            pickle.dump(cal, f)
    else:
        with open('calendar.pkl', 'rb') as f:
            cal = pickle.load(f)

    try:
        st.subheader('100 dòng dữ liệu từng mặt hàng')
        st.dataframe(stv.head(100))
        st.subheader('100 dòng dữ liệu về giá bán')
        st.dataframe(sellp.head(100))
        st.subheader('100 dòng dữ liệu về các ngày')
        st.dataframe(cal.head(100))
    except:
        pass


with tab2:
    with st.container():
        tab21, tab22, tab23 = st.tabs(['Tổng quan', 'Theo cửa hàng', 'Theo từng sản phẩm'])
        if sales is None:
            with open('sales.pkl', 'rb') as f:
                stv = pickle.load(f)
        if sell_prices is None:
            with open('sell_prices.pkl', 'rb') as f:
                sellp = pickle.load(f)
        if calendar is None:
            with open('calendar.pkl', 'rb') as f:
                cal = pickle.load(f)

        if stv is not None and sellp is not None and cal is not None:
            d_cols = [c for c in stv.columns if 'd_' in c]  # sales data columns
            # Calendar data looks like this (only showing columns we care about for now)
            cal[['d', 'date', 'event_name_1', 'event_name_2',
                 'event_type_1', 'event_type_2', 'snap_CA']].head()
            # %%
            # Merge calendar on our items' data
            with tab23:
                st.subheader('Dữ liệu chuỗi thời gian của một mặt hàng')
                product_id = st.selectbox('Chọn sản phẩm', stv['id'], 8412)
                query_day23=st.slider(
                    "Chọn khoảng thời gian",
                    min_value=datetime(2011,1,29),
                    max_value=datetime(2016,6,19),
                    value=(datetime(2011,1,29), datetime(2016,6,19)),format="MM/YY",key=23)
                example = stv.loc[stv['id'] == product_id][d_cols].T
                example = example.rename(columns={example.columns[0]: product_id})  # Name it correctly
                example = example.reset_index().rename(columns={'index': 'd'})  # make the index "d"
                example = example.merge(cal, how='left', validate='1:1')
                example['date']=pd.to_datetime(example['date'])
                example=example.set_index('date').loc[query_day23[0]:query_day23[1]]
                example[product_id] \
                    .plot(figsize=(15, 5),
                          color=next(color_cycle),
                          title=f'{product_id} ')
                st.pyplot()


                examples = [product_id]
                example_df = [example]

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))
                example_df[0].groupby('wday').mean([examples[0]])[examples[0]] \
                    .plot(kind='line',
                          title='Trung bình tuần bán',
                          lw=5,
                          color=color_pal[0],
                          ax=ax1)
                example_df[0].groupby('month').mean([examples[0]])[examples[0]] \
                    .plot(kind='line',
                          title='Trung bình tháng bán',
                          lw=5,
                          color=color_pal[4],
                          ax=ax2)
                example_df[0].groupby('year').mean([examples[0]])[examples[0]] \
                    .plot(kind='line',
                          lw=5,
                          title='Trung bình năm bán',
                          color=color_pal[2],
                          ax=ax3)
                fig.suptitle(f'Xu hướng của sản phẩm: {examples[0][:-16]}',
                             size=20,
                             y=1.1)
                plt.tight_layout()
                st.pyplot(fig)

                st.subheader('Giá của một mặt hàng')
                fig, ax = plt.subplots(figsize=(15, 5))
                stores = []
                # st.write(product_id[:-16])
                for store, d in sellp.query(f'item_id == "{product_id[:-16]}"').groupby('store_id'):
                    d.plot(x='wm_yr_wk',
                           y='sell_price',
                           style='.',
                           color=next(color_cycle),
                           figsize=(15, 5),
                           title=f'Giá của {product_id[:-16]} theo từng cửa hàng',
                           ax=ax,

                           legend=store)
                    stores.append(store)
                    plt.legend()
                plt.legend(stores)
                ax.set_xlabel('Id của tuần')
                ax.set_ylabel('Giá bán')
                st.pyplot()

            # st.subheader('20 mặt hàng khác')
            # twenty_examples = stv.sample(20, random_state=529) \
            #     .set_index('id')[d_cols] \
            #     .T \
            #     .merge(cal.set_index('d')['date'],
            #            left_index=True,
            #            right_index=True,
            #            validate='1:1') \
            #     .set_index('date')
            # # %%
            # fig, axs = plt.subplots(10, 2, figsize=(15, 20))
            # axs = axs.flatten()
            # ax_idx = 0
            # for item in twenty_examples.columns:
            #     twenty_examples[item].plot(title=item,
            #                                color=next(color_cycle),
            #                                ax=axs[ax_idx])
            #     ax_idx += 1
            # plt.tight_layout()
            # st.pyplot()
            #

            # st.subheader('Số loại sản phẩm theo phân loại sản phẩm')
            # stv['cat_id'].unique()
            # # %%
            # stv.groupby('cat_id').count()['id'] \
            #     .sort_values() \
            #     .plot(kind='barh', figsize=(15, 5))
            # st.pyplot()
            with tab21:

                query_day21 = st.slider(
                    "Chọn khoảng thời gian",
                    min_value=datetime(2011, 1, 29),
                    max_value=datetime(2016, 6, 19),
                    value=(datetime(2011, 1, 29), datetime(2016, 6, 19)), format="MM/YY",key=21)

                st.subheader('Tổng số lượng sản phẩm bán được theo phân loại sản phẩm')
                fig, ax = plt.subplots(1, 1)
                past_sales = stv.set_index('id')[d_cols] \
                    .T \
                    .merge(cal.set_index('d')['date'],
                           left_index=True,
                           right_index=True,
                           validate='1:1') \
                    .set_index('date')
                past_sales.index=pd.to_datetime(past_sales.index)
                for i in stv['cat_id'].unique():
                    items_col = [c for c in past_sales.columns if i in c]
                    past_sales[items_col].loc[query_day21[0]:query_day21[1]] \
                        .sum(axis=1) \
                        .plot(figsize=(15, 5),
                              alpha=0.8)
                plt.legend(stv['cat_id'].unique())
                st.pyplot(fig)
                # %%

                st.subheader('Phần trăm hàng đã bán')
                past_sales_clipped = past_sales.clip(0, 1)
                fig, ax = plt.subplots(1, 1)
                for i in stv['cat_id'].unique():
                    items_col = [c for c in past_sales.columns if i in c]
                    (past_sales_clipped[items_col].loc[query_day21[0]:query_day21[1]] \
                     .mean(axis=1) * 100) \
                        .plot(figsize=(15, 5),
                              alpha=0.8,
                              style='.')
                plt.legend(stv['cat_id'].unique())
                st.pyplot(fig)

                st.subheader('Phân tích theo từng ngày')


                def calmap(ax, year, data):
                    ax.tick_params('x', length=0, labelsize="medium", which='major')
                    ax.tick_params('y', length=0, labelsize="x-small", which='major')

                    # Month borders
                    xticks, labels = [], []
                    start = datetime(year, 1, 1).weekday()
                    for month in range(1, 13):
                        first = datetime(year, month, 1)
                        last = first + relativedelta(months=1, days=-1)

                        y0 = first.weekday()
                        y1 = last.weekday()
                        x0 = (int(first.strftime("%j")) + start - 1) // 7
                        x1 = (int(last.strftime("%j")) + start - 1) // 7

                        P = [(x0, y0), (x0, 7), (x1, 7),
                             (x1, y1 + 1), (x1 + 1, y1 + 1), (x1 + 1, 0),
                             (x0 + 1, 0), (x0 + 1, y0)]
                        xticks.append(x0 + (x1 - x0 + 1) / 2)
                        labels.append(first.strftime("%b"))
                        poly = Polygon(P, edgecolor="black", facecolor="None",
                                       linewidth=1, zorder=20, clip_on=False)
                        ax.add_artist(poly)

                    ax.set_xticks(xticks)
                    ax.set_xticklabels(labels)
                    ax.set_yticks(0.5 + np.arange(7))
                    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
                    ax.set_title("{}".format(year), weight="semibold")

                    # Clearing first and last day from the data
                    valid = datetime(year, 1, 1).weekday()
                    data[:valid, 0] = np.nan
                    valid = datetime(year, 12, 31).weekday()
                    # data[:,x1+1:] = np.nan
                    data[valid + 1:, x1] = np.nan

                    # Showing data
                    ax.imshow(data, extent=[0, 53, 0, 7], zorder=10, vmin=-1, vmax=1,
                              cmap="RdYlBu_r", origin="lower", alpha=.75)


                # # %%
                # print('The lowest sale date was:', past_sales.sum(axis=1).sort_values().index[0],
                #       'with', past_sales.sum(axis=1).sort_values().values[0], 'sales')
                # print('The lowest sale date was:', past_sales.sum(axis=1).sort_values(ascending=False).index[0],
                #       'with', past_sales.sum(axis=1).sort_values(ascending=False).values[0], 'sales')
                # # %%
                from sklearn.preprocessing import StandardScaler

                sscale = StandardScaler()
                past_sales.index = pd.to_datetime(past_sales.index)
                cat_id_plot = st.multiselect('Chọn loại sản phẩm', stv['cat_id'].unique(), default=['FOODS'])
                for i in cat_id_plot:
                    fig, axes = plt.subplots(3, 1, figsize=(20, 8))
                    items_col = [c for c in past_sales.columns if i in c]
                    sales2013 = past_sales.loc[past_sales.index.isin(pd.date_range('31-Dec-2012',
                                                                                   periods=371))][items_col].mean(
                        axis=1)
                    vals = np.hstack(sscale.fit_transform(sales2013.values.reshape(-1, 1)))
                    calmap(axes[0], 2013, vals.reshape(53, 7).T)
                    sales2014 = past_sales.loc[past_sales.index.isin(pd.date_range('30-Dec-2013',
                                                                                   periods=371))][items_col].mean(
                        axis=1)
                    vals = np.hstack(sscale.fit_transform(sales2014.values.reshape(-1, 1)))
                    calmap(axes[1], 2014, vals.reshape(53, 7).T)
                    sales2015 = past_sales.loc[past_sales.index.isin(pd.date_range('29-Dec-2014',
                                                                                   periods=371))][items_col].mean(
                        axis=1)
                    vals = np.hstack(sscale.fit_transform(sales2015.values.reshape(-1, 1)))
                    calmap(axes[2], 2015, vals.reshape(53, 7).T)
                    plt.suptitle(i, fontsize=30, x=0.4, y=1.01)
                    plt.tight_layout()
                    st.pyplot()



            with tab22:
                query_day22 = st.slider(
                    "Chọn khoảng thời gian",
                    min_value=datetime(2011, 1, 29),
                    max_value=datetime(2016, 6, 19),
                    value=(datetime(2011, 1, 29), datetime(2016, 6, 19)), format="MM/YY",key=22)
                store_list = sellp['store_id'].unique()
                store_list = st.multiselect('Chọn cửa hàng',store_list,default=store_list)
                st.subheader(f'Trung bình số lượng sản phẩm bán trong')
                sum_day = st.slider('Chọn số ngày', min_value=1, max_value=365, value=90, step=1,key='sum_day1')
                fig, ax = plt.subplots(1, 1)
                for s in store_list:
                    store_items = [c for c in past_sales.columns if s in c]
                    past_sales[store_items] \
                        .sum(axis=1) \
                        .rolling(sum_day).mean() \
                        .plot(figsize=(15, 5),
                              alpha=0.8,
                              )
                plt.legend(store_list)
                st.pyplot(fig)


                st.subheader(f'Trung bình số lượng sản phẩm bán trong')
                sum_day2 = st.slider('Chọn số ngày', min_value=1, max_value=365, value=7, step=1,key='sum_day2')
                fig, axes = plt.subplots(int(np.ceil(len(store_list)/2)), 2, figsize=(15, 10), sharex=True)
                axes = axes.flatten()
                ax_idx = 0
                for s in store_list:
                    store_items = [c for c in past_sales.columns if s in c]
                    past_sales[store_items] \
                        .sum(axis=1) \
                        .rolling(sum_day2).mean() \
                        .plot(alpha=1,
                              ax=axes[ax_idx],
                              title=s,
                              lw=3,
                              color=next(color_cycle))
                    ax_idx += 1
                # plt.legend(store_list)
                plt.suptitle(f'Trung bình số lượng sản phẩm bán trong {sum_day2} ngày theo cửa hàng')
                plt.tight_layout()
                st.pyplot()

            # %%

            # # %%
            # thirty_day_avg_map = stv.set_index('id')[d_cols[-30:]].mean(axis=1).to_dict()
            # fcols = [f for f in ss.columns if 'F' in f]
            # for f in fcols:
            #     ss[f] = ss['id'].map(thirty_day_avg_map).fillna(0)
            #
            # ss.to_csv('submission.csv', index=False)
