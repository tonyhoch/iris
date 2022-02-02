import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import xlsxwriter
from datetime import date
import plotly.express as px

# header written in markdown
st.write("""
# Iris Species Prediction App
- Predicting the total sales based on an adverstising campaign
""")

# load train data to show users
iris = pd.read_csv('iris.csv')

st.write("""
Training dataset for the iris model:
""")

# write train data to app
st.write(iris)

# Draw correlation data

# Draw figures
st.write("## Training Data Graphs")

# All training datadata
st.write(" - All Training Data Points")


st.write(" - Move the sliders to adjust data shown in the training graphs below")

# Filter dataset to fit sliders

fig = sns.pairplot(iris, hue='species')
st.pyplot(fig)

# PLOTLY CHART
fig = px.scatter(iris, x='petal_length', y="petal_width", color="species", title="Iris Petal Width vs. Petal Length")
st.plotly_chart(fig)

#fig = px.scatter(df, x="TV", y="sales", color="newspaper", title="Sales vs. TV w/ Newspaper Spend", hover_data=["newspaper"])
#st.plotly_chart(fig)

#fig = px.scatter(df, x="radio", y="sales", color="newspaper", title="Sales vs. Radio w/ Newspaper Spend", hover_data=["newspaper"])
#st.plotly_chart(fig)


# create sidebar
st.sidebar.header("Please input the campaign data below.")

# upload file
uploaded_file = st.sidebar.file_uploader("Upload prediction data", type=["csv"])
if uploaded_file is not None:
    campaign = pd.read_csv(uploaded_file)
    #campaign.reset_index(drop=True, inplace=True)
else:
    def user_input_features():
        # get inputs
        tv = st.sidebar.slider('TV Spend:', min_value=0, max_value=500)
        radio = st.sidebar.slider('Radio Spend:', min_value=0, max_value=500)
        newspaper = st.sidebar.slider('Newspaper Spend:', min_value=0, max_value=500)
        campaign = [[tv, radio, newspaper]]

        return campaign

    # if no file is uploaded, get campagin from sidebar
    campaign = user_input_features()


# predict on the input data
#loaded_poly = pickle.load(open('final_poly_converter.pkl', 'rb'))
#loaded_model = pickle.load(open('sales_poly_model.pkl', 'rb'))

# transform input data
#campaign_poly = loaded_poly.transform(campaign)
#pred = loaded_model.predict(campaign_poly)


# write to screen with variable
st.write("""
# Predictions
""")

# concat campaign and predictions into final dataframe
if uploaded_file is not None:
    #pred = pd.DataFrame(pred)
    #pred = pred.rename(columns={0:"Predictions"})
    #final_pred_df = pd.concat([campaign, pred], axis=1)
    pass
else:
    #final_pred_df = pd.DataFrame(pred)
    #campaign = pd.DataFrame(campaign)
    #campaign = campaign.rename(columns={0:"TV", 1:"Radio", 2:"Newspaper"})
    #final_pred_df = pd.concat([campaign,final_pred_df], axis=1)
    #final_pred_df = final_pred_df.rename(columns={"TV":"TV", "Radio":"Radio", "Newspaper":"Newspaper", 0:"Prediction"})
    pass
#st.write(final_pred_df)


# export predictions to excel
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:H', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data
#df_xlsx = to_excel(final_pred_df)

# get current date
today = date.today()
# dd/mm/YY
today = today.strftime("%Y_%m_%d")
st.download_button(label='📥 Download Current Predictions',
    data=iris ,
    file_name= f'advertising_predictions_{today}.xlsx')







# @st.cache
# def convert_df(df):
#    return df.to_csv(index=False).encode('utf-8')

# csv = convert_df(final_pred_df)

# st.download_button(
#    "Download Predictions: CSV",
#    csv,
#    "final_sales_predictions.csv",
#    "text/csv",
#    key='download-csv'
# )