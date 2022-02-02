import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import xlsxwriter
from datetime import date
import plotly.express as px
from PIL import Image

# header written in markdown
st.write("""
# Iris Species Prediction App
- Predicting the total sales based on an adverstising campaign
""")

st.write("Explanation of dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set")

# create columns
col1, col2, col3 = st.columns(3)

with col1:
    # write the images to the app
    img_setosa = Image.open('Iris_setosa.jpg')
    st.image(img_setosa, caption='Iris Setosa')

with col2:
    img_versicolor = Image.open('Iris_versicolor.jpg')
    st.image(img_versicolor, caption='Iris versicolor')

with col3:
    img_virginica = Image.open('Iris_virginica.jpg')
    st.image(img_virginica, caption='Iris virginica')

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

# seaborn pairplot
#fig = sns.pairplot(iris, hue='species')
#st.pyplot(fig)
iris_pairplot = Image.open('iris_pairplot.png')
st.image(iris_pairplot, caption='Iris Pairplot')


# PLOTLY CHART
fig = px.scatter(iris, x='petal_length', y="petal_width", color="species", title="Iris Petal Width vs. Petal Length")
st.plotly_chart(fig)

# create sidebar
st.sidebar.header("Please input the campaign data below.")

# upload file
uploaded_file = st.sidebar.file_uploader("Upload prediction data", type=["csv"])
if uploaded_file is not None:
    sample = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        # get inputs
        sepal_length = st.sidebar.slider('Sepal Length:', min_value=0.0, max_value=10.0, step=0.1)
        sepal_width = st.sidebar.slider('Sepal Width:', min_value=0.0, max_value=5.0, step=0.1)
        petal_length = st.sidebar.slider('Petal Length:', min_value=0.0, max_value=10.0, step=0.1)
        petal_width = st.sidebar.slider('Petal Width:', min_value=0.0, max_value=5.0, step=0.1)
        sample = [[sepal_length, sepal_width, petal_length, petal_width]]

        return sample

    # if no file is uploaded, get campagin from sidebar
    sample = user_input_features()


# predict on the input data
loaded_model = pickle.load(open('iris_log_model.pkl', 'rb'))
loaded_scaler = pickle.load(open('iris_scaler.pkl', 'rb'))

# transform input data
scaled_sample = loaded_scaler.transform(sample)
pred = loaded_model.predict(scaled_sample)


# write to screen with variable
st.write("""
# Predictions
""")

# concat campaign and predictions into final dataframe
if uploaded_file is not None:
    pred = pd.DataFrame(pred)
    pred = pred.rename(columns={0:"Predictions"})
    final_pred_df = pd.concat([sample, pred], axis=1)
else:
    final_pred_df = pd.DataFrame(pred)
    sample = pd.DataFrame(sample)
    sample = sample.rename(columns={0:"Sepal Length", 1:"Sepal Width", 2:"Petal Length", 3:"Petal Width"})
    final_pred_df = pd.concat([sample,final_pred_df], axis=1)
    final_pred_df = final_pred_df.rename(columns={"Sepal Length":"Sepal Length", "Sepal Width":"Sepal Width", "Petal Length":"Petal Length", "Petal Width":"Petal Width",  0:"Prediction"})
st.write(final_pred_df)

st.write("Prediction Probability")
pred_prob = loaded_model.predict_proba(scaled_sample)
pred_prob_df = pd.Dataframe(pred_prob)
pred_prob_df = pred_prob_df.rename(columns={0:,"Setosa", 1:"Versicolo", 2:"Virginica"})

st.write(pred_prob_df)


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
df_xlsx = to_excel(final_pred_df)

# get current date
today = date.today()
# dd/mm/YY
today = today.strftime("%Y_%m_%d")
st.download_button(label='📥 Download Current Predictions',
    data= df_xlsx,
    file_name= f'iris_species_predictions_{today}.xlsx')







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
