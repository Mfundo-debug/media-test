
# Lets create streamlit webapp for revenue prediction and able to filter
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


# Add logo image
image = Image.open('download.png')
st.image(image, use_column_width=True)

st.write('Welcome to Arerna Media Revenue Prediction App')
st.write('This app is created to predict the revenue of Arerna Media based on the marketing spend')

# Load the model
model = joblib.load('model.pkl')

# Load the data
data = pd.read_csv('forecast.csv')

# Filter the data with the ability to select multiple options
selected_publications = st.multiselect('Select Publications', ['ST', 'BL', 'FM', 'BD','All'], default=['ST', 'BL', 'FM', 'BD'])
selected_types = st.multiselect('Select Types', ['Corporate', 'Individual','All'], default=['Corporate', 'Individual'])
selected_methods = st.multiselect('Select Methods', ['Digital', 'Print','All'], default=['Digital', 'Print'])

# Apply the filters
if 'All' not in selected_publications:
    data = data[data['Publication'].isin(selected_publications)]
if 'All' not in selected_types:
    data = data[data['Type'].isin(selected_types)]
if 'All' not in selected_methods:
    data = data[data['Method'].isin(selected_methods)]

# Show the data
st.write(data)

# Create the first set of tabs for the visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Publication vs Revenue", "Revenue Distribution", "Subscribers Distribution", 
    "Subscribers vs Revenue", "Type vs Revenue"
])

with tab1:
    import plotly.express as px

    fig = px.bar(data, x='Publication', y='Revenue', title='Publication vs Revenue')
    st.plotly_chart(fig)

with tab2:
    fig = px.histogram(data, x='Revenue', nbins=30, title='Revenue Distribution')
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig)

with tab3:
    fig = px.histogram(data, x='Subscribers', nbins=30, title='Subscribers Distribution', marginal='rug')
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig)

with tab4:
    fig = px.scatter(data, x='Subscribers', y='Revenue', title='Subscribers vs Revenue')
    st.plotly_chart(fig)

with tab5:
    fig = px.box(data, x='Type', y='Revenue', title='Type vs Revenue')
    st.plotly_chart(fig)

# Create the second set of tabs for the visualizations
tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Method vs Revenue", "Revenue Trend", "ARPU Trend", 
    "Cancellations by Reason", "Churn Types"
])

with tab6:
    fig = px.box(data, x='Method', y='Revenue', title='Method vs Revenue')
    st.plotly_chart(fig)

with tab7:
    fig = px.line(data, x='Period', y='Revenue', title='Revenue Trend')
    st.plotly_chart(fig)

with tab8:
    fig = px.line(data, x='Period', y='ARPU', title='ARPU Trend')
    st.plotly_chart(fig)

with tab9:
    fig = px.pie(data, names='Reason', title='Cancellations by Reason')
    st.plotly_chart(fig)

with tab10:
    churn_data = data[['Voluntary Churn', 'Involuntary Churn']].apply(pd.Series.value_counts).fillna(0).sum(axis=1).reset_index()
    churn_data.columns = ['Churn Type', 'Count']
    
    fig = px.pie(churn_data, names='Churn Type', values='Count', title='Distribution of Churn Types')
    st.plotly_chart(fig)



# Ensure the data has the necessary columns for prediction
if 'Revenue' in data.columns and 'Subscribers' in data.columns and 'ARPU' in data.columns:
    # Create tabs for the model predictions
    tab11, tab12, tab13 = st.tabs(["Predicted Revenue", "Predicted Subscribers", "Predicted ARPU"])

    with tab11:
        # Predict the revenue
        #revenue = model.predict(data[['Revenue']])
        #st.write('The predicted revenue is:', revenue)

         # Ensure the 'Period' index is in datetime format
        data.index = pd.to_datetime(data.index)
        # Split the data into training and testing sets
        train_size = int(len(data) * 0.8)
        train, test = data['Revenue'][:train_size], data['Revenue'][train_size:]

        # Fit the ARIMA model
        model = ARIMA(train, order=(5, 1, 0))  # (p, d, q) parameters can be tuned
        model_fit = model.fit()
            # Make predictions
        predictions = model_fit.forecast(steps=len(test))
        test.index = predictions.index  # Align the index of the test set with the predictions

        # Evaluate the model
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        st.write(f'Root Mean Squared Error: {rmse}')
        data['Period'] = data['Period'].astype(str)
        data.set_index('Period', inplace=True)
        data.sort_index(inplace=True)

        # Generate the forecast for the next 12 months
        forecast = model_fit.forecast(steps=12, alpha=0.05)  # 95% confidence interval
        forecast.index = pd.date_range(start=data.index[-1], periods=12, freq='M')

        # Create traces for the plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Revenue'], mode='lines', name='Historical'))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Forecast', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast - 2 * rmse, fill=None, mode='lines', line_color='gray', showlegend=False))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast + 2 * rmse, fill='tonexty', mode='lines', line_color='gray', fillcolor='rgba(128, 128, 128, 0.2)', showlegend=False))

        # Update layout
        fig.update_layout(
            title='Revenue Forecast for the Next 12 Months',
            xaxis_title='Period',
            yaxis_title='Revenue',
            legend=dict(x=0, y=1),
            xaxis=dict(tickangle=45)
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

    with tab12:
        st.write("The model is loading soon...")
        

    with tab13:
        # Predict the ARPU
        st.write("The model is loading soon...")
        
else:
    st.write("The data does not exist")

        
