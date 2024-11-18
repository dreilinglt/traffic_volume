# Traffic Volume App
# Import libraries
import streamlit as st
import pandas as pd
import pickle

# Set up the app title and image
# Note Chat GPT was utlized to tweak code found on Stack Overflow for the color gradient text
color1 = 'red'
color2 = 'lime'
content = '**Traffic Volume Predictor**'
st.markdown(
    f'''
    <p style="
        text-align: center;
        background: linear-gradient(to right, {color1}, {color2});
        -webkit-background-clip: text;
        color: transparent;
        font-size: 54px;
        border-radius: 2%;
    ">
        {content}
    </p>
    ''', 
    unsafe_allow_html=True
)

st.write('Utilize our advanced Machine Learning application to predict traffic volume.')
st.image('traffic_image.gif')

# Reading the pickle file that we created before 
model_pickle = open('traffic_volume.pickle', 'rb') 
reg_model = pickle.load(model_pickle) 
model_pickle.close()

# Load the default dataset
default_df = pd.read_csv('Traffic_Volume.csv')
default_df['date_time'] = pd.to_datetime(default_df['date_time'])

# Get month, day of the week, and time of day
default_df['month'] = default_df['date_time'].dt.month_name()
default_df['weekday'] = default_df['date_time'].dt.day_name()  # Returns day names
default_df['hour'] = default_df['date_time'].dt.hour.astype(str)  # hour of the day
# drop unwanted columns
x = default_df.drop(columns = ['traffic_volume', 'date_time'])

# get options for forms
# Note Chat gpt was used to help with the syntax to ensure holiday_options had the right options
holiday_options = ['None'] + default_df['holiday'].dropna().unique().tolist() # Make a list with unique values and 'None' instead of Nan values
weather_main_options = default_df['weather_main'].unique()
# Ensure months, weekdays and hours are in the right order
month_options = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
weekday_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
hours_options = [str(i) for i in range(24)]

st.sidebar.image('traffic_sidebar.jpg')
st.sidebar.write('Traffic Volume Predictor')
st.sidebar.header('Input Features')
st.sidebar.write('You can either upload your data file or manually enter input features.')

with st.sidebar.expander("Option 1: Upload CSV file:"):
    user_file = st.file_uploader('Upload a CSV File containig the diamond details.')
    st.header('Sample Data Format for Upload')
    st.write(x.head())
    st.warning(':warning: Ensure your CSV file has the same column names and data types as shown above.')
with st.sidebar.expander('Option 2: Fill out form:'):
    with st.form('Enter the traffic details manually using the form below.'):
        holiday = st.selectbox('Choose whether or not today is a designated holiday or not', options=holiday_options)
        if holiday == 'None':
            holiday = None
        temp = st.number_input('Average Temperature in Kelvin', min_value=245.0, max_value=315.0)
        rain_1h = st.number_input('Amount in mm of rain in the past hour', min_value=0.0, max_value=default_df['rain_1h'].max())
        snow_1h = st.number_input('Amount in mm of snow in the past hour', min_value=0.0, max_value=default_df['snow_1h'].max())
        clouds_all = st.number_input('Percentage of cloud cover', min_value=0.0, max_value=100.0)
        weather_main = st.selectbox('Choose the current weather', options=weather_main_options)
        month = st.selectbox('Choose month', options=month_options)
        weekday = st.selectbox('Choose day of the week', options=weekday_options)
        hour = str(st.selectbox('Choose hour', options=hours_options))
        submit_button = st.form_submit_button('Submit Form')


if submit_button:
    st.success(':white_check_mark: Form data submitted successfully')
    # Encode the inputs for model prediction
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns=['traffic_volume', 'date_time'])
    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, month, weekday, hour]
    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)
    user_encoded_df = encode_dummy_df.tail(1)
    # Confidence Interval Slider
    alpha = st.slider('Select Alpha Values for prediction intervals:', min_value=.01, max_value=.5, step=.01)
    # Get the prediction with its intervals
    prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha)
    pred_value = prediction[0]
    if intervals[0,0]<0:
        lower_limit=[0,0]
    else:
        lower_limit = intervals[0, 0]
    upper_limit = intervals[0, 1]
    # limit 2 decimal places by using ': .2f'
    # Display price predictions
    st.header('Pedicting Traffic Volume...')
    st.write('Predicted Traffic Volume:')
    st.header(f'{round(prediction[0],0): .0f}')
    confidence_interval = (1 - alpha)*100
    # round answers then show 0 decimal places to ensure it looks correct
    st.write(f'**Confidence Interval** ({confidence_interval}%): [{round(lower_limit[0],0): .0f}, {round(upper_limit[0],0): .0f}]')
elif user_file is not None:
    # report success
    st.success(':white_check_mark: CSV file successfully uploaded')
    # Encode the inputs for model prediction
    encode_df = default_df.copy()
    # Combine the list of user data as rows to default_df
    user_file_df = pd.read_csv(user_file)
    # Ensure Hour is input as a string to match other data
    for i in range(len(user_file_df)):
        user_file_df['hour'][i] = str(user_file_df['hour'][i])
    encode_df_combined = pd.concat([encode_df, user_file_df]) 
    encode_df_combined =  encode_df_combined.drop(columns = ['traffic_volume', 'date_time'])
    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df_combined)
    user_file_length = len(user_file_df)
    user_encoded_df = encode_dummy_df.tail(user_file_length)
    # Confidence Interval Slider
    alpha = st.slider('Select Alpha Values for prediction intervals:', min_value=.01, max_value=.5, step=.01)
    prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha)
    lower_limit = intervals[:, 0]
    upper_limit = intervals[:, 1]
    # Get the prediction with its intervals
    user_file_df['Predicted Traffic Volume'] = prediction
    user_file_df['Lower Volume Limit'] = lower_limit
    user_file_df['Upper Volume Limit'] = upper_limit
    # Ensure lowest value is 0 and not negative
    user_file_df['Lower Volume Limit'] = user_file_df['Lower Volume Limit'].clip(lower=0)
    user_file_df = user_file_df.round({'Predicted Traffic Volume': 0, 'Lower Volume Limit': 0, 'Upper Volume Limit': 0})
    # Display price predictions
    st.header('Pedicting Volumes...')
    st.write('Predicted Volumes:')
    st.write(user_file_df)
else:
    st.info(':information_source: Please choose a data input method to proceed')
    # Confidence Interval Slider
    alpha = st.slider('Select Alpha Values for prediction intervals:', min_value=.01, max_value=.5, step=.01)

# Additional tabs for DT model performance
st.subheader("Model Performance and Inference")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('traffic_feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('traffic_residuals.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('traffic_pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('traffic_coverage.svg')
    st.caption("Range of predictions with confidence intervals.")