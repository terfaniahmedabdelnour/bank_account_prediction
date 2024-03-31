import numpy as np 
import  pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

# df = pd.read_csv('/home/ahmed/Desktop/formation_gomycode/Data Science Bootcamp/Datasets/Financial_inclusion_dataset.csv')

# encoded_label = LabelEncoder()

# df['country'] = encoded_label.fit_transform(df['country'])
# df['bank_account'] = encoded_label.fit_transform(df['bank_account'])
# df['location_type'] = encoded_label.fit_transform(df['location_type'])
# df['cellphone_access'] = encoded_label.fit_transform(df['cellphone_access'])
# df['age_of_respondent'] = encoded_label.fit_transform(df['age_of_respondent'])
# df['relationship_with_head'] = encoded_label.fit_transform(df['relationship_with_head'])
# df['marital_status'] = encoded_label.fit_transform(df['marital_status'])
# df['education_level'] = encoded_label.fit_transform(df['education_level'])
# df['job_type'] = encoded_label.fit_transform(df['job_type'])


# X = df.drop(columns=['bank_account','year','uniqueid'])
# y = df['bank_account']

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X_train, y_train)   
    
    
def load_data():
    df = pd.read_csv('Financial_inclusion_dataset.csv')
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
    return df

def train_model(df):
    X = df.drop(columns=['bank_account','year','uniqueid'])
    y = df['bank_account']  
    encoder = LabelEncoder()
    y_encoded = encoder.fit(df['bank_account'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = tree.DecisionTreeClassifier() 
    model.fit(X_train, y_train)
    return model

def main():
    st.title('Bank Account Prediction')
    
    # Load data
    data_load_state = st.text('Loading data...')
    df = load_data()
    data_load_state.text('Data loaded successfully!')
    

    # Train model
    model_load_state = st.text('Training model...')
    model = train_model(df)
    model_load_state.text('Model trained successfully!')

    # Input features
    countries = ['Kenya', 'Rwanda', 'Tanzania', 'Uganda']
    selected_country = st.selectbox("Countries: ", countries)
    # st.text(selected_country_encoded)
    
    location_type = st.radio("Select Your Location Type: ", ['Rural','Urban'])
    st.text(location_type)
    cellphone_access = st.checkbox("Cellphone Access")
    cellphone_access = 1 if cellphone_access else 0
    st.text(cellphone_access)
    household_size = st.slider("Select the Number of people living in one house", 1, 21)
    st.text(household_size)
    age_of_respondent = st.slider("Select Your age", 18, 100)
    st.text(age_of_respondent)
    gender_of_respondent = st.radio("Select Your Gender: ", ['Male','Female'])
    gender_of_respondent = 1 if gender_of_respondent == 'Female' else 0
    st.text(gender_of_respondent)
    relationship_with_head = st.selectbox("Relationship with the head of the house: ",['Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent','Other non-relatives'])
    st.text(relationship_with_head)
    marital_status = st.selectbox("The martial status: ",['Married/Living together', 'Widowed', 'Single/Never Married','Divorced/Seperated', 'Dont know'])
    st.text(marital_status)
    education_level = st.selectbox("The Highest level of education: ",['Secondary education', 'No formal education',
                                                          'Vocational/Specialised training', 'Primary education','Tertiary education', 'Other/Dont know/RTA'])
    st.text(education_level)
    job_type = st.selectbox("The Type of your job has: ",['Self employed', 'Government Dependent',
       'Formally employed Private', 'Informally employed',
       'Formally employed Government', 'Farming and Fishing',
       'Remittance Dependent', 'Other Income',
       'Dont Know/Refuse to answer', 'No Income'])
    st.text(job_type)
    
    # st.text(job_type)
    # Encode categorical features
    label_encoder = LabelEncoder()
    selected_country_encoded = label_encoder.fit_transform([selected_country])[0]
    relationship_with_head_encoded = label_encoder.fit_transform([relationship_with_head])[0]
    marital_status_encoded = label_encoder.fit_transform([marital_status])[0]
    education_level_encoded = label_encoder.fit_transform([education_level])[0]
    job_type_encoded = label_encoder.fit_transform([job_type])[0]
    location_type_encoded = label_encoder.fit_transform([location_type])[0]
    y_encoded = label_encoder.fit(df['bank_account'])
    # st.text(df['bank_account'])

    # Prediction
    if st.button('Predict'):
        X_user = [selected_country_encoded, location_type_encoded, cellphone_access, household_size, age_of_respondent, gender_of_respondent, relationship_with_head_encoded, marital_status_encoded, education_level_encoded, job_type_encoded]
    
        prediction = model.predict([X_user])
        if prediction == 1:
            st.success('You have a bank account')
        else:
            st.error('You dont have a bank account')
        
        # st.text(X_user.shape)
        # st.write('Prediction:',prediction)

if __name__ == '__main__':
    main()
