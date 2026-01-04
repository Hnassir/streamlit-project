import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#from decisiontree import decisiontree


#========================================

st.header('Expresso churn Prediction - Form',text_alignment='center')
st.write('### Please Fill The Form :')

#========================================

with st.form('Form'):

    region=st.text_input('enter the location of the client',placeholder='text')
    tenure= st.text_input('enter the clients duration in the network',placeholder='text')
    Montant= st.number_input('enter the clients top-up amount',min_value=0)
    freq_rech=st.number_input('enter the number of times the customer refilled',min_value=0)
    revenue=st.number_input('enter the monthly income of the client',min_value=0)
    freq=st.number_input('enter the number of times the client has made an income',min_value=0)
    volume=st.number_input('enter the number of connections',min_value=0)
    
    st.subheader('enter the calls made by the client to :')

    expresso=st.number_input('expresso',placeholder='to the inter expresso',min_value=0)
    orange=st.number_input('orange',placeholder='to orange',min_value=0)
    tigo=st.number_input('tigo',placeholder='to tigo',min_value=0)
    zone1=st.number_input('zone1',placeholder='to zone1',min_value=0)
    zone2=st.number_input('zone2',placeholder='to zone2',min_value=0)

    st.space('small')

    mrg=st.toggle('client is going ? (Y/N)')
    st.space('small')

    regularity=st.number_input('enter the number of times the client is active for 90 days',min_value=0)
    top_pack=st.text_input('enter the most active packs',placeholder='text')
    freq_pack=st.number_input('enter the number of times the client has activated the top pack packages',min_value=0)
    
    st.space('medium')

    submit=st.form_submit_button('Predict Client Churn Probability',type='primary',width='stretch')

#========================================

new_data = pd.DataFrame([[freq,freq_pack,freq_rech,top_pack,tigo,int(mrg),regularity,zone1,zone2,orange,expresso,volume,revenue,Montant,tenure,region]])

#====================

encoder=LabelEncoder()

catg_cols=new_data.select_dtypes(include=['object'])

for col in catg_cols:
    new_data.loc[:,col]=encoder.fit_transform(new_data[col])

#======================

clf=joblib.load("decisiontree.pkl",mmap_mode='r+')

#clf=decisiontree()

res=clf.predict(new_data)

#============================

if submit:
    if res==1:
        st.warning("clients' churn probability is high")
    else :
        st.success("clients' churn probability is low")




