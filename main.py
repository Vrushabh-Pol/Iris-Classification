import streamlit as st
import pickle
import numpy as np

lin_model = pickle.load(open("lin_model.pkl","rb"))
log_model = pickle.load(open("log_model.pkl","rb"))
svc_model = pickle.load(open("svc_model.pkl","rb"))

def classify(num):
    if num < 0.5:
        return "Setosa"
    elif num<1.5:
        return "Versicolor"
    else:
        return "Virginica"

def main():

    html_temp = """<div style = "background-color:#8A2BE2;padding:16px">
     <h2 style="color:black;text-align:center;"> Iris Classification Using ML</h2>
     </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.write('')
    st.write('')

    activities = ["Linear Regression ","Logistic Regression ","SVM "]
    #names = ["1","2"]
    option = st.sidebar.selectbox("Which Model Wold You Like To Use ? ",activities)
    #option1 = st.sidebar.selectbox("Which Model Wold You Like To Use ? ",names)

    st.sidebar.write('') # to give space between 2 lines
    st.sidebar.write('')
    st.sidebar.write('<span style="color: black;">The Accuracy Of Models Are Shown Below.</span>', unsafe_allow_html=True)
    linscore = st.sidebar.write ('<span style="color: green;">Linear Regression      : 0.922096</span>', unsafe_allow_html=True)
    logscore = st.sidebar.write ('<span style="color: green;">Logistic Regression    : 0.947368</span>', unsafe_allow_html=True)
    svmscore = st.sidebar.write ('<span style="color: green;">Support Vector Machine : 0.947368</span>', unsafe_allow_html=True)



    st.subheader(option)

    sl = st.number_input('Enter Sepal Length :', 0.0, 10.0, step=1.0)
    sw = st.number_input('Enter Sepal Width :', 0.0, 10.0, step=1.0)
    pl = st.number_input('Enter Petal Length :', 0.0, 10.0, step=1.0)
    pw = st.number_input('Enter Petal Width :', 0.0, 10.0, step=1.0)

    inputs = [[sl,sw,pl,pw]]
    if st.button("classify"):
        if option == "Linear Regression":
            st.success(classify(lin_model.predict(inputs)))
        elif option == "Logistic Regression":
            st.success(classify(log_model.predict(inputs)))
        else:
            st.success(classify(svc_model.predict(inputs)))

if __name__ == '__main__':
    main()
