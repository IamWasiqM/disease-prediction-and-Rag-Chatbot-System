

import streamlit as st
from streamlit_option_menu import option_menu  
import pickle



import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv
load_dotenv()

# loading the saved models
diabetes_model = pickle.load(open("saved model files/diabetes_model.sav",'rb'))
heart_disease_model = pickle.load(open("saved model files/heart_disease_model.sav",'rb'))
parkinsons_model = pickle.load(open("saved model files/parkinsons_model.sav",'rb'))


# sidebar for navigation
with st.sidebar:
    selected = option_menu('Disease Prediction System',
    ['Diabetes Prediction',
    'Heart Disease Prediction',
    'Rag Chatbot'],
    icons = ['activity','heart','chat-dots'],
    default_index = 0)

# Diabetes prediction page

if (selected == 'Diabetes Prediction'):

    #page title
    st.title('Diabetes Prediction Using ML')

    # Getting the input data from the user
    # Columns for input fields 
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    #code for prediction
    heart_diagnosis = ''

    # creating button for prediction

    if st.button("Diabetes Test Result"):
        heart_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age ]])

        if (heart_prediction[0]== 1):
            heart_diagnosis = "The person is Diabetic"
        elif (heart_prediction[0]== 0):
            heart_diagnosis = "The person is Not Diabetic"

    st.success(heart_diagnosis)



# Heart Disease prediction page

if (selected == 'Heart Disease Prediction'):

    #page title
    st.title('Heart Disease Prediction Using ML') 

    # Getting the input data from the user
    # Columns for input fields 
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Gender')
    with col3:
        cp = st.text_input('Chest Pain Types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestrol in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')  
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    #code for prediction
    heart_diagnosis = ''

    if st.button("Heart Disease Test Result"):
        try:
            input_data = [[
                float(age), float(sex), float(cp),
                float(trestbps), float(chol), float(fbs),
                float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca),
                float(thal)
            ]]

            heart_prediction = heart_disease_model.predict(input_data)

            if heart_prediction[0] == 1:
                heart_diagnosis = "The Person has a Heart Disease."
            else:
                heart_diagnosis = "The Person does not have a Heart Disease."

        except ValueError:
            st.error("Please enter valid numeric values in all fields.")

    st.success(heart_diagnosis)




# Rag Chatbotpage

if (selected == 'Rag Chatbot'):

    #page title
    st.title('Preliminary Medical Rag Chatbot ') 
    
    DB_FAISS_PATH="vectorstore/db_faiss"
    @st.cache_resource
    def get_vectorstore():
        embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db


    def set_custom_prompt(custom_prompt_template):
        prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
        return prompt


    def main():
        

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['content'])

        prompt=st.chat_input("Pass your prompt here")

        if prompt:
            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append({'role':'user', 'content': prompt})
                    
            try: 
                vectorstore=get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")

                GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
                GROQ_MODEL_NAME = "llama-3.1-8b-instant"  # Change to any supported Groq model
                llm = ChatGroq(
                    model=GROQ_MODEL_NAME,
                    temperature=0.5,
                    max_tokens=512,
                    api_key=GROQ_API_KEY,
                )
                
                retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

                # Document combiner chain (stuff documents into prompt)
                combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

                # Retrieval chain (retriever + doc combiner)
                rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

                response=rag_chain.invoke({'input': prompt})

                result=response["answer"]
                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append({'role':'assistant', 'content': result})

            except Exception as e:
                st.error(f"Error: {str(e)}")

    if __name__ == "__main__":
        main()
        
