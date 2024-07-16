import pandas as pd
import json
import re
import openai
from dotenv import load_dotenv
import os
import streamlit as st
import time

def loadData():
    # Load the dataset
    #csvfile = r"D:\Jojo\LAMBTON\3. Spring May2024\1. BDM 3035 - Big Data Capstone Project\Capstone Project\Datasets\tweets-engagement-metrics.csv"
    csvfile = r"D:\Jojo\LAMBTON\3. Spring May2024\1. BDM 3035 - Big Data Capstone Project\Capstone Project\NLP_LLM\clean_data.csv"
    #df = pd.read_csv(csvfile, nrows=200)
    df = pd.read_csv(csvfile)
    #df = pd.read_csv(csvfile)
    #df = pd.read_csv(csvfile)


## Tweet clean up
# Preprocess the data if necessary (e.g., removing URLs, hashtags, mentions, etc.)
def procTweet(tweet):
    # Add your preprocessing steps here
    tweet = tweet.lower()
    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user mentions
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove punctuations
    tweet = re.sub(r'[^\w\s]', '', tweet)
    return tweet

def map_sentiment(senti_score):
    if senti_score > 0.051:
        return "Positive"
    elif senti_score < 0.049:
        return "Negative"
    else:
        return "Neutral"


## Coverting DF to JSON format
def dfToJSON(df):
    # Convert the DataFrame to the required format
    fine_tune_data = df[['tweet', 'sentiment']].rename(columns={'tweet': 'prompt', 'sentiment': 'completion'})

    # Save the dataset to a JSONL file
    with open('fine_tune_data.jsonl', 'w') as f:
        for i, row in fine_tune_data.iterrows():
            json.dump({"prompt": row['prompt'], "completion": row['completion']}, f)
            f.write('\n')

    #fine_tune_data.head()

## Fine Tune OpenAI
def fineTune_OpenAI(client):
    # * Upload the training JSON data file for fine tuning
    # * Check the status of the upload job process 
    # * Run the fine tuning based on our training data
    # * Check the status of the fine tuning job process

    ## Upload the training data JON file to OpenAI (we have limit it to 200 for now see the "nrows" when the CSV data is loaded)
    response = client.files.create(file=open('fine_tune_data.jsonl', 'rb'),
                                purpose='fine-tune')

    ## Print the id and the status of the "response" 
    # Here's the sample output:
    ## FileObject(id='file-mWIX5wYT7Lzem7ExBfNe8lU2', bytes=18895, created_at=1718510363, 
    #             filename='fine_tune_data.jsonl', object='file', purpose='fine-tune', 
    #             status='processed', status_details=None)
    #print(f"File ID : {response.id}")
    #print(f"Status : {response.status}")

    ## List all the values of the "response"
    #for key, value in response:
    #    print(f"{key} : {value}")
    return response.id, response.status

def createJob_OpenAI(client, responseID):
    ## Creata a job to fine-tune the model. It might take a while depending on the size of the file that is being processed
    ## So, ideally we only use a small subset of training data
    finetune_response = client.fine_tuning.jobs.create(
        training_file=responseID,
        model="davinci-002"
    )
    return finetune_response.id

def checkJob_OpenAI(client, responseID):
    ## Print the results/status information of the fine-tune job after submission
    #print(f"Fine tuning Job ID          : {finetune_response.id}")
    #print(finetune_response)
    #for key, value in finetune_response:
    #    print(f"{key} : {value}")

    ## Print the current results/status information of the fine-tuning job, make sure it is finished/successful
    ## before running the model, otherwise it will fail
    #client.fine_tuning.jobs.list(limit=1)
    jobStatus = client.fine_tuning.jobs.retrieve(responseID)
    #for key, value in jobStatus:
    #    print(f"{key} : {value}")
    
    ## Once the fine tuning is successful, we will get/used the "fine_tuned_model" from JobStatus object,
    ## otherwise, it will return "None"
    fineTunedModel = jobStatus.fine_tuned_model
    return fineTunedModel

def listJobEvents(client, responseID):
    ## To list all the events for the fine-tuning job ##
    response = client.fine_tuning.jobs.list_events(responseID, limit=5)
    events = response.data
    events.reverse()

    #for event in events:
    #    print(event.message)
    return events

def getlastModel(client):
    # List 10 fine-tuning jobs
    response = client.fine_tuning.jobs.list(limit=1)
    #response
    for i in enumerate(response):
        #print(f"Job ID: {i[1].id}, Fine tuned Model: {i[1].fine_tuned_model}, Status: {i[1].status}")
        ft_jobid = i[1].id
        ft_model = i[1].fine_tuned_model
        ft_status = i[1].status
        break
    return ft_model, ft_status
    

def AnalyzeSentiment(client, model, tweet):
    response = client.completions.create(
        model=model,
        prompt=tweet,
        max_tokens=1
        #temperature=0,
        #top_p=1,
        #frequency_penalty=0,
        #presence_penalty=0,
        #stop=["\n"]
    )
    return response


def uiTweets():
    #st.set_page_config(page_title="NLP with OpenAI", page_icon=":panda_face:")
    ## Testing the fine-tuned model
    #tweet = "AWS is just okay"
    user_tweet = st.text_input("Tweet: ")
    return user_tweet


def testTweet(client, fineTunedModel, tweet):
    TestResponse = AnalyzeSentiment(client, fineTunedModel, tweet)
    #TestResponse = client.completions.create(
    #        model=fineTunedModel,
    #        prompt=tweet,
    #        max_tokens=1
    #)
    #for key, val in TestResponse:
    #    print(f"{key} : {val}")

    #print(TestResponse.choices[0].text)
    st.write(TestResponse.choices[0].text)
    

def main():
    ## Load your API key from an environment variable or a configuration file
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")
    client = openai.OpenAI(api_key=openai.api_key)
    
    #with st.sidebar:
    #col1, col2 = st.columns(2)
    #st.sidebar.title("NLP with LLM support using OpenAI")
    #col1.st.subheader("Fine tune OpenAI with Tweet datasets")
    #if st.sidebar.button("Click here to start training"):
    #loadData()
    csvfile = r"D:\Jojo\LAMBTON\3. Spring May2024\1. BDM 3035 - Big Data Capstone Project\Capstone Project\NLP_LLM\clean_data.csv"
    df = pd.read_csv(csvfile)

    #df['tweet'] = df['tweet'].apply(procTweet)
    df['tweet'] = df['text'].apply(procTweet)
    #df.head()
        
    df["sentiment"] = df["Sentiment"].apply(map_sentiment)
    #df.head()
    ## Imbalance data...too many Neural
    #df["sentiment"].value_counts()
    ## This is just a TEMP routine to remove the Neutral for testing purposes
    #df = df[df["sentiment"] != "Neutral"]

    dfToJSON(df)
    
    ## Create 2 columns, 1st for training and 2nd for progress reporting
    col1, col2 = st.columns(2, gap="large")
    
    ##with col1:
    st.header("NLP with OpenAI : :panda_face:")
    
    #user_input = st.text_area("Enter Tweet to analyze: ", "")
    user_tweet = st.text_input("Tweet: ")
    #print(user_tweet)
    #st.write(user_tweet)
    
    button_label = "Evaluate sentiment"
    train_new = 0
    opt_rad1 = st.radio("Option", ["Train and fine-tune new model", "Use last Fine-tuned model"], captions = ["Train and fine-tune new model in OpenAI", "Use last Fine-tuned model from OpenAI"])
    if opt_rad1 == "Train and fine-tune new model":
        button_label = "Train new model"
        train_new = 1
    else:
        ft_model, ft_status = getlastModel(client)
        ft_msg = f"Last trained model was *{ft_model}* with status of *{ft_status}*"
        st.write(ft_msg)
   
    
    if st.button(button_label):    
        msg_placeholder = st.empty()
        msg_list = []
        if train_new:    ## Train new model
            responseID, responseStatus = fineTune_OpenAI(client)
            ##responseID, responseStatus = "001", "In Progress"
            #print(f"Response ID: {responseID}, Status: {responseStatus}")
            #msg1 = f"Response ID: {responseID}, Status : {responseStatus}"
            msg_list.append(f"Response ID: {responseID}, Status : {responseStatus}")
            msg_placeholder.text_area("Processing", "\n".join(msg_list), height=200)
            responseID = createJob_OpenAI(client, responseID)
                
            with st.spinner("Fine tuning new model in OpenAI..."):
                while True:
                    fineTunedModel = checkJob_OpenAI(client, responseID)
                    ##fineTunedModel = "Test tune Model"
                    #msg2 = f"Fine Tune Model : {fineTunedModel} - {type(fineTunedModel)}"
                    msg_list.append(f"Fine Tune Model : {fineTunedModel} - {type(fineTunedModel)}")
                    msg_placeholder.text_area("Processing", "\n".join(msg_list), height=200)
                    if fineTunedModel is not None:
                        break
                    else:
                        time.sleep(20)
                        events = listJobEvents(client, responseID)
                        #events = ["Processing...1", "Processing...2", "Processing...3", "Processing...4"]
                        for event in events:
                            msg_list.append(event.message)
                            msg_placeholder.text_area("Processing", "\n".join(msg_list), height=200)
                        continue
        else:   ## Use the last good model
            st.write(user_tweet)
            if ft_model is not None:
                fineTunedModel = ft_model
                    
        TestResponse = AnalyzeSentiment(client, fineTunedModel, user_tweet)
        #TestResponse = "Test response"
        st.write(TestResponse)
        sentiResults = TestResponse.choices[0].text 
        st.write(sentiResults)
    
    #tweet = uiTweets()
    #testTweet(client, fineTunedModel, tweet)
    

if __name__ == "__main__":
    main()