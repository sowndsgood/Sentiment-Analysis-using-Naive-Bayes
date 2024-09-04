import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
nltk.download('punkt_tab')
nltk.download("stopwords")
import streamlit as st
from PIL import Image
import math
from sklearn.model_selection import train_test_split



st.set_page_config(page_title="Tweet Sentiment Analysis",page_icon="😄",layout='centered',initial_sidebar_state='auto')
#Load the dataset
df=pd.read_csv(r"data/train.csv")
df=df.drop(columns=["selected_text"])
df["text"]=df["text"].astype(str)

def preprocess_text(text):
    #Convert to lowercase
    text=text.lower() 
    #Remove Punctuation
    translation_table=str.maketrans("","",string.punctuation)
    text=text.translate(translation_table)
    #Remove stopwords
    stopwords_set=set(stopwords.words("english"))
    #Tokenisation
    tokens=nltk.word_tokenize(text)
    #Stemming
    stemmer=PorterStemmer()
    tokens=[stemmer.stem(token) for token in tokens if token not in stopwords_set]

    return tokens

def split_data_by_sentiment(data,sentiment):
    """
    This function separates the data based on each sentiment.
    """
    return data[data['sentiment']==sentiment]['text'].tolist()

positive_data=split_data_by_sentiment(df,'positive')
negative_data=split_data_by_sentiment(df,'negative')
neutral_data=split_data_by_sentiment(df,'neutral')

train_df,test_df=train_test_split(df,test_size=0.1,random_state=42)

def calculate_word_count(texts):
    word_count=defaultdict(int)
    for text in texts:
        tokens=preprocess_text(text)
        for token in tokens:
            word_count[token]+=1
    return word_count

positive_word_count=calculate_word_count(train_df[train_df['sentiment']=='positive']['text'])
negative_word_count=calculate_word_count(train_df[train_df['sentiment']=='negative']['text'])
neutral_word_count=calculate_word_count(train_df[train_df['sentiment']=='neutral']['text'])

print("The Positive word count:",positive_word_count)
print("The Negative word count:",negative_word_count)
print("The Neutral word count:",neutral_word_count)

def calculate_likelihood(word_count,total_words,laplace_smoothing=1):
    likelihood={}
    vocabulary_size=len(word_count)
    for word,count in word_count.items():
        likelihood[word]=(count+laplace_smoothing)/(total_words+vocabulary_size*laplace_smoothing)
    return likelihood

total_positive_words=sum(positive_word_count.values())
total_negative_words=sum(negative_word_count.values())
total_neutral_words=sum(neutral_word_count.values())

positive_likelihood=calculate_likelihood(positive_word_count,total_positive_words)
negative_likelihood=calculate_likelihood(negative_word_count,total_negative_words)
neutral_likelihood=calculate_likelihood(neutral_word_count,total_neutral_words)

log_positive_likelihood={word:math.log(prob) for word,prob in positive_likelihood.items()}
log_negative_likelihood={word:math.log(prob) for word,prob in negative_likelihood.items()}
log_neutral_likelihood={word:math.log(prob) for word,prob in neutral_likelihood.items()}

def calculate_log_prior(data,sentiment):
    log_prior=math.log(len(data[data['sentiment']==sentiment])/len(data))
    return log_prior

positive_log_prior=calculate_log_prior(df,'positive')
negative_log_prior=calculate_log_prior(df,'negative')
neutral_log_prior=calculate_log_prior(df,'neutral')

def classify_sentiment(text,positive_log_prior,negative_log_prior,neutral_log_prior,log_positive_likelihood,log_negative_likelihood,log_neutral_likelihood):
    tokens=preprocess_text(text)
    positive_score=positive_log_prior+sum([log_positive_likelihood.get(token, 0) for token in tokens])
    negative_score=negative_log_prior+sum([log_negative_likelihood.get(token, 0) for token in tokens])
    neutral_score=neutral_log_prior+sum([log_neutral_likelihood.get(token, 0) for token in tokens])
    scores={'Positive':positive_score,
            'Negative':negative_score,
            'Neutral':neutral_score}
    
    predicted=max(scores,key=scores.get)
    return predicted,scores

test_tweet="Hi I am satisfied"

predicted,scores=classify_sentiment(test_tweet,positive_log_prior,negative_log_prior,neutral_log_prior,log_positive_likelihood,log_negative_likelihood,log_neutral_likelihood)

print("Predicted Sentiments:",predicted)
print("Scores:",scores)

correct_predictions=0
total_predictions=len(test_df)

for index,row in test_df.iterrows():
    predicted,scores=classify_sentiment(row['text'],positive_log_prior,negative_log_prior,neutral_log_prior,log_positive_likelihood,log_negative_likelihood,log_neutral_likelihood)
    if predicted.lower()==row['sentiment'].lower():
        correct_predictions+=1

accuracy=correct_predictions/total_predictions
print("Accuracy:",accuracy)

def main():
    banner=Image.open(r"images\download.png")
    banner=banner.resize((300,200))
    st.image(banner,use_column_width=True)

    st.title("Sentiment Analysis using Naive Bayes")
    st.sidebar.title("Sentiment Analysis using Naive Bayes")
    st.sidebar.write("Sentiment Analysis with the Naive Bayes algorithm is a powerful approach, using probability and linguistic analysis to categorize text sentiments as positive 😊, negative 😡, or neutral 😐." )
    st.sidebar.write('\n')
    st.sidebar.write("By preprocessing text, calculating log priors, and deriving log-likelihoods, this method quantifies sentiment, guiding accurate classification. Its adaptability across domains and Python’s NLTK library make it accessible for diverse applications.")
    st.sidebar.write('\n')
    st.sidebar.write("Naive Bayes enriches decision-making, offering insights into customer satisfaction, brand perception, and trends, crucial in an era of abundant textual data.")

    input=st.text_area("Enter the text here:","For example: I love the film",key="text_input")

    if st.button("Classify sentiment",key="classify_button"):
        predicted,scores=classify_sentiment(input,positive_log_prior,negative_log_prior,neutral_log_prior,log_positive_likelihood,log_negative_likelihood,log_neutral_likelihood)

        image_size=(200,200)

        if predicted=='Positive':
            image=Image.open(r"images\positive.jpg")
        elif predicted=='Negative':
            image=Image.open(r"images\negative.jpg")
        else:
            image=Image.open(r"images\neutral.jpg")

        
        resized_image=image.resize(image_size)
       
        st.markdown(f'<p align="center"><b>{predicted.upper()}</b></p>', unsafe_allow_html=True)

        st.image(resized_image)

        scores=pd.DataFrame(scores.items(),columns=["Sentiment","Scores"])
    
        st.write("Scores:")
        st.table(scores)

if __name__=="__main__":
    main()

