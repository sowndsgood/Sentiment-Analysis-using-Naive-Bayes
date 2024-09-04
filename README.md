# Tweet Sentiment Analysis using Naive Bayes

## Introduction

This project implements a sentiment analysis tool using the Naive Bayes classifier. The application processes tweets to predict whether their sentiment is positive, negative, or neutral. The tool leverages the NLTK library for natural language processing and is implemented with a simple, interactive Streamlit web interface.

## Features

- **Text Preprocessing:** Converts text to lowercase, removes punctuation and stopwords, and applies stemming.
- **Sentiment Classification:** Uses a Naive Bayes classifier with Laplace smoothing to predict sentiment.
- **Streamlit Interface:** Provides an interactive web interface for sentiment classification.
- **Performance Evaluation:** Measures the accuracy of the classifier on test data.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/username/tweet-sentiment-analysis.git
    cd tweet-sentiment-analysis
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download NLTK resources:**
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

## Usage

1. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2. **Classify Sentiment:**
    - Enter a tweet in the text area provided in the web app.
    - Click on the "Classify sentiment" button.
    - The predicted sentiment and associated scores will be displayed.

3. **Evaluate Accuracy:**
    - Modify the code to evaluate the accuracy of the classifier on the test dataset.

## Project Structure

tweet-sentiment-analysis/ │ ├── data/ # Folder containing the dataset │ ├── train.csv │ ├── images/ # Folder containing images for sentiment display │ ├── positive.jpg │ ├── negative.jpg │ └── neutral.jpg │ ├── app.py # Streamlit app script ├── train.py # Script to train the model (if needed) ├── requirements.txt # List of dependencies └── README.md # Project documentation (this file)


## Example

### Input: I love the film

### Output:
- **Predicted Sentiment:** Positive
- **Scores:**
    | Sentiment | Score  |
    |-----------|--------|
    | Positive  | 1.75   |
    | Negative  | -2.25  |
    | Neutral   | -1.50  |

## License

This project is licensed under the MIT License.
