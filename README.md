# Eurovision NLP Project

This repo hosts the Final Project for the NLP course from my Master's in Data Science at USF. 

## Authors:

* Danai Avgerinou - [LinkedIn](https://www.linkedin.com/in/danai-avgerinou/)

## Data Collection:

In order to fulfil this project, I had to retrieve old twitter data. Due to limitations of twitter API, I used the got3 folder from [GetOldTweets-python](https://github.com/Jefferson-Henrique/GetOldTweets-python). Further, there needed to be some validation of the countries, so I used [countryinfo.py](https://gist.github.com/pamelafox/986163).

## Process

So, first we need to collect the data through running [tweeterparser.py](https://github.com/dsavg/Eurovision_NLP_Project/blob/master/tweeterparser.py). This script outputs [eurovision_.csv](https://github.com/dsavg/Eurovision_NLP_Project/blob/master/eurovision_.csv). I created [nlp_functions.py](https://github.com/dsavg/Eurovision_NLP_Project/blob/master/nlp_functions.py), which hosts the nlp functions used. All the analysis is done in [Eurovision.ipynb](https://github.com/dsavg/Eurovision_NLP_Project/blob/master/Eurovision.ipynb) notebook.

## Results

Israel, the winning country, was identified in TFI-DF, as well as in the LDA topic model. This gives some validation, that those models can capture the winner of the Eurovision contest. Also, diving a little bit into the tweets that were about Israel, I identified people talking about the contest, as well as the current political situation at Israel.

## References:

- https://gist.github.com/pamelafox/986163
- https://github.com/Jefferson-Henrique/GetOldTweets-python