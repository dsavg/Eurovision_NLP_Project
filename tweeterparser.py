from datetime import timedelta, datetime
import got3
from textblob import TextBlob
import pandas as pd
import tweepy
from nlp_functions import *


def tweeter_parser(hashtag, starttime, endtime, maxtweets):
    """
    inputs: hashtag, starttime, endtime, maxtweets, thershold
    output: a dataframe with all the data
    In this function, I am trying to get twitter data from different
    times, cause got3 retrieves the most recent tweets
    """
    starting_str = starttime
    ending_str = endtime
    starting = datetime.strptime(starttime, '%Y-%m-%d')
    ending = datetime.strptime(endtime, '%Y-%m-%d')

    dd = (ending - starting).days

    # getting data from different time periods
    # the last time frame is 8 pm as the event starts at 9 pm
    days_list = []
    for i in range(dd+1):
        for j in [" %.2d" % k for k in range(24)]:
            days_list.append(starting_str + j + ':00')
            starting = datetime.strptime(starting_str, '%Y-%m-%d') + timedelta(days=1)
            starting_str = starting.strftime('%Y-%m-%d')

    days_list = days_list[:-1]
    print(days_list)
    data = []

    for i in range(len(days_list) - 1):

        tweetCriteria = got3.manager.TweetCriteria().\
            setLang('en').setQuerySearch(hashtag).setSince(starttime).\
            setUntil(days_list[i]).setMaxTweets(maxtweets)
        tweets = got3.manager.TweetManager.getTweets(tweetCriteria)

        for tweet in tweets:
            if TextBlob(tweet.text).detect_language() == 'en':
                data.append([tweet.text,
                             tweet.date,
                             tweet.hashtags,
                             tweet.retweets])
    eurovision = pd.DataFrame(data,
                              columns=['tweet', 'date', 'hashtag', 'retweets'])
    return(eurovision)


if __name__ == "__main__":
    # I am using as start date, the date of the 1st semi final,
    # and as last day the day of the competition
    eurovision_data = \
        tweeter_parser('#eurovision', "2018-05-08", "2018-05-12", 10000000)

    # removing duplicates
    eurovision_data = \
        eurovision_data.drop_duplicates(subset=['tweet'], keep='first')
    # applying tokenizer
    eurovision_data["tokenized_tweet"] = \
        eurovision_data['tweet'].apply(tokenizer)

    eurovision_data = pd.read_csv('./eurovision_tokenized.csv')
