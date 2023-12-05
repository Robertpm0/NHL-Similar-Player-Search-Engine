
import streamlit as st
import pandas as pd
import requests
import json
def getNewPlayers():

    playerNames=[]
    for i in range(0,7700,100):
        #print(i)
    #i=0
        playerURL=rf"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=true&isGame=false&sort=%5B%7B%22property%22:%22goals%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start={i}&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20232024%20and%20seasonId%3E=19171918"

        allPlayers=requests.get(playerURL).json()["data"]

        playersDf = pd.DataFrame(allPlayers)
        #print(playersDf.skaterFullName)
        for name in playersDf.skaterFullName:
            playerNames.append(name)

    print(len(playerNames))

    pd.to_pickle(playerNames,"NHLNames.pkl")

st.title("NHL Similar Player Search")


st.multiselect("")
import streamlit as st

import datetime as datetime

 

players=["0x1","0x2","0x3"]

st.title("Similar Player Search")

 

selectedPlayers=st.multiselect("Select Player(s)",players)

 

today = datetime.datetime.now()

start = today.year -10

jan_1 = datetime.date(start, 1, 1)

dec_31 = datetime.date(start, 12, 31)

minDate=None

 

#Logic here to get first date of first game played for each player

 

#logic here to get most recent date played for each player

 

#logic to get

 

 

 

# def findSimilarPlayers(numPlayers,player,dateRange,comparisonRange,isCum):

 

#similarlity algorithm:

#different algorithms for different selected options

# if time series, use DTW fo differet metrics, jaccard for actions, weight results of each for final simialrity

 

#cosine between each data points

 

 

 

date_range=st.date_input("Basis Range",(jan_1,datetime.date(start,1,7)),min_value=minDate,max_value=datetime.datetime.today())

col1,col2=st.columns(2)

with col1:  

    isCum=st.checkbox("Cumulative")

with col2:

    st.markdown("",help="Compares total stats for selected date range rather than a game by game basis")

col3,col4=st.columns(2)

with col3:

    matchDates=st.checkbox("Analyze Exact Dates")

with col4:

    st.markdown("",help="When selected, only the play during the date range will be analyzed")

col5,col6=st.columns(2)

isCustomRange=False

if not matchDates:

    with col5:

        isCustomRange=st.checkbox("Custom Comparison Range")

    with col6:

        st.markdown("",help="When selected, you will choose a specifc range to analyze similar player rather than of all current player's history")

 

if isCustomRange:

    custom_date_range=st.date_input("Comparison Range",(jan_1,datetime.date(start,1,4)),min_value=minDate,max_value=datetime.datetime.today())

 

numMatches=st.number_input("How many similar players would you like to find?",min_value=1,max_value=25)

 

st.info('This is an experimental tool, all results should be taken with a grain of salt', icon="ℹ️")

similar=st.button("Search")

 