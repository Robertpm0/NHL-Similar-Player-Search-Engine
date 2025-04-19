import requests
import pandas as pd
import json
import streamlit as st
import datetime as datetime
from datetime import timedelta
import numpy as np
import stumpy
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from functools import partial
import secrets
import heapq

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import plotly.graph_objects as go


# class for finding best distances or matches used in a heap
class SearchDistance:
    distance=0
    player=""
    series=""
    def __init__(self,dist,name,srs):
        self.distance=dist
        self.player=name
        self.series=srs
        # over ride less than operator so heap can return player names 
    def __lt__(self,other):
        return self.distance < other.distance
#defmovingAverage():

# computers a rolling sum of a timeseries passed to function
# used for trending out player stats
def rollingSum(timeSeries,rollLen):
    rollSum=[]
    newSeries=[]
    miniLen=0

    for dataPoint in timeSeries:
        rollSum.append(dataPoint)
        #print(dataPoint)
        if len(rollSum)>rollLen:
            print("POPPED")
            rollSum.pop(0)
        newSeries.append(sum(rollSum))
    if len(newSeries) <rollLen:
        return newSeries
    else:
        
        return newSeries[rollLen:]

# gets All time series Player data --> gAtsPd
# is in certain range
# must check if position code is needed
def gAtsPd(startDate,endDate,baseLen,lagLen,isDtw,positions=None):
    # retrieve players and their total games played
    playerList,gamesPlayed=getSpecificPlayers(startDate,endDate,positions)
   # print(playerList)
   # print(gamesPlayed)
    #max games played for period desired
    maxGames=max(gamesPlayed)
    # print("MAX: ",maxGames)

    timeSeries=[]
    ti=0
    tsProgress=st.progress(0.0)
   # with ThreadPool(processes=8) as pool:
    workerCOunt=0
    validPlayers=[]

    for player in playerList:
        tsProgress.progress(ti/len(playerList))
        #print(player)
        # only want players who have atleast 85% of games played 
        #else we could compare to dudes who showed up for 1-2 games even lol

        if gamesPlayed[ti]>=baseLen and isDtw==False:
            workerCOunt+=1
            validPlayers.append(player)
        elif gamesPlayed[ti]>=lagLen and isDtw==True:
            workerCOunt+=1
            validPlayers.append(player)



        ti+=1
    print("NUM VALID Players: ",len(validPlayers))
    # use multi threading to get the players data quickers
    with ThreadPool(processes=8) as pool:
                    #print(ti)
        part=partial(getPlayerData,startDate=startDate-timedelta(days=lagLen),endDate=endDate)
        timeSeries=pool.map(part,validPlayers)


    #finalData=pd.concat(timeSeries)
        # return list of df/s to make easier to search
    tsProgress.empty()
    return timeSeries




#Gets ALL NHL Players Data who are active in a specifc date range, positions, and cumulitative
def getAllPlayerData(startDate,endDate,positions=None):

    posString0="positionCode%3D%22"
   # posString="positionCode%3D"
    #"positionCode%3D%22L%22%20or"
    # string manipulation to query specifc player positions
    if positions!=None:
        for l in range(0,len(positions)):
            if len(positions)==1:
                posString=posString0+fr"{positions[l]}%22"

            elif l<len(positions)-1:
                if l==0:
                    posString=posString0+fr'"{positions[l]}%22or%20"'
                else:
                    posString=posString+fr'"{positions[l]}%22or%20"'

            else:
                    posString=posString+fr'"{positions[l]}%22"'


    #print(posString)



    #get total records for player
    req=requests.get(fr"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=true&isGame=true&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22goals%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22assists%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=({posString})%20and%20gameDate%3C=%22{endDate}%2023%3A59%3A59%22%20and%20gameDate%3E=%22{startDate}%22%20and%20gameTypeId=2")
    data=[]
    # print("TOT",req.json()["total"])
    # now gather all data for player by chunking 100 at a time due to nhl api limits
    if req.json()["total"] ==0:
        return pd.DataFrame()
    for x in range(0,req.json()["total"],100):
         req=requests.get(fr"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=true&isGame=true&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22goals%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22assists%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start={x}&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=({posString})%20and%20gameDate%3C=%22{endDate}%2023%3A59%3A59%22%20and%20gameDate%3E=%22{startDate}%22%20and%20gameTypeId=2")
         df=pd.DataFrame(req.json()["data"])
         data.append(df)
    # print(data)
    allData=pd.concat(data,ignore_index=True)

    return allData
# gets NHL data for a given player for a given date range
def getPlayerData(playerName,startDate,endDate,cumul=False,allDates=False,positions=None):
    splitName=playerName.split()
    # print(splitName)
    # get max amount of data for player
    if allDates:
        endDate=datetime.date.today()
        req = requests.get(fr"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22gameDate%22,%22direction%22:%22ASC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3C=%22{endDate}%2023%3A59%3A59%22%20and%20gameDate%3E=%221917-12-19%22%20and%20gameTypeId=2%20and%20skaterFullName%20likeIgnoreCase%20%22%25{splitName[0]}%20{splitName[1]}%25%22")

        reqJ=req.json()["data"]
        df = pd.DataFrame(reqJ)
        #print(df.head())
        data=[]
        for x in range(0,req.json()["total"],100):
            req = requests.get(fr"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22gameDate%22,%22direction%22:%22ASC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start={x}&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3C=%22{endDate}%2023%3A59%3A59%22%20and%20gameDate%3E=%221917-12-19%22%20and%20gameTypeId=2%20and%20skaterFullName%20likeIgnoreCase%20%22%25{splitName[0]}%20{splitName[1]}%25%22")
            df=pd.DataFrame(req.json()["data"])

            #print(x)
            data.append(df)
            #print(df.head())
        allData = pd.concat(data)
        #print(allData.head())
        #print(len(allData))
        return allData
# get cumulalitive stats for the player
    elif cumul:
         req = requests.get(fr"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=true&isGame=true&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22goals%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22assists%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=50&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3C=%22{endDate}%2023%3A59%3A59%22%20and%20gameDate%3E=%22{startDate}%22%20and%20gameTypeId=2%20and%20skaterFullName%20likeIgnoreCase%20%22%25{splitName[0]}%20{splitName[1]}%25%22")
         reqJ=req.json()["data"]
         df = pd.DataFrame(reqJ)
         allData=df
         return allData
        #print(df.head())
# get game by game stats
    else:
        #print(startDate)
        #print(endDate)
        try:
            req = requests.get(fr"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22gameDate%22,%22direction%22:%22ASC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3C=%22{endDate}%2023%3A59%3A59%22%20and%20gameDate%3E=%22{startDate}%22%20and%20gameTypeId=2%20and%20skaterFullName%20likeIgnoreCase%20%22%25{splitName[0]}%20{splitName[1]}%25%22")
        except:
            return pd.DataFrame()
        #print("*******************************************************")
        #print("REQYEST:",req.json()["data"])
        try:
            reqJ=req.json()["data"]
            df = pd.DataFrame(reqJ)
        except:
            print(playerName)
            
            return pd.DataFrame()

        data=[]
        for x in range(0,req.json()["total"],100):
            req = requests.get(fr"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22gameDate%22,%22direction%22:%22ASC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start={x}&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3C=%22{endDate}%2023%3A59%3A59%22%20and%20gameDate%3E=%22{startDate}%22%20and%20gameTypeId=2%20and%20skaterFullName%20likeIgnoreCase%20%22%25{splitName[0]}%20{splitName[1]}%25%22")
            df=pd.DataFrame(req.json()["data"])

            #print(x)
        #    print(df)
            data.append(df)
            #print(df.head())
        if len(data)!=0:

            allData = pd.concat(data)

            return allData
        else:
            return pd.DataFrame()


# driver function to find similar players based on a given input player
def findPlayers(cumul,player,basePeriod,searchPeriod,numMatches,allDates): 
    baseData=getPlayerData(player[0],basePeriod[0],basePeriod[1],cumul)
    # cumul means we are chechking total stats so far game by game
    if cumul:

        # how far are we searching in days
        searchDistance=basePeriod[1]-basePeriod[0]
        # print(searchDistance)
        searchResults={}
        # get distance or similarity between each player and the search player
        for player in players:
            td=datetime.date.today()
            if allDates:

               firstTemp=getfirstGame(player)
               compareDistance=firstGame-td
               startDate=firstTemp
               endDate=firstTemp+searchDistance
            else:
                compareDistance=(searchPeriod[1]-searchPeriod[0]).days
                endDate=searchPeriod[0]+searchDistance
                startDate=searchPeriod[0]
                
            # if the num games played is more than the ones played in the base player
            #period than we must reduce the number of games we look at so it is a fair comaprison
            loops=1
            if compareDistance > searchDistance.days:
                loops=compareDistance/searchDistance
            playerStats=[]
            if loops >1:
                end = searchDistance

                start=0
                # get all th
                while end< searchDistance*loops:
                    tempData=getPlayerData(player,startDate,endDate,True) # get the given players data
                    playerStats.append(tempData)
                    start=start+1 
                    end=end+1
                    startDate=startDate+1
                    endDate=endDate+1
            else:
                tempData=getPlayerData(player,searchPeriod[0],searchPeriod[1],True,allDates=allDates) # player has same games as base player
                playerStats.append(tempData)
            searchResults[player]=playerStats
        # print(searchResults)


    else:

        baseData=getPlayerData(player)
        #print(">>>>>>>>>>>>>>>>>>>")

# get the euclidean distance between a given stat    
def calculate_weighted_euclidean_distance(row, reference_row, columns, weights):
    squared_distances = [(row[col] - reference_row[col]) ** 2 * weight for col, weight in zip(columns, weights)]
    return np.sqrt(sum(squared_distances).astype(np.float64))


# returns the rank of all dats in a list from a datasource in terms of similarity
def rankStats(stats,data,name):
    returnData=[]
    for stat in stats:
        rank=getRank(data,name,stat)
        returnData.append(rank)
    return returnData
#gets th rank of a given stat from a dataframe
def getRank(df, name, value_col, ascending=True):


  filtered_df = df.sort_values(by=value_col, ascending=ascending)
  filtered_df=filtered_df.reset_index(drop=True)
  print(filtered_df)
  return filtered_df.index[filtered_df["skaterFullName"]==name].tolist()[0] + 1  # Add 1 for 1-based ranking

# main driver to find and rank all players in a specific daterange to a base player(s)
def findAllPlayers(cumul,player,basePeriod,searchPeriod,numMatches,compStats,pos,lbs0): 
    

    baseDistance=basePeriod[1]-basePeriod[0]
    compDistance=searchPeriod[1]-searchPeriod[0]
    # two main methodologies for cumulative and non cumulative data
    if cumul:
        # gets max data for players in a range
        compData=getAllPlayerData(searchPeriod[0],searchPeriod[1],positions=pos)
   
        # print(player[0])
        # search playe we want to find who's closest
        baseData=getPlayerData(player[0],basePeriod[0],basePeriod[1],cumul)
        # print(compDistance)
        # print(baseDistance)
        # print(baseData)
        
        # find the similarity score for all players who have played games less than or = to 
        # the search period but atleast 85 * of the given games in that period
        if compDistance<=baseDistance:
            compData=getAllPlayerData(searchPeriod[0],searchPeriod[1],positions=pos)
            comparisonStats=compStats
  
            fo=False
            sp=False
            toi=False 
            # normalize a few specifc stats for proper compariopsoj
            for stat in compStats:
                if stat=="faceoffWinPct":
                    comparisonStats.remove("faceoffWinPct")
                    fo=True
                elif stat=="shootingPct":
                    comparisonStats.remove("shootingPct")
                    sp=True
                elif stat=="timeOnIcePerGame":
                    comparisonStats.remove("timeOnIcePerGame")
                    toi=True
            compData[comparisonStats]=compData[comparisonStats].div(compData["gamesPlayed"],axis=0)
            baseData[comparisonStats]=baseData[comparisonStats].div(baseData["gamesPlayed"],axis=0)
            
            if fo:

                comparisonStats.append("faceoffWinPct")
            if sp:
                comparisonStats.append("shootingPct")
            if toi:
                comparisonStats.append("timeOnIcePerGame")
            # weight certain stats based on user inputs
            weights=[]
            for i in range(0,len(comparisonStats)):
                weights.append(lbs0[i])
            # Columns to consider
            columns_to_consider = comparisonStats  # Add the columns you want to consider
            condition= (baseData["skaterFullName"] == player)
            reference_row = baseData.loc[condition]  # Replace with your specific row

# Function to calculate weighted Euclidean distance for specific columns
            maxGames=compData["gamesPlayed"].max()
            
            compData=compData.where(compData["gamesPlayed"]>maxGames*0.65)
# Apply the function to each row to calculate the weighted Euclidean distance for specific columns
            compData['WED'] = compData.apply(
    lambda row: calculate_weighted_euclidean_distance(row, reference_row, columns_to_consider, weights),
    axis=1
)
            # now find the best players by sorting on the distance function
            sortd=compData.sort_values(by="WED",ascending=True)
            cats=comparisonStats
            sortBased=pd.concat([sortd,baseData],ignore_index=True)
            # print(sortBased)
            newBase=rankStats(cats,sortBased,baseData.skaterFullName.values[0])

            nc=0
            # now make a radar chart with the best / most similar stats per player
            for p in sortd[comparisonStats][0:numMatches+1].index:
                # rank stats:
                p=rankStats(cats,sortBased,sortd.skaterFullName.values[nc])
                # print("P")
                # print(p)
               # p=sortd[comparisonStats][0:numMatches+1].loc[p]
                radarFig=go.Figure()
                radarFig.add_trace(go.Scatterpolar(
                r=newBase
                ,theta=cats
                ,fill='toself'
                ,name=baseData.skaterFullName.values[0]
            ))
                ranCol=secrets.token_hex(3)
                radarFig.add_trace(go.Scatterpolar(
                r=p
                ,theta=cats 
                ,fill='toself'
                ,name=sortd.skaterFullName.values[nc]
                ,fillcolor=f"#{ranCol}"
                ,opacity=0.5
            ))
                nc+=1
               

                
                radarFig.update_layout(polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0,len(compData.dropna(how='all'))+1]
                )
            ),showlegend=False)
                st.plotly_chart(radarFig)
            comparisonStats.append("skaterFullName")
            return [sortd[comparisonStats][0:numMatches+1],reference_row[comparisonStats]]
        # else we must  search the base distance in chunks in the search players to see when that search players
        #distance could be most similar
        # so when we want to see if any player has ever played similar to this given time frame for this given player 
        else:
            scores=[]
            # print("DISTANCE",compDistance.days-baseDistance.days)
            stepLength=(compDistance.days-baseDistance.days)/20
            if (stepLength *20) <= 25:
                stepLength=1

            myProgress=st.progress(0)
            print("SL",stepLength)
            for start in range(0,compDistance.days-baseDistance.days,int(stepLength)):
                myProgress.progress(int((start/(stepLength*20))*100))
                print("DIF",start)
                end =start+baseDistance.days
                strt=searchPeriod[0]+timedelta(days=start)
                nd=searchPeriod[0]+timedelta(days=end)

                print(strt,nd)
                compData=getAllPlayerData(strt,nd,positions=pos)
                if compData.empty:
                    continue
                comparisonStats=compStats
            #comparisonStats=["assists","evGoals","evPoints","goals","otGoals","plusMinus","points","ppGoals","ppPoints","shGoals","shPoints","shots"]
                #print(compData[comparisonStats])
                #print(compData["gamesPlayed"])
                fo=False
                sp=False
                toi=False
                for stat in compStats:
                    
                    if stat=="faceoffWinPct":
                        comparisonStats.remove("faceoffWinPct")
                        fo=True
                    elif stat=="shootingPct":
                        comparisonStats.remove("shootingPct")
                        sp=True
                    elif stat=="timeOnIcePerGame":
                        comparisonStats.remove("timeOnIcePerGame")
                        toi=True
                        
                compData[comparisonStats]=compData[comparisonStats].div(compData["gamesPlayed"],axis=0)
                baseData[comparisonStats]=baseData[comparisonStats].div(baseData["gamesPlayed"],axis=0)

                if fo:

                    comparisonStats.append("faceoffWinPct")
                if sp:
                    comparisonStats.append("shootingPct")
                if toi:
                    comparisonStats.append("timeOnIcePerGame")


                weights=[]
                for i in range(0,len(lbs0)):
                    weights.append(lbs0[i])
                # Columns to consider
                columns_to_consider = comparisonStats  # Add the columns you want to consider
                condition= (baseData["skaterFullName"] == player)
                reference_row = baseData.loc[condition]  # Replace with your specific row

        # Function to calculate weighted Euclidean distance for specific columns

        # Apply the function to each row to calculate the weighted Euclidean distance for specific columns
                compData['WED'] = compData.apply(
        lambda row: calculate_weighted_euclidean_distance(row, reference_row, columns_to_consider, weights),
        axis=1
        )
                maxGames=compData["gamesPlayed"].max()
                #only show results where players have atleast 85% of the maximum games, else we could get outliers
                compData=compData.where(compData["gamesPlayed"]>=baseData["gamesPlayed"].values[0]*.80)
                #print(compData)
                sortd=compData.sort_values(by="WED",ascending=True)
                scores.append(sortd[0:numMatches+1])
            myProgress.empty()
            print(scores)
            scores=pd.concat(scores)
            scores=scores.sort_values(by="WED",ascending=True)
            comparisonStats.append("skaterFullName")
            return [scores[comparisonStats][:numMatches],reference_row[comparisonStats]]
        # NOT CUMULATIVE
        #get data for both players, 
        # if equal do comp
        # if not windowed search with m = min(p1,p2)
        # return smallest distances
    # time series comparison of player stats
    else:
        score=[]
        # print(compDistance.days)
        # print(baseDistance.days)
        # dererming if we are making an equal comaprions
        if compDistance.days!=baseDistance.days:
            # if the search distance is greater than the baseline distance 
            if compDistance>baseDistance:
                #print(player[0])

                baseTs2=getPlayerData(player[0],basePeriod[0],basePeriod[1])
                baseTs=getPlayerData(player[0],basePeriod[0]-timedelta(days=baseTs2.shape[0]),basePeriod[1])
                
                #print(baseTs)
                # get all time series data for players within a search range
                searchTs=gAtsPd(searchPeriod[0],searchPeriod[1],positions=pos,baseLen=baseTs.shape[0],lagLen=baseTs2.shape[0],isDtw=False)

                resultDf=pd.DataFrame()
                resultMap={}
                valueMap={}
             #   print("TSSSSSSS",searchTs)
             # loop through the time series and get simialrit score for each stat's time series
                for ts in searchTs:
                    # if the time series we are comparing is bigger than the base line
                    #find best sub sequnce 
                    if len(ts) > baseTs.shape[0]:
                      #  print(compStats[0])
                        for stat in compStats:
                         #   print(stat)
                            # find most similar range
                            distance_profile = stumpy.mass(np.array(rollingSum(baseTs[compStats[0]],baseTs2.shape[0])).astype(np.float64),np.array(rollingSum(ts[compStats[0]],baseTs2.shape[0])).astype(np.float64))
                            k = 1
                            idxs = np.argpartition(distance_profile, k)[:k]
                            idxs = idxs[np.argsort(distance_profile[idxs])]
                        # best_match_index = np.argmin(distance_profile[:, 0])
                            best_match_index = np.argmin(distance_profile)

                            resultMap[ts.skaterFullName.values[0]]=best_match_index
                            valueMap[ts.skaterFullName.values[0]]=distance_profile[best_match_index]

                    # else we must use DTW to see which time series is most simlar to the base one as theyre smaller
                    else:
                        # preform dtw on search to base
                        # still to be implemented
                        pass
                # find the best time series and plot them on top of the comaprios one
                # for now we are only plotting one stat
                # to do is to incoropate more stats into the final calcualtion
                sorted_items = sorted(valueMap, key=valueMap.get)
                top_n_keys=sorted_items[:numMatches]
                for k in top_n_keys:
                    for ts in searchTs:
                        if ts.skaterFullName.values[0]==k:
                            similar_subseries = ts[compStats[0]][resultMap[k] : resultMap[k] + len(baseTs)]
                            fig=plt.figure()
                            ranCol=secrets.token_hex(3)
                            plt.plot(rollingSum(similar_subseries,len(baseTs2)),c=f"#{ranCol}",label=k,linewidth=4)
                            plt.plot(rollingSum(baseTs[compStats[0]],len(baseTs2)),c="black",label=baseTs.skaterFullName.values[0],linewidth=2)
                            legnd=plt.legend()
                            plt.xlabel("Games")
                            plt.ylabel("Rolling Goal Sum")
                
                           # fig.tight_layout()
                            plt.title(f"{k} vs. {baseTs.skaterFullName.values[0]} : {stat}")
                            st.pyplot(fig)

                return [top_n_keys,baseTs]
                        

                #use base as motif and search compTS
                # we must do a windowed search via dtw or try stump
                # finding best time series with DTW since we are garunteed for them to be less than the base

            else:
                rollLength=10
                #baseTs2=getPlayerData(player[0],basePeriod[0],basePeriod[1])
                baseTs=getPlayerData(player[0],basePeriod[0]-timedelta(days=rollLength),basePeriod[1])
                
                #print(baseTs)
                searchTs=gAtsPd(searchPeriod[0],searchPeriod[1],positions=pos,baseLen=baseTs.shape[0],lagLen=rollLength,isDtw=True)
                distances=[]
                # print(baseTs)
                querySeries=rollingSum(baseTs[compStats[0]],rollLength)
                # print(compDistance.days)
                # print("SER",querySeries)
                for ts in searchTs:
                    if len(ts)==0:
                        continue
                    for stat in compStats:
                                         #   print(stat)
                        # print(ts)
                        g=rollingSum(ts[compStats[0]],rollLength)
                        # print(g)
                        #pritn()
                            #distance_profile = stumpy.mass(np.array(rollingSum(baseTs[compStats[0]],baseTs2.shape[0])
                        # perform DTW on the time srries to the base time series
                        # print("HI")
                        if len(g)==0:
                            break
                        x,_=fastdtw(np.array(querySeries),np.array(g),dist=2)
                        # add it to the heap for easily findig best time series
                        heapq.heappush(distances,SearchDistance(x,ts.skaterFullName.values[0],g))
                
                # find the best time series now and plot them against search frame
                for temp in range(0,numMatches):
                #for k in top_n_keys:
                    j=heapq.heappop(distances)
                    k=j.player
                    for ts in searchTs:
                        if len(ts)==0:
                            continue
                        elif ts.skaterFullName.values[0]==baseTs.skaterFullName.values[0]:
                            continue
                    
                            
                
                        if ts.skaterFullName.values[0]==k:
                            similar_subseries = j.series
                            fig=plt.figure()
                            ranCol=secrets.token_hex(3)
                            plt.plot(similar_subseries,c=f"#{ranCol}",label=k,linewidth=4)
                            plt.plot(querySeries,c="black",label=baseTs.skaterFullName.values[0],linewidth=2)
                            legnd=plt.legend()
                            plt.xlabel("Games")
                            plt.ylabel("Rolling Goal Sum")
                
                           # fig.tight_layout()
                            plt.title(f"{k} vs. {baseTs.skaterFullName.values[0]} : {stat}")
                            st.pyplot(fig)



#else distance are equal
        else:
            print("sup")

        #determine distance ratios



                

            
#ALTERNATIVE METHOD: 
            # CALC % DIF OF EACH VAL, SUM DIFFERENCE , weight each value based on improtance
            # BASE - COMP % change = DIF


# finds first game for a given player and their postion
def getfirstGame(playerName):

    splitName=playerName[0].split()
    req = requests.get(fr"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22gameDate%22,%22direction%22:%22ASC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=5&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20232024%20and%20seasonId%3E=19171918%20and%20skaterFullName%20likeIgnoreCase%20%22%25{splitName[0]}%20{splitName[1]}%25%22")
    gameDf=pd.DataFrame(req.json()["data"])
    firstGame=gameDf["gameDate"].min()
    position=gameDf["positionCode"][0]
    firstGame=datetime.datetime.strptime(firstGame,'%Y-%m-%d')
    return firstGame,position


#print(df.head())


#https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=true&isGame=true&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22goals%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22assists%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=50&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3C=%222023-12-04%2023%3A59%3A59%22%20and%20gameDate%3E=%222023-10-01%22%20and%20gameTypeId=2

#//*[@id="root"]/main/div[5]/div[2]/div[2]/span[1]/text()[1]
#import streamlit as st

#import datetime as datetime

# gets all active players
def getNewPlayers():

    playerNames=[]
    start=datetime.date.today().year
    end=0
    if datetime.date.today().month>8:
        end=start+1
    else:
        end = start
        start-=1
    i=0
    playerURL=rf"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=true&isGame=false&sort=%5B%7B%22property%22:%22goals%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start={i}&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C={start}{end}%20and%20seasonId%3E=19171918"
    print(requests.get(playerURL).json())
    for i in range(0,requests.get(playerURL).json()["total"],100):
        #print(i)
    #i=0
        playerURL=rf"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=true&isGame=false&sort=%5B%7B%22property%22:%22goals%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start={i}&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C={start}{end}%20and%20seasonId%3E=19171918"

        allPlayers=requests.get(playerURL).json()["data"]

        playersDf = pd.DataFrame(allPlayers)
        #print(playersDf.skaterFullName)
        for name in playersDf.skaterFullName:
            playerNames.append(name)

    # print(len(playerNames))
    # saves to pickle for use later
    pd.to_pickle(playerNames,"NHLNames.pkl")




# function to grab all players active within a specifc time period
# also returns games played
def getSpecificPlayers(startDate,endDate,positions=None):
# https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=true&isGame=true&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22goals%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22assists%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=100&cayenneExp=active%3D1%20and%20gameDate%3C=%22{endDate}%2023%3A59%3A59%22%20and%20gameDate%3E=%22{startDate}%22%20and%20gameTypeId=2
    req = requests.get(fr"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=true&isGame=true&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22goals%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22assists%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=100&cayenneExp=active%3D1%20and%20gameDate%3C=%22{endDate}%2023%3A59%3A59%22%20and%20gameDate%3E=%22{startDate}%22%20and%20gameTypeId=2")



    playerNames=[]
    gamesPlayed=[]
    for i in range(0,req.json()["total"],100):
        
        #print(i)
    #i=0
        playerURL=rf"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=true&isGame=true&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22goals%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22assists%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start={i}&limit=100&cayenneExp=active%3D1%20and%20gameDate%3C=%22{endDate}%2023%3A59%3A59%22%20and%20gameDate%3E=%22{startDate}%22%20and%20gameTypeId=2"

        allPlayers=requests.get(playerURL).json()["data"]

        playersDf = pd.DataFrame(allPlayers)
        if len(playersDf)==0:
            continue
        #print(playersDf.skaterFullName)
        ti=0
        #print(playersDf)
        for name in playersDf.skaterFullName:
            #print(name)
            if positions!=None:
                #print("SUP",ti)
                for pos in positions:
                    #print("x")
                    if playersDf.positionCode[ti] ==pos:
                       # print("y")
                        playerNames.append(name)
                        gamesPlayed.append(playersDf.gamesPlayed[ti])

                        break

            else:
                print(name)
                playerNames.append(name)
                gamesPlayed.append(playersDf.gamesPlayed[ti])
            ti+=1
    #print(playerNames)
    return playerNames,gamesPlayed

if st.button("Get newest Players"):
    getNewPlayers()

#players=["Alex Ovechkin","0x2","0x3"]
players=pd.read_pickle(r"NHLNames.pkl")
st.image(r"nhlLogo.png",width=250)
st.title("Similar Player Search")


selectedPlayers=st.multiselect("Select Player(s)",players)
today = datetime.datetime.now()

start = today.year -10

playerData=0
firstGame= datetime.date(start, 1, 1)

#perform any initiializion of values // get data for using in UI
if len(selectedPlayers) >0:
    firstGame,position=getfirstGame(selectedPlayers)





jan_1 = datetime.date(start, 1, 1)

dec_31 = datetime.date(start, 12, 31)

minDate=None
# stats we allow to compare on for now
attrOps=["assists","evGoals","evPoints","goals","otGoals","plusMinus","points","shootingPct","ppGoals","ppPoints","shGoals","shPoints","shots","faceoffWinPct","timeOnIcePerGame"]
 

att=st.multiselect("Comparison Stats",attrOps,default=attrOps)

statWeights=st.columns(len(att))
customWeights=[]
# get the weights for each sat
for lbs in range(0,len(statWeights)):
    with statWeights[lbs]:
        statLbs=st.number_input(f"Stat {lbs+1} weight",value=(1.0/len(statWeights)),min_value=0.01,max_value=1.0)
        customWeights.append(statLbs)


st.text(f"Weight Total (Must Sum to 1): {sum(customWeights)}")



date_range=st.date_input("Basis Range",(firstGame,today),min_value=firstGame,max_value=datetime.datetime.today())

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

col7,col8=st.columns(2)

with col7:

    positionCodes=st.checkbox("Return same Position Only",value=True)

with col8:

    st.markdown("",help="When selected, only players of the same position as your base player(s) will be analyzed")

posMap={
    "Center":"C",
    "Defenseman":"D",
    "Left Wing":"L",
    "Right Wing":"R"
}

if not positionCodes:
    tempPos=st.multiselect("Choose Position(s)",["Center","Defenseman","Left Wing","Right Wing"])
    position=[]
    for pos in tempPos:
        position.append(posMap[pos])





    
    # set position codes to be same as the base player
allDates=False
if isCustomRange:

    custom_date_range=st.date_input("Comparison Range",(jan_1,datetime.date(start,1,4)),min_value=minDate,max_value=datetime.datetime.today())
elif not matchDates:
    #add a search al feature
    custom_date_range=date_range
    allDates=True

else:
    custom_date_range=date_range
  
 

numMatches=st.number_input("How many similar players would you like to find?",min_value=1,max_value=25)

 

st.info('This is an experimental tool, all results should be taken with a grain of salt', icon="ℹ️")



similar=st.button("Search")
# kick of search function and display results
if similar:
    attribs=att
    #matches=findPlayers(isCum,selectedPlayers,date_range,custom_date_range,numMatches,allDates)
    matches=findAllPlayers(isCum,selectedPlayers,date_range,custom_date_range,numMatches,attribs,position,customWeights)
    try:
        st.text("Base Data")
        st.write(matches[1])
        st.text("Results")
        st.write(matches[0])
    except:
        pass
    st.balloons()

 



 
