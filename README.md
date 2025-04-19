# NHL-Similar-Player-Search-Engine
This is a comprehesive search engine to find n most similar players compared to a given input player or base player as refered to in the code

The Search Engine allows you to input a player and you can choose the search time frame for this player and then choose the time frame to search through 
in order to find n most similar players

The searches can be done in two main forms:

 - Distance based cumulative searching: Get some stats within a given date range and find who's sum was closest to that
     - can be multivariate as in you can use multiple different stats for the comparison and you can change the weights / importance for each stat in the euclidean distance computation
  
 - Time Series Comparision : A filter is applied to chosen stats to analyze to create an interpretable time series eg rolling sum from a 10 game window, then the time series shape is compared to other players to find n most similar players
     - there is both DTW and motif searching with the stumpy library
     - only works for comparing 1 stat at a time but plans to make it multivariate are in the works
