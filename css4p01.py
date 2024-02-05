#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 18:27:44 2024

@author: user
"""
# importing the modules

import pandas as pd
import seaborn as sns
from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# importing the data by reading the csv file
df = pd.read_csv('movie_dataset.csv')

df

"""
(Extract,Tranform & Load)

"""
# Dataframe inspection 


print(df.shape)
print(df.dtypes)

df.isnull().sum()

df.isnull().sum(axis=1)

round(100*(df.isnull().sum()/len(df.index)), 2)

df = df.drop("Description", axis=1)
#df = df.drop("Runtime(Minutes)",axis=1)
#df = df.drop("Votes",axis=1)
df = df.drop("Metascore", axis=1)
round(100*(df.isnull().sum()/len(df.index)), 2)
# print(df.columns.tolist())

df["Revenue (Millions)"].fillna(
    int(df["Revenue (Millions)"].mean()), inplace=True)
print(df)

print(len(df.index))
print(round((len(df.index) / 5043)*100, 2))

""" 
Exploratory Data Analysis

"""
#What is the highest rated movie in the dataset?

movie_Title_Rating_List = ['The Dark Knight',
                           'Jason Bourne', 'Rogue One', 'Trolls']

Rating = [x[4]
          for x in movie_Title_Rating_List]

max(Rating)

# What is the average revenue of all movies in the dataset? 

avrg_revenue = df['Revenue (Millions)'].mean()
print(f"The average revenue of all movies in the dataset is: {avrg_revenue:.2f}")

# What is the average revenue of movies from 2015 to 2017 in the dataset?


revenue_of_movies = df[(df['Year'] >= 2015) & (df['Year'] <= 2017)]
avg_revenue = revenue_of_movies['Revenue (Millions)'].mean()
print(f"Average revenue of movies from 2015 to 2017: {avg_revenue:.2f}")

# How many movies were released in the year 2016?

year_2016_movies = df[df['Year'] == 2016]
number_of_year_2016_movies = len(year_2016_movies)
print(f"Number of movies released in 2016: {number_of_year_2016_movies}")

# How many movies were directed by Christopher Nolan?


christopher_nolan_movies = df[df['Director'] == 'Christopher Nolan']
number_of_christopher_nolan_movies = len(christopher_nolan_movies)
print(f"Number of movies directed by Christopher Nolan: {number_of_christopher_nolan_movies:.2f}")

# How many movies in the dataset have a rating of at least 8.0?

movies_with_rates_8 = df[df['Rating'] >= 8.0]
number_of_movies_with_rates_8 = len(movies_with_rates_8)
print(f"Number of movies with a rating of at least 8.0: {number_of_movies_with_rates_8:.2f}")


#What is the median rating of movies directed by Christopher Nolan?

christopher_nolan_movies = df[df['Director'] == 'Christopher Nolan']
median_rating_of_christopher_nolan_movies = christopher_nolan_movies['Rating'].median()
print(f"Median rating of movies directed by Christopher Nolan: {median_rating_of_christopher_nolan_movies:.2f}")

#Find the year with the highest average rating?

avrg_rating_by_year = df.groupby('Year')['Rating'].mean()
highest_avrg_rating_year = avrg_rating_by_year.idxmax()
print(f"Year with the highest average rating: {highest_avrg_rating_year}")

#What is the percentage increase in number of movies made between 2006 and 2016?

movies_made_between_2006_and_2016 = df[(df['Year'] >= 2006) & (df['Year'] <= 2016)]
num_movies_2006 = len(df[df['Year'] == 2006])
num_movies_2016 = len(df[df['Year'] == 2016])
percentage_increase = ((num_movies_2016 - num_movies_2006) / num_movies_2006) * 100
print(f"Percentage increase in the number of movies made between 2006 and 2016: {percentage_increase:.2f}%")

#Find the most common actor in all the movies?

actors = [actor for actors_list in df['Actors'].dropna() for actor in actors_list.split(', ')]
actor_counts = pd.Series(actors).value_counts()
most_common_actor = actor_counts.idxmax()
#actors = actor_df['Actors'].tolist()
print(f"The most common actor in all movies is: {most_common_actor}")


#How many unique genres are there in the dataset?

genres = [genre for genres_list in df['Genre'].dropna() for genre in genres_list.split(', ')]
unique_genres = set(genres)
unique_genres = len(unique_genres)
print(f"The number of unique genres in the dataset is: {unique_genres}")

#Do a correlation of the numerical features, what insights can you deduce? Mention at least 5 insights


#correlation_matrix=df.corr(method='pearson')
#sns.heatmap(correlation_matrix, annot=True)
#plt.title('Correlation Matrix for Numeric Features')
#plt.xlabel('Movie Features')
#plt.ylabel('Movie Features')
#plt.show()


corr = df.corr()
plt.figure(figsize = (15,8))
sns.heatmap(corr , annot=True , annot_kws= {'size':12})
plt.title("Correlation Heatmap - Movies" , fontsize = 20)
plt.xticks(fontsize = 15,rotation = 90)
plt.yticks(fontsize = 15)
plt.title('Correlation Matrix for Numeric Features',fontsize = 20)
plt.xlabel('Moviee_Features')
plt.ylabel('Movie_Features')
plt.show()


