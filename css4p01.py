# -*- coding: utf-8 -*-

"""
Created on Mon Feb  5 08:10:57 2024

@author: Keketso
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('movie_dataset.csv')
print(df)
"""
     Rank                    Title  ... Revenue (Millions) Metascore
0       1  Guardians of the Galaxy  ...             333.13      76.0
1       2               Prometheus  ...             126.46      65.0
2       3                    Split  ...             138.12      62.0
3       4                     Sing  ...             270.32      59.0
4       5            Suicide Squad  ...             325.02      40.0
..    ...                      ...  ...                ...       ...
995   996     Secret in Their Eyes  ...                NaN      45.0
996   997          Hostel: Part II  ...              17.54      46.0
997   998   Step Up 2: The Streets  ...              58.01      50.0
998   999             Search Party  ...                NaN      22.0
999  1000               Nine Lives  ...              19.64      11.0

[1000 rows x 12 columns]
"""


# Renaming the columns
# print(df.rename({'A': 'Col_1'}, axis='columns'))
print(df.rename({'Revenue (Millions)': 'Revenue'}, axis='columns'))
print(df.rename({'Runtime (Minutes)': 'Runtime'}, axis='columns'))

# Question 1: What's the highest rated movie?

# df.loc['Rogue One':]
# print(df)

# search_value = 'Rogue One'
# result = df[df.eq('Rogue One').any(axis=1)]
# result
# search_value = 'Trolls'
# result = df[df.eq('Trolls').any(axis=1)]
# result
# search_value = 'Jason Bourne'
# result = df[df.eq('Jason Bourne').any(axis=1)]
# result
# search_value = 'The Dark Knight'
# result = df[df.eq('The Dark Knight').any(axis=1)]
# result

# ANSWER: Question 1: Rogue One


# Question 2: What is the average revenue of all movies in the dataset?

#Average the NANs
x1 = df["Metascore"].mean()
df["Metascore"].fillna(x1, inplace = True)
print(df)
"""
58.98504273504273
"""

#Question 2
x2 = df["Revenue (Millions)"].mean()
df["Revenue (Millions)"].fillna(x2, inplace = True)
print(df)
"""
82.95637614678898
"""
#ANSWER Question 2: 70 - 100 Million
print(df.rename({'Revenue (Millions)': 'Revenue'}, axis='columns'))
x = df["Revenue (Millions)"].mean()
df["Revenue (Millions)"].fillna(x, inplace = True) 
print(df)

print(df.info())
print(df.describe())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 12 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   Rank                1000 non-null   int64  
 1   Title               1000 non-null   object 
 2   Genre               1000 non-null   object 
 3   Description         1000 non-null   object 
 4   Director            1000 non-null   object 
 5   Actors              1000 non-null   object 
 6   Year                1000 non-null   int64  
 7   Runtime (Minutes)   1000 non-null   int64  
 8   Rating              1000 non-null   float64
 9   Votes               1000 non-null   int64  
 10  Revenue (Millions)  1000 non-null   float64
 11  Metascore           936 non-null    float64
dtypes: float64(3), int64(4), object(5)
memory usage: 93.9+ KB
None
              Rank         Year  ...  Revenue (Millions)   Metascore
count  1000.000000  1000.000000  ...         1000.000000  936.000000
mean    500.500000  2012.783000  ...           82.956376   58.985043
std     288.819436     3.205962  ...           96.412043   17.194757
min       1.000000  2006.000000  ...            0.000000   11.000000
25%     250.750000  2010.000000  ...           17.442500   47.000000
50%     500.500000  2014.000000  ...           60.375000   59.500000
75%     750.250000  2016.000000  ...           99.177500   72.000000
max    1000.000000  2016.000000  ...          936.630000  100.000000

[8 rows x 7 columns]
"""
#ANSWER Quetion 2: 70-100 million

#Question 3
# x3 = df["Revenue (Millions)"].mean()
# df["Revenue (Millions)"].fillna(x3, inplace = True)
# print(df)
year_rev = (df[df['Revenue (Millions)'].notnull()][['Year','Revenue (Millions)']].groupby('Year').mean())
# year_rev.plot(Figsize=(10))
"""
Year	Revenue (Millions)
2006	86.1448352793995
2007	87.51048121862559
2008	98.77262261820748
2009	110.27618636445403
2010	103.97531880733945
2011	87.53835517693315
2012	107.97328125
2013	86.98449591692712
2014	84.99209698558322
2015	78.8622776854728
2016	63.44658789732184
"""
data_2015_2017 = [78.8622776854728, 63.44658789732184, 0]
print(len(data_2015_2017))
print(sum(data_2015_2017))
average = sum(data_2015_2017)/len(data_2015_2017)
print(average)
"""
3
142.30886558279465
47.436288527598215
"""
#ANSWER Question 3: 50-80 Million


#Question 5
search_value = 'Christopher Nolan'
result = df[df.eq('Christopher Nolan').any(axis=1)]
result

#ANSWER Question 5: 5 movies

# #Question 7
import statistics
print(statistics.median([8.6, 9, 8.5, 8.8, 8.5]))
"""
8.6
"""
#ANSWER Question 7: 8.6

# #Question 6:
# df["Calories"].fillna(x, inplace = True) 
# median(Chris_ratings)

# # # df = df.query("Rating == 10")
# # # print(df["Title"].value_counts().head(10))
df = df.query("Rating == 8")
print(df["Title"].value_counts().head(8))
df = df.query("Rating == 8.1")
print(df["Title"].value_counts().head(8.1))
df = df.query("Rating == 8.2")
print(df["Title"].value_counts().head(8.2))
df = df.query("Rating == 8.3")
print(df["Title"].value_counts().head(8.3))
df = df.query("Rating == 8.4")
print(df["Title"].value_counts().head(8.4))
df = df.query("Rating == 8.5")
print(df["Title"].value_counts().head(8.5))
df = df.query("Rating == 8.6")
print(df["Title"].value_counts().head(8.6))
df = df.query("Rating == 8.7")
print(df["Title"].value_counts().head(8.7))
df = df.query("Rating == 8.8")
print(df["Title"].value_counts().head(8.8))
df = df.query("Rating == 8.9")
print(df["Title"].value_counts().head(8.9))
df = df.query("Rating == 9")
print(df["Title"].value_counts().head(9))
df = df.query("Rating == 9.1")
print(df["Title"].value_counts().head(9.1))
df = df.query("Rating == 9.2")
print(df["Title"].value_counts().head(9.2))
df = df.query("Rating == 9.3")
print(df["Title"].value_counts().head(9.3))
df = df.query("Rating == 9.4")
print(df["Title"].value_counts().head(9.4))
df = df.query("Rating == 9.5")
print(df["Title"].value_counts().head(9.5))
df = df.query("Rating == 9.6")
print(df["Title"].value_counts().head(9.6))
df = df.query("Rating == 9.7")
print(df["Title"].value_counts().head(9.7))
df = df.query("Rating == 9.8")
print(df["Title"].value_counts().head(9.8))
df = df.query("Rating == 9.9")
print(df["Title"].value_counts().head(9.9))
df = df.query("Rating == 10")
print(df["Title"].value_counts().head(10))


"""
8: 19 rows, 12 colums
8.1: 26, 12
8.2: 10, 12
8.3: 7, 12
8.4: 4, 12
8.5: 6, 12
8.6: 3, 12
8.7: 0, 12
8.8: 2, 12
8.9: 0, 12
9: 1, 12
9.1 to 10: 0, 12
"""
#ANSWER Question 6: 78

#Question 10
def repp(string):
    return string.replace("[","").replace("]","").replace("u'","").replace("',",",")[:-1]

movies_series = df['Actors'].apply(repp)

actors_list = []
for movie_actors in movies_series:
    actors_list.append([e.strip() for e in movie_actors.split(',')])

actor_dict = {}

for actor in actors_list:
    for a in actor:
        if a in actor_dict:
            actor_dict[a] +=1
        else:
            actor_dict[a] = 1

actor_dict
"""
Answer is in Variable Explorere under the name actor_dict
"""
#ANSWER Question 10: Mark Wahlberg with 14 inputs to their name.

#Question 4, 8 and 9
all_year = df['Year'].value_counts()
print(all_year)

all_years2 = all_year[all_year >= 10]
all_years2
"""
Year
2016    297
2015    127
2014     98
2013     91
2012     64
2011     63
2010     60
2007     53
2008     52
2009     51
2006     44
Name: count, dtype: int64
"""
#ANSWER Question 4: 297 
#ANSWER Question 8: 2016
#def percent_change(old, new):
pc = round((297 - 44) / abs(44) * 100, 2)
print(f"from 44 to 296   -> {pc}% change")
"""
from 44 to 296   -> 575.0% change
"""
#ANSWER Question 9: 575.0% change

#Question 12




#Question 13


# 1.
# 2.
# 3.
# 4.
# 5
