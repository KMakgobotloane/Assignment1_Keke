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


#Question 12 and 13

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('movie_dataset.csv')

pd.set_option('display.max_rows',None)

print(df)

x = df["Revenue (Millions)"].mean()

df["Revenue (Millions)"].fillna(x, inplace = True)

x2 = df["Metascore"].mean()

df["Metascore"].fillna(x2, inplace = True)

plt.figure(figsize=(2, 7))
sns.countplot(data=df, y='Genre', order=df['Genre'].value_counts().index, palette='viridis')
plt.xlabel('Count', fontsize=2, fontweight='bold')
plt.ylabel('Genre', fontsize=2, fontweight='bold')

plt.figure(figsize=(2, 7))
counts = df['Genre'].value_counts()
sns.barplot(x=counts.index, y=counts, palette='viridis')
plt.xlabel('Genre', fontsize=2, fontweight='bold')
plt.ylabel('Count', fontsize=2, fontweight='bold')
plt.title('Distribution of Genres', fontsize=2, fontweight='bold')
plt.xticks(rotation=90, fontsize=2, fontweight='bold')
plt.show()

"""
Genre	count
Action,Adventure,Sci-Fi	50
Drama	48
Comedy,Drama,Romance	35
Comedy	32
Drama,Romance	31
Animation,Adventure,Comedy	27
Action,Adventure,Fantasy	27
Comedy,Drama	27
Comedy,Romance	26
Crime,Drama,Thriller	24
Crime,Drama,Mystery	23
Action,Adventure,Drama	18
Action,Crime,Drama	17
Horror,Thriller	16
Drama,Thriller	15
Adventure,Family,Fantasy	14
Biography,Drama,History	14
Action,Adventure,Comedy	14
Biography,Drama	14
Action,Comedy,Crime	12
Action,Crime,Thriller	12
Action,Adventure,Thriller	11
Horror	11
Crime,Drama	10
Biography,Crime,Drama	9
Thriller	9
Horror,Mystery,Thriller	9
Action,Thriller	9
Animation,Action,Adventure	9
Biography,Drama,Sport	8
Adventure,Comedy,Drama	8
Drama,Mystery,Thriller	8
Action,Biography,Drama	8
Mystery,Thriller	7
Biography,Comedy,Drama	7
Comedy,Crime,Drama	7
Action,Drama,Thriller	7
Action,Sci-Fi,Thriller	7
Drama,Mystery,Romance	6
Action,Adventure,Crime	6
Horror,Mystery	6
Action,Comedy	6
Drama,Horror,Thriller	6
Drama,Fantasy,Horror	6
Drama,Mystery,Sci-Fi	6
Drama,Horror,Sci-Fi	6
Adventure,Drama,Fantasy	6
Drama,Horror	5
Comedy,Crime	5
Drama,Fantasy,Romance	5
Biography,Drama,Romance	5
Action,Horror,Sci-Fi	5
Action,Adventure,Mystery	5
Drama,Sport	5
Comedy,Drama,Music	5
Comedy,Family	5
Animation,Comedy,Family	5
Drama,Romance,Sci-Fi	4
Action,Drama,Sci-Fi	4
Adventure,Biography,Drama	4
Action,Adventure,Family	4
Drama,Sci-Fi,Thriller	4
Action,Drama,History	4
Crime,Horror,Thriller	4
Adventure,Drama,Family	4
Drama,Horror,Mystery	4
Drama,History,Thriller	4
Action,Drama,Fantasy	3
Mystery,Sci-Fi,Thriller	3
Action,Sci-Fi	3
Action,Comedy,Fantasy	3
Crime,Drama,Horror	3
Horror,Sci-Fi,Thriller	3
Comedy,Drama,Family	3
Adventure,Drama,Thriller	3
Action,Mystery,Thriller	3
Adventure,Fantasy	3
Biography,Drama,Thriller	3
Drama,War	3
Adventure,Sci-Fi,Thriller	3
Action,Horror,Thriller	3
Action,Mystery,Sci-Fi	3
Drama,Romance,Thriller	3
Action,Fantasy,Horror	3
Drama,Music	3
Comedy,Fantasy,Horror	3
Action,Crime,Mystery	3
Action,Adventure	3
Action,Adventure,Western	2
Drama,Sci-Fi	2
Adventure,Drama,Sci-Fi	2
Action,Comedy,Horror	2
Adventure,Comedy,Family	2
Drama,Music,Romance	2
Comedy,Horror,Thriller	2
Adventure,Drama,Romance	2
Comedy,Horror	2
Comedy,Fantasy	2
Comedy,Drama,Fantasy	2
Comedy,Drama,Horror	2
Comedy,Music	2
Action,Adventure,Biography	2
Action,Adventure,Horror	2
Drama,History	2
Adventure,Mystery,Sci-Fi	2
Action,Adventure,Romance	2
Action	2
Adventure,Comedy,Romance	2
Action,Comedy,Romance	2
Action,Drama,Mystery	2
Crime,Mystery,Thriller	2
Fantasy,Horror	2
Adventure,Comedy,Sci-Fi	2
Animation,Adventure,Family	2
Adventure,Horror	2
Adventure,Comedy	2
Action,Drama,Family	2
Sci-Fi	2
Drama,Fantasy	2
Crime,Drama,History	2
Action,Drama,Romance	2
Action,Drama,Sport	2
Drama,Horror,Musical	1
Adventure,Drama,Western	1
Drama,Fantasy,Mystery	1
Fantasy,Horror,Thriller	1
Drama,Mystery,War	1
Drama,Thriller,War	1
Adventure,Drama	1
Romance,Sci-Fi,Thriller	1
Comedy,Romance,Western	1
Animation,Fantasy	1
Drama,Family	1
Adventure,Biography	1
Comedy,Crime,Thriller	1
Animation,Comedy,Drama	1
Fantasy,Mystery,Thriller	1
Adventure,Biography,Crime	1
Action,Comedy,Sport	1
Comedy,Drama,Musical	1
Drama,Fantasy,Music	1
Crime,Drama,Fantasy	1
Comedy,Family,Romance	1
Comedy,Western	1
Biography,Drama,Family	1
Drama,Fantasy,Musical	1
Adventure,Family	1
Comedy,Horror,Sci-Fi	1
Crime,Thriller	1
Action,Crime,Sci-Fi	1
Adventure,Comedy,Fantasy	1
Adventure	1
Action,Thriller,War	1
Comedy,Horror,Romance	1
Animation,Drama,Romance	1
Biography,Drama,Mystery	1
Drama,Family,Music	1
Mystery,Romance,Thriller	1
Adventure,Fantasy,Mystery	1
Drama,Romance,War	1
Action,Comedy,Sci-Fi	1
Drama,History,War	1
Action,Drama,War	1
Comedy,Drama,Thriller	1
Mystery,Romance,Sci-Fi	1
Action,Drama,Horror	1
Action,Fantasy	1
Drama,Musical,Romance	1
Drama,Fantasy,War	1
Drama,Family,Fantasy	1
Thriller,War	1
Action,Comedy,Mystery	1
Drama,Western	1
Mystery,Thriller,Western	1
Comedy,Family,Musical	1
Adventure,Crime,Mystery	1
Action,Fantasy,War	1
Romance,Sci-Fi	1
Horror,Mystery,Sci-Fi	1
Animation,Drama,Fantasy	1
Biography,Comedy,Crime	1
Adventure,Drama,War	1
Adventure,Drama,History	1
Action,Comedy,Drama	1
Comedy,Music,Romance	1
Action,Horror	1
Crime,Drama,Music	1
Action,Horror,Mystery	1
Drama,Fantasy,Thriller	1
Animation,Action,Comedy	1
Comedy,Romance,Sport	1
Action,Fantasy,Thriller	1
Biography,History,Thriller	1
Adventure,Drama,Horror	1
Sci-Fi,Thriller	1
Action,Crime,Sport	1
Adventure,Horror,Mystery	1
Comedy,Fantasy,Romance	1
Animation,Family,Fantasy	1
Action,Horror,Romance	1
Action,Biography,Crime	1
Comedy,Sci-Fi	1
Action,Comedy,Family	1
Action,Crime,Fantasy	1
Comedy,Mystery	1
Adventure,Comedy,Horror	1
Comedy,Family,Fantasy	1
"""
# ANSWER Q12: The series has a size of 207, as seen on the Variable Explorer.


# 1.
# 2.
# 3.
# 4.
# 5
