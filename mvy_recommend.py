import numpy as np
import pandas as pd

#for large datasets default print behavior in many environments (including Jupyter Notebooks) will truncate the output for readability
# Set the print options to display all elements
np.set_printoptions(threshold=np.inf)

movies = pd.read_csv('C://Users//MANASWINI KARNATAKA//Downloads//tmdb_5000_movies.csv')
credits=pd.read_csv('C://Users//MANASWINI KARNATAKA//Downloads//tmdb_5000_credits.csv')

movies.head(1)
movies.shape
credits.head(1)
credits.shape

#merging both datasets based on title
movies=movies.merge(credits,on='title')  
movies.head()  

movies.info() 
movies['original_language'].value_counts() 


#as eng mvy count is very much greater we are not considering lang column
#sometimes we can even consider the following
#popularity ,production_companies,release_date
#considering the following columns only
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.head()

#we should create a dataframe for preprocessing so that it contains
# id mvy_name tags
# tag is the combination of ['overview','genres','keywords','cast','crew']
movies.isnull().sum()

#we have 3 overview nulls so remove them
movies.dropna(inplace=True)
movies.isnull().sum()

movies.duplicated().sum()

movies.iloc[0].genres
#changing the format of genres to [Action,Adven,Fanctasy,....] i.e from dict to list
import ast  #converts the ----- to lists
def convert(text):
    l=[]
    for i in ast.literal_eval(text):
        l.append(i['name'])
    return l     



movies['genres'].apply(convert)

movies['genres']=movies['genres'].apply(convert)
# do the same for keywords
movies['keywords']=movies['keywords'].apply(convert)

#for cast lets consider the first 3 cast
def convert_cast(text):
    l=[]
    cnt=0
    for i in ast.literal_eval(text):
        if cnt<3:
            l.append(i['name'])
            cnt+=1
        else:
            break
    return l       
        
  
movies['cast'].apply(convert_cast)       

movies['cast']=movies['cast'].apply(convert_cast)
movies['crew'][0]
#for crew lets only consider director
def fetch_crew(text):
    l=[]
    for i in ast.literal_eval(text):
        if i['job']=='Director':
            l.append(i['job'])
            break
    return l        


movies['crew']=movies['crew'].apply(fetch_crew)
movies.head()

movies['overview'] #overview is str
#covert overview into list 
movies['overview']=movies['overview'].apply(lambda x:x.split())

#removing spaces btw words  for example : 
#sam worthingtan and sam mendes ---> sammendes 
#science is diffirent and sciencefiction is different
def collapse(text):
    l = []
    for i in text:
        l.append(i.replace(" ",""))
    return l

movies['genres']=movies['genres'].apply(collapse)
movies['keywords']=movies['keywords'].apply(collapse)
movies['cast']=movies['cast'].apply(collapse)
movies['crew']=movies['crew'].apply(collapse)

#can also apply lambda function  like this:
#movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies.head()

#creating tags by concatination 
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

#creating new dataframe
#new_df=movies[['movie_id','title','tags']]
# or can do like this

new_df=movies.drop(columns=['overview','genres','keywords','cast','crew'])

new_df.head()

new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

new_df.head()




import nltk
from nltk.stem import PorterStemmer


# Download the NLTK data for the first time use
nltk.download('punkt')
# Initialize the PorterStemmer
ps = PorterStemmer()
# Define the stem function
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# Test the function
print(ps.stem('loving'))  # Output: 'love'
print(ps.stem('loved'))   # Output: 'love'
print(ps.stem('love'))    # Output: 'love'

# Example usage with a sentence
example_text = "loving loved love"
stemmed_text = stem(example_text)
print(stemmed_text)  # Output: 'love love love'

#apply stem to tags
new_df['tags']=new_df['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(new_df['tags']).toarray()
vector.shape


from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vector)
similarity[0]

#we have to arrage them in an order
#if we sort we will lose the mvy_id
#so we use enumeration
#output :[( 0th mvy, distance of 0th mvy to 0th mvy),
#         (1st mvy,distance of 0th mvy to 1st mvy),.......]
#apply sort based on distance then take the first 5 values as most similar ones

sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distance=similarity[movie_index]
    mvy_list=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]

    for i in mvy_list:
        print(new_df.iloc[i[0]].title)


new_df[new_df['title'] == 'The Lego Movie'].index[0]

recommend('Avatar')


import pickle
#pickle.dump(new_df,open('movie_list.pkl','wb'))
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))

#rather than sending dataframe i will send dict
#

#pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))

