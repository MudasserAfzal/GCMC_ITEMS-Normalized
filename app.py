from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


# load the model from disk
# filename = 'knnPickle_file.pkl'
# clf = pickle.load(open(filename, 'rb'))
# cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    # Fit Nearest Neighbor using Cosine similarity on the user feature matrix
    user_features = pd.read_csv('df_4.csv')
    user_features.drop(['Unnamed: 0', 'STATE'], inplace=True, axis=1)
    scaler = MinMaxScaler()
    user_features = scaler.fit_transform(user_features)
  
    complete_df = pd.read_csv('rating matrix norm GCMC+feat with items.csv')
    complete_df.drop(['Unnamed: 0'], inplace=True, axis=1)
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(user_features)

    if request.method == 'POST':     
        message1 = request.form['open_wearing_0']
        message2 = request.form['open_wearing_1']
        message3 = request.form['open_wearing_2']
        message4 = request.form['open_wearing_3']
        message5 = request.form['open_wearing_4']
        message6 = request.form['open_wearing_5']
        message7 = request.form['age']
        message8 = request.form['living']
        message9 = request.form['gender']
        
        data = [message1, message2, message3, message4, message5, message6, message7, message8, message9]
        data = scaler.transform(np.array(data).reshape(1, -1))
        print('-----', data)

        n = 1
        distances, indices = model_knn.kneighbors(np.array(data).reshape(1, -1), n_neighbors = n)
        itm_ = []
        for i, d in enumerate(distances[0]):
          # print(f'For test user {new_user.userId.values[0]} similar user {indices[0][i]} has distance is {d}')
          top = 5
          print('Closeset user is with index ',indices[0][i])
          # print('Similar user had feature vector',user_features.iloc[indices[0][i]].to_numpy())
          # print('Our test feature vector is',data)
          itm_.extend(complete_df.iloc[indices[0][i]].nlargest(top, keep='first').index.values)
        prediction = itm_
        new_pred = []
        print(itm_)
        dic = {'0': 'Bearded Goat' , '1':'Western Rise' , '2': 'KOTN', '3': 'johnnie-O' , '4': 'XTRATUF' , "5": 'Cotopaxi', '6': "Roark", '7': 'Taylor Stitch',     # Branch 1
               '8': 'Nasty Gal', '9': 'Birddogs', '10': 'John Elliott', '11': 'tentree' , '12': 'Marine Layer', '13': 'Rhone' , '14': 'Slowtide', '15': 'Fabletics' ,    # Branch 2
               '16': 'The North Face', '17': 'L.L. Bean' , '18': 'Girlfriend Collective', '19': 'Obey' , '20': 'Southern Marsh', '21': 'Mink Pink' , '22': 'The Row', '23': 'Salty Crew' ,     # Branch 3
               '24': 'Outdoor Voices', '25': 'Tommy Bahama' , '26': 'Faherty',  '27': 'Patagonia' , '28': 'Salt Life',  '29': 'OnlyNY', '30': 'Free People', '31': 'Ivory Ella'}     # Branch 4
            
        for i in range(len(itm_)):
            if itm_[i] in dic.keys():
                new_pred.append(dic[itm_[i]])
            else:
                new_pred.append(itm_[i])
              
    return render_template('result.html',prediction = new_pred)


if __name__ == '__main__':
	app.run(debug=True)
    