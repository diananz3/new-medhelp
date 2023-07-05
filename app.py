# libraries
import random
import numpy as np
import pickle
import csv
import re
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
from flask import Flask, render_template, request, flash, redirect,url_for,session,logging
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
from flask_sqlalchemy import SQLAlchemy
from flask_mysqldb import MySQL
import MySQLdb.cursors


# chat initialization
best_model = load_model("model_best_cv.hdf5")
factory = StopWordRemoverFactory().get_stop_words()
more_stopword = ["permisi","saya", "aku", "kamu", "kita", "kak", "min", "apa"]

data = factory + more_stopword

dictionary = ArrayDictionary(data)
stop = StopWordRemover(dictionary)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

f = open('tokenizer_best_cv.pickle', 'rb')
tokenizer = pickle.load(f)

f = open('tags.pickle', 'rb')
tag = pickle.load(f)

f = open('slangs.pickle', 'rb')
slangs = pickle.load(f)


app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'medhelp'
 
mysql = MySQL(app)
app.secret_key = 'medhelp'

@app.route("/")
def form():
    return render_template("login.html")

@app.route("/try")
def index():
    return render_template("index.html")

@app.route("/login",methods=["GET", "POST"])
def login():
    if request.method == 'POST' and 'uname' in request.form and 'passw' in request.form:
        username = request.form['uname']
        password = request.form['passw']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE username = % s AND password = % s', (username, password, ))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['userid'] = user['userid']
            session['username'] = user['username']
            session['email'] = user['email']
            return redirect(url_for("index"))
    return render_template('login.html')

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        userName = request.form['username']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('INSERT INTO user VALUES (NULL, % s, % s, % s)',(userName, email, password))
        mysql.connection.commit()
        return redirect(url_for("login"))
    return render_template('register.html')
 

@app.route("/logout")
def logout():
    session.pop('loggedin', None)
    session.pop('userid', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route("/get", methods=["POST"])
def chatbot_response():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    text = request.form["msg"]
    userid = session['userid']
    pred = predict_class(text)
    score = confidence(text)

    cursor.execute('SELECT idtag FROM tags WHERE namatag = % s', [pred])
    predict = cursor.fetchone()
    id_pred = predict['idtag']

    if text in ('Ya', 'ya'):
        res = "Terima kasih sudah menggunakan MedHelp Puskesmas Keputih. Apakah ada pertanyaan lain?"
        return res
    elif text in ('Tidak', 'tidak'):
        last = getLastchat(userid)
        last2 = getLastchat2(userid)
        idlabel = getlabel(last)
        cursor.execute('SELECT * FROM tags WHERE idtag != %s AND idlabel = %s LIMIT 1', (last,idlabel))
        resp = cursor.fetchone()
        if resp is not None:
            newpred = resp['namatag']
            id_newpred = resp['idtag']
            if id_newpred == last2:
                res = "Mohon maaf bot belum bisa memberikan informasi yang Anda butuhkan. Silakan menghubungi Puskesmas Keputih pada nomor 082137498585 atau datang langsung ke puskesmas"
            else :            
                cursor.execute('INSERT INTO chat VALUES (NULL, %s, %s, NULL)',(userid, id_newpred))
                mysql.connection.commit()
                res = getJawaban(newpred)+"<br><br>Apakah jawaban tersebut membantu?(Ya/Tidak)"
        else:
            res = "Mohon maaf bot belum bisa memberikan informasi yang Anda butuhkan. Silakan menghubungi Puskesmas Keputih pada nomor 082137498585 atau datang langsung ke puskesmas"
        return res
    else :
        if score >= 0.5:
            cursor.execute('INSERT INTO chat VALUES (NULL, %s, %s, NULL)',(userid, id_pred))
            mysql.connection.commit()
            if pred == "greeting" or pred == "closing":
                res = getJawaban(pred)
            else:
                res = getJawaban(pred)+"<br><br>Apakah jawaban tersebut membantu?(Ya/Tidak)"
            return res
        else:
            res = "Mohon maaf, bot tidak bisa memberikan informasi tentang hal tersebut. Apakah ada pertanyaan lain?"
            return res

# chat functionalities
def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    return text_nopunct

def change_slang(text):
    res = ' '.join([slangs.get(i, i) for i in text.split()])
    return res

def stopwords_remover(text) :
    text = stop.remove(text)
    return text

def stem(text) :
    text = stemmer.stem(text)
    return text


def preprocess(text) :
    text = remove_punct(text)
    text = text.lower()
    text = change_slang(text)
    text = stopwords_remover(text)
    text = stem(text)
    # text = remove_number(text)
    return text

def text2vec(text) :
    sequences = tokenizer.texts_to_sequences([preprocess(text)])
    data = pad_sequences(sequences, maxlen=20)
    return data

def predict_class(text) :
    vec = text2vec(text)
    pred = best_model.predict(vec)
    y_pred = np.argmax(pred, axis=-1)
    return tag[y_pred[0]]

def confidence(text) :
    vec = text2vec(text)
    pred = best_model.predict(vec)
    y_pred = np.max(pred)
    return y_pred

# def list_predict_class(text) :
#     vec = text2vec(text)
#     pred = best_model.predict(vec)
#     list_class_prob = dict()
#     for i in enumerate(pred[0]) :
#         list_class_prob[tag[i[0]]] = i[1]
#     return list_class_prob

# def getResponse(pred):
#     cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
#     cursor.execute('SELECT response FROM tags WHERE namatag = % s', [pred])
#     resp = cursor.fetchone()
#     res = resp['response']
#     return res
        
def getResponse(pred):
   with open('responses.csv') as f:
     reader = csv.reader(f, delimiter=',')
     for row in reader:
        if  row[1] == pred:
             return row[2]
        
def getJawaban(pred):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM responses LEFT JOIN tags ON responses.id_tag = tags.idtag WHERE tags.namatag = % s', [pred])
    pred = cursor.fetchone()
    resp = pred['resp']
    return resp
         
def getLastchat(userid):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT pred FROM chat WHERE id_user = %s ORDER BY date DESC LIMIT 1', [userid])
    pred = cursor.fetchone()
    id_pred = pred['pred']
    return id_pred

def getLastchat2(userid):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT pred FROM chat WHERE id_user = %s ORDER BY date DESC LIMIT 1,1', [userid])
    pred = cursor.fetchone()
    id_pred = pred['pred']
    return id_pred

def getlabel(pred):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT idlabel FROM tags WHERE idtag = % s', [pred])
    label = cursor.fetchone()
    idlabel = label['idlabel']
    return idlabel

if __name__ == "__main__":
    app.run(debug=True)

