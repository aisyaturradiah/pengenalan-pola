import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import nltk
nltk.download('punkt')
import string 
import re
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
nltk.download('stopwords')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
import pickle
import ast
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from gensim.models import Word2Vec
import time
import seaborn as sns

st.title("Informatika Pariwisata A ")
st.write("### Dosen Pengampu : Cucun Very Angkoso, S.T., MT.")
st.write("##### Kelompok 3")
st.write("##### R.Bella Aprilia Damayanti - 200411100082")
st.write("##### Aisyatur Radiah - 200411100014 ")


#Navbar
data_set_description, preprocessing, modeling, implementation = st.tabs(["Data Set Description", "Preprocessing", "Modeling", "Implementation"])
dataset = pd.read_csv("https://raw.githubusercontent.com/RBellaApriliaDamayanti22/projectt/main/Data_Toroan.xlsx%20-%20Sheet1%20(1).csv")

#data_set_description
with data_set_description:
    st.write("###### Judul : Analisis Sentiment Review Tempat Pariwisata dengan Automated Lexicon Word2Vec dan Naive Bayes ")
    st.write("""###### Penjelasan Prepocessing Data : """)
    st.write("""1. Cleaning :
    
    Cleaning adalah Merupakan proses untuk menghilangkan tanda baca, simbol yang tidak diperlukan, dan spasi yang lebih dari satu pada suatu kalimat.
    """)
    st.write("""2. Case Folding :
    
    Case folding adalah proses dalam pemrosesan teks yang mengubah semua huruf dalam teks menjadi huruf kecil atau huruf besar. Tujuan dari case folding adalah untuk mengurangi variasi yang disebabkan oleh perbedaan huruf besar dan kecil dalam teks, sehingga mempermudah pemrosesan teks secara konsisten.
    
    Dalam case folding, biasanya semua huruf dalam teks dikonversi menjadi huruf kecil dengan menggunakan metode seperti lowercasing. Dengan demikian, perbedaan antara huruf besar dan huruf kecil tidak lagi diperhatikan dalam analisis teks, sehingga memungkinkan untuk mendapatkan hasil yang lebih konsisten dan mengurangi kompleksitas dalam pemrosesan teks.
    """)
    st.write("""3. Tokenize :

    Tokenisasi adalah proses pemisahan teks menjadi unit-unit yang lebih kecil yang disebut token. Token dapat berupa kata, frasa, atau simbol lainnya, tergantung pada tujuan dan aturan tokenisasi yang digunakan.

    Tujuan utama tokenisasi dalam pemrosesan bahasa alami (Natural Language Processing/NLP) adalah untuk memecah teks menjadi unit-unit yang lebih kecil agar dapat diolah lebih lanjut, misalnya dalam analisis teks, pembentukan model bahasa, atau klasifikasi teks.
    """)
    st.write("""4. word normalization :

    kata berulang yang tidak dilengkapi tanda baca '-' menjadi satu kesatuan. 
    """)
    st.write("""5. Filtering (Stopword Removal) :

    Filtering atau Stopword Removal adalah proses penghapusan kata-kata yang dianggap tidak memiliki makna atau kontribusi yang signifikan dalam analisis teks. Kata-kata tersebut disebut sebagai stop words atau stopwords.

    Stopwords biasanya terdiri dari kata-kata umum seperti “a”, “an”, “the”, “is”, “in”, “on”, “and”, “or”, dll. Kata-kata ini sering muncul dalam teks namun memiliki sedikit kontribusi dalam pemahaman konten atau pengambilan informasi penting dari teks.

    Tujuan dari Filtering atau Stopword Removal adalah untuk membersihkan teks dari kata-kata yang tidak penting sehingga fokus dapat diarahkan pada kata-kata kunci yang lebih informatif dalam analisis teks. Dengan menghapus stopwords, kita dapat mengurangi dimensi data, meningkatkan efisiensi pemrosesan, dan memperbaiki kualitas hasil analisis.
    """)
    st.write("""6. Stemming :

    Stemming adalah proses mengubah kata ke dalam bentuk dasarnya atau bentuk kata yang lebih sederhana, yang disebut sebagai “stem”. Stemming bertujuan untuk menghapus infleksi atau imbuhan pada kata sehingga kata-kata yang memiliki akar kata yang sama dapat diidentifikasi sebagai bentuk yang setara.
    """)
    
   
    st.write("###### Aplikasi ini untuk : ")
    st.write("""Analisis Sentiment Review Tempat Pariwisata dengan Automated Lexicon Word2Vec dan Naive Bayes """)
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : https://github.com/RBellaApriliaDamayanti22/projectt")
    st.write(dataset)

#Uploud data
# with upload_data:
#     uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
#     for uploaded_file in uploaded_files:
#         df = pd.read_csv(uploaded_file)
#         st.write("Nama File Anda = ", uploaded_file.name)
#         # view dataset asli
#         st.header("Dataset")
#         st.dataframe(df)

with preprocessing:
    #dataset = pd.read_csv("https://raw.githubusercontent.com/RBellaApriliaDamayanti22/projectt/main/Data%20toroan.csv")
    st.subheader("Preprocessing Data")
    df = pd.read_csv("https://raw.githubusercontent.com/RBellaApriliaDamayanti22/projectt/main/Data_Toroan.xlsx%20-%20Sheet1%20(1).csv")
    df

    st.write("Symbol & Punctuation Removal")
    def remove_numbers (text):
        return re.sub(r"\d+", "", text)
    df['Ulasan'] = df['Ulasan'].apply(remove_numbers)
    df['Ulasan']
    
    def remove_text_special (text):
        text = text.replace('\\t',"").replace('\\n',"").replace('\\u',"").replace('\\',"")
        text = text.encode('ascii', 'replace').decode('ascii')
        return text.replace("http://"," ").replace("https://", " ")
    df['Ulasan'] = df['Ulasan'].apply(remove_text_special)
    print(df['Ulasan'])

    #menghilangkan tanda baca
    def remove_tanda_baca(text):
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
        return text

    df['Ulasan'] = df['Ulasan'].apply(remove_tanda_baca)
    df.head(20)

    st.write("Case Folding:")
    def casefolding(Comment):
        Comment = Comment.lower()
        Comment = Comment.strip(" ")
        return Comment
    df['Ulasan'] = df['Ulasan'].apply(casefolding)
    df['Ulasan']
    
    st.write("Tokenize:")
    #proses tokenisasi
    from nltk.tokenize import TweetTokenizer
    def word_tokenize(text):
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        return tokenizer.tokenize(text)

    df['review_token'] = df['Ulasan'].apply(word_tokenize)
    df['review_token']

    st.write("Word Normalization")
    #Normalisasi kata tidak baku
    normalize = pd.read_csv("https://raw.githubusercontent.com/RBellaApriliaDamayanti22/projectt/main/Normalization%20Data.csv", sep=';')

    normalize_word_dict = {}

    for index, row in normalize.iterrows():
        if row[0] not in normalize_word_dict:
            normalize_word_dict[row[0]] = row[1]

    def normalized_term(comment):
        return [normalize_word_dict[term] if term in normalize_word_dict else term for term in comment]

    df['comment_normalize'] = df['review_token'].apply(normalized_term)
    df['comment_normalize']

    st.write("Stopword Removal")
    #Stopword Removal
    txt_stopwords = stopwords.words('indonesian')

    def stopwords_removal(filtering) :
        filtering = [word for word in filtering if word not in txt_stopwords]
        return filtering

    df['stopwords_removal'] = df['comment_normalize'].apply(stopwords_removal)
    df['stopwords_removal'].head(20)

    #stopword removal 2
    data_stopwords = pd.read_csv("https://raw.githubusercontent.com/RBellaApriliaDamayanti22/projectt/main/list_stopwords.csv")
    print(data_stopwords)

    def stopwords_removal2(filter) :
        filter = [word for word in filter if word not in data_stopwords]
        return filter

    df['stopwords_removal_final'] = df['stopwords_removal'].apply(stopwords_removal2)
    df['stopwords_removal_final']
    

    st.write("Stemming:")
    #proses stem
    # from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemming (term):
        return stemmer.stem(term)

    term_dict = {}

    for document in df['stopwords_removal_final']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ''

    for term in term_dict:
        term_dict[term] = stemming(term)

    def get_stemming(document):
      return [term_dict[term] for term in document]

    df['stemming'] = df['stopwords_removal_final'].swifter.apply(get_stemming)

    df['stemming']

    # def fittweet(text):
    #     text = np.array(text)
    #     text = ' '.join(text)
    #     return text

    # df ['Text'] = df['stemming'].apply(lambda x: fittweet(x))
    # df = df.drop('TweetStop', axis=1)
    # df.head(20)
    
    """### Load Data Hasil Labelling dengan nilai Compoundnya"""

    #load data
    # df = pd.read_csv('https://raw.githubusercontent.com/RBellaApriliaDamayanti22/projectt/main/hasil_stemming.csv')
    # df

    #menghilangkan kolom 0-2
    # df.drop(df.columns[[0,1,2,3,4,5,6,7]], axis=1, inplace=True)
    # # df.drop(df.columns[[0,1,2,3,4,5]], axis=1, inplace=True)
    # df
    #translating
    #melakukan translate ke bahasa inggris
    import googletrans
    from googletrans import Translator
    translator = Translator()
    # translations = {}
    # for column in df.columns:
    #   unique_elements = df[column].unique()
    #   for element in unique_elements:
    #     translations[element] = translator.translate(element).text
    # translations

    # df.replace(translations, inplace=True)
    #load data traslate



    df = pd.read_csv('https://raw.githubusercontent.com/RBellaApriliaDamayanti22/projectt/main/translate_data_review.csv')

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    sentiment=[]
    scores = []
    # Declare variables for scores
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []
    for i in df['Text']:
        compound = analyzer.polarity_scores(i)["compound"]
        pos = analyzer.polarity_scores(i)["pos"]
        neu = analyzer.polarity_scores(i)["neu"]
        neg = analyzer.polarity_scores(i)["neg"]

        scores.append({"Compound": compound,
                            "Positive": pos,
                            "Negative": neg,
                            "Neutral": neu
                        })
    # if compound>=0.05:
    #   nilai_sentiment='positive'
    # elif compound > -0.05 and compound <0.05:
    #   nilai_sentiment='neutral'
    # elif compound <=-0.05:
    #   nilai_sentiment='negative'
        if compound>0:
            nilai_sentiment='positive'
        elif compound == 0:
            nilai_sentiment='neutral'
        elif compound < 0:
            nilai_sentiment='negative'
        sentiment.append({"sentiment": nilai_sentiment})

    sentiments_score = pd.DataFrame.from_dict(scores)
    sentiments_data=pd.DataFrame.from_dict(sentiment)

    df2 = pd.concat([df,sentiments_score],axis=1)
    df = pd.concat([df2,sentiments_data],axis=1)
    # df = tweet_df.join(sentiments_score)


    s = pd.value_counts(df['sentiment'])
    ax = s.plot.bar()
    n = len(df.index)

    klasifikasi_df = df[df.sentiment != 'neutral']
    klasifikasi_df.reset_index(drop=True, inplace=True)

    s_1 = klasifikasi_df[klasifikasi_df['sentiment']=='positive'].sample(1000, replace=True)
    s_2 = klasifikasi_df[klasifikasi_df['sentiment']=='negative'].sample(1000, replace=True)
    klasifikasi_df = pd.concat([s_1,s_2])
    klasifikasi_df

    s = pd.value_counts(klasifikasi_df['sentiment'])
    ax = s.plot.bar()
    n = len(klasifikasi_df.index)

    klasifikasi_df.to_csv('https://raw.githubusercontent.com/RBellaApriliaDamayanti22/projectt/main/data_klasifikasi_shopee.csv', encoding='utf8', index=False)
    klasifikasi_df
    def arr(text):
        array_perkata = text.split()
        return array_perkata
    klasifikasi_df['Text'] = klasifikasi_df['Text'].apply(arr)
    klasifikasi_df

    #Splitting Data
    from sklearn.model_selection import train_test_split
    # Train Test Split Function
    X = klasifikasi_df[['Text']]
    y = klasifikasi_df['sentiment']
    def split_train_test(data_df, test_size=0.2, shuffle_state=True): # 80% Training & 20% Testing. Ubah nilai pada variable test_size jika ingin skala yang lain
        X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                            y,
                                                            shuffle=shuffle_state,
                                                            test_size=test_size, 
                                                            stratify=y,
                                                            random_state=42)
        print("Value counts for Train sentiments")
        print(Y_train.value_counts())
        print("Value counts for Test sentiments")
        print(Y_test.value_counts())
        print(type(X_train))
        print(type(Y_train))
        X_train = X_train.reset_index()
        X_test = X_test.reset_index()
        Y_train = Y_train.to_frame()
        Y_train = Y_train.reset_index()
        Y_test = Y_test.to_frame()
        Y_test = Y_test.reset_index()
        print(X_train.head())
        return X_train, X_test, Y_train, Y_test

    # Call the train_test_split
    X_train, X_test, Y_train, Y_test = split_train_test(klasifikasi_df)

    # Word2vec
    OUTPUT_FOLDER = 'word2vec_1000.model' #Untuk menyimpan file model, sesuaikan dengan drive sendiri. 
    # Skip-gram model (sg = 1)
    size = 1000
    window = 3
    min_count = 1
    workers = 3
    sg = 1

    word2vec_model_file = OUTPUT_FOLDER 
    start_time = time.time()
    stemmed_tokens = pd.Series(klasifikasi_df['Text']).values
    # Train the Word2Vec Model
    w2v_model = Word2Vec(stemmed_tokens, min_count = min_count, vector_size = size, workers = workers, window = window, sg = sg)
    print("Time taken to train word2vec model: " + str(time.time() - start_time))
    w2v_model.save("word2vec_model_file")

    # # Load the model from the model file
    sg_w2v_model = Word2Vec.load("word2vec_model_file")
    # # # Unique ID of the word
    # # print("Index of the word 'bagus':")
    # # # print(sg_w2v_model.wv.vocab["bagus"].index)
    # # # Total number of the words 
    # # print(len(sg_w2v_model.wv.vocab))
    # # # Print the size of the word2vec vector for one word
    # # print("Length of the vector generated for a word")
    # # print(len(sg_w2v_model['bagus']))
    # # # Get the mean for the vectors for an example review
    # # print("Print Vector Kata di index ke-0:")
    # # print(np.mean([sg_w2v_model[token] for token in df['stemming'][2]], axis=0))

    # # '''
    # # print("\n Print Semua Kata")
    # # for i in range (len(df['stemming'])):
    # # print(np.max([sg_w2v_model[token] for token in df['stemming'][i]], axis=0))'''

    #Vector Setiap Kata
    stem = []
    for i in range (len(X_train['Text'])):
        for y in range(len(X_train['Text'][i])):
            stem.append(X_train['Text'][i][y])

    #Vector Setiap Kata 
    v = []
    for i in stem:
        vector_w2v = np.mean(sg_w2v_model.wv.get_vector(i))
        v.append(vector_w2v)
    print(v)

    vector_w2v = pd.DataFrame(list(zip(stem, v)), columns =['kata', 'vector'])


    word2vec_filename = OUTPUT_FOLDER 
    vector_w2v_file = vector_w2v.to_csv(word2vec_filename, index = False)

    with open(word2vec_filename, 'w+') as word2vec_file:
        for index, row in X_train.iterrows():
            model_vector = (np.mean([sg_w2v_model.wv.get_vector(token) for token in row['Text']], axis=0)).tolist()
            if index == 0:
                header = ",".join(str(ele) for ele in range(1000))
                word2vec_file.write(header)
                word2vec_file.write("\n")
            # Check if the line exists else it is vector of zeros
            if type(model_vector) is list:  
                line1 = ",".join( [str(vector_element) for vector_element in model_vector] )
            else:
                line1 = ",".join([str(0) for i in range(1000)])
            word2vec_file.write(line1)
            word2vec_file.write('\n')

with modeling:
    import time
    from sklearn.naive_bayes import GaussianNB
    # Load from the filename
    word2vec_df = pd.read_csv(word2vec_filename)
    gnb = GaussianNB(priors=None, var_smoothing= 2.848035868435799e-06)

    start_time = time.time()
    # Fit the model
    gnb.fit(word2vec_df, Y_train['sentiment'])
    print("Time taken to fit the model with word2vec vectors: " + str(time.time() - start_time))

    from sklearn.metrics import classification_report
    test_features_word2vec = []
    for index, row in X_test.iterrows():
        model_vector = np.mean([sg_w2v_model.wv.get_vector(token) for token in row['Text']], axis=0)
        if model_vector.dtype == 'float32':
            test_features_word2vec.append(model_vector)
        else:
            test_features_word2vec.append(np.array([0 for i in range(1000)]))
    test_predictions_word2vec = gnb.predict(test_features_word2vec)
    print(classification_report(Y_test['sentiment'],test_predictions_word2vec))

    from sklearn.metrics import classification_report, confusion_matrix

    #Prediciting the test results
    from sklearn.metrics import classification_report, confusion_matrix

    #Prediciting the test results
    y_predict_test = gnb.predict(test_features_word2vec)
    cm = confusion_matrix(Y_test['sentiment'], y_predict_test)
    categories = ['Negative','Positive']

    plt.figure()
    st.write("Confusion Matrix")
    sns.heatmap(cm, annot=True, fmt='', cmap='Blues', xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix Naive Bayes", fontdict = {'size':18}, pad = 20)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    # categories = ['Negative','Positive']
    # y_predict_test = gnb.predict(test_features_word2vec)
    # cm = confusion_matrix(Y_test['sentiment'], y_predict_test)
    # cm_df = pd.DataFrame(cm, index=categories,columns=categories)
    # st.write("Confusion Matrix")
    # sns.heatmap(cm_df, annot=True, cmap='Blues', xticklabels = categories, yticklabels = categories)
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # st.pyplot()
    # plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    # plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    # plt.title ("Confusion Matrix Naive Bayes", fontdict = {'size':18}, pad = 20)
    # st.write("Menyimpan data hasil preprocessing ke pickle")
    # with open('data.pickle', 'wb') as file:
    #     pickle.dump(dataset, file)

    # # Memuat data dari file pickle
    # with open('data.pickle', 'rb') as file:
    #     loaded_data = pickle.load(file)
    # Data_Ulasan = pd.DataFrame(loaded_data, columns=["label", "Ulasan"])
    # Data_Ulasan.head()
    # Ulasan = Data_Ulasan['Ulasan']
    # sentimen = Data_Ulasan['label']
    # X_train, X_test, y_train, y_test = train_test_split(Ulasan, sentimen, test_size=0.2, random_state=42)

    # def convert_text_list(texts):
    #     try:
    #         texts = ast.literal_eval(texts)
    #         if isinstance(texts, list):
    #             return texts
    #         else:
    #             return []
    #     except (SyntaxError, ValueError):
    #         return []

    # Data_Ulasan["Ulasan_list"] = Data_Ulasan["Ulasan"].apply(convert_text_list)
    # print(Data_Ulasan["Ulasan_list"][90])
    # print("\ntype: ", type(Data_Ulasan["Ulasan_list"][90]))

    # # Klasifikasi menggunakan Naive bayes
    # # Load from the filename
    # word2vec_df = pd.read_csv(word2vec_filename)
    # #Initialize the model https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5
    # # clf_decision_word2vec = SVC(C=0.1, gamma=0.1, kernel="rbf")
    # # clf_decision_word2vec = SVC()
    # gnb = GaussianNB()

    # start_time = time.time()
    # # Fit the model
    # gnb.fit(word2vec_df, Y_train['sentiment'])
    # print("Time taken to fit the model with word2vec vectors: " + str(time.time() - start_time))
    # gaussian = GaussianNB()
    # gaussian = gaussian.fit(word2vec_df, y_train)

    # from sklearn.metrics import classification_report
    # test_features_word2vec = []
    # for index, row in X_test.iterrows():
    #     model_vector = np.mean([sg_w2v_model.wv.get_vector(token) for token in row['Text']], axis=0)
    #     if model_vector.dtype == 'float32':
    #         test_features_word2vec.append(model_vector)
    #     else:
    #         test_features_word2vec.append(np.array([0 for i in range(1000)]))
    # test_predictions_word2vec = gnb.predict(test_features_word2vec)
    # print(classification_report(Y_test['sentiment'],test_predictions_word2vec))


# Implementasi dengan Streamlit
# with implementation:
#     st.title("Klasifikasi Sentimen Ulasan Menggunakan Naive Bayes")
#     st.write("Masukkan Ulasan di bawah ini:")
#     from preprocessor import preprocessing, vectorizer, model

#     inp = st.text_area(label="inp", label_visibility="hidden", placeholder="Please input text...", height=350)
#     btn_submit = st.button("submit")


#     if btn_submit:
#         if inp == "":
#             st.write("error")
#         else:
#             prep = preprocessing(inp)
#             st.write("prep : ",prep)
            
#             transform = vectorizer.transform([" ".join(prep)])

#             label = model.predict(np.asarray(transform.todense()))
#             st.write(label[0])
    # input_text = st.text_input("Ulasan")
    # gnb = GaussianNB(priors=None, var_smoothing= 2.848035868435799e-06)

    # if st.button("Prediksi"):
    #     # Mengubah input Ulasan menjadi vektor
    #     input_vector = test_features_word2vec(input_text)

    #     # Melakukan prediksi pada input Ulasan
    #     predicted_label = gnb.predict([input_vector])

    #     # Menampilkan hasil prediksi
    #     st.write("Hasil Prediksi:")
    #     st.write(f"Ulasan: {input_text}")
    #     st.write(f"Label: {predicted_label[0]}")

    # # Menghitung akurasi pada data uji
    # y_predict_test = gnb.predict(input_text)
    # # accuracy = accuracy_score(Y_test['sentiment'], y_predict_test)

    # # Menampilkan akurasi
    # # st.write("Akurasi: {:.2f}%".format(accuracy * 100))

    # # Menampilkan label prediksi
    # st.write("Label Prediksi:")
    # for i, (label, Ulasan) in enumerate(zip(y_predict_test, y_test)):
    #     st.write(f"Data Uji {i+1}:")
    #     st.write(f"Ulasan: {Ulasan}")
    #     st.write(f"Label: {label}")


# Implementasi dengan Streamlit
with implementation:
    st.title("Klasifikasi Sentimen Ulasan Menggunakan Naive Bayes")
    st.write("Masukkan Ulasan di bawah ini:")
    from preprocessor import preprocessing, vectorizer, model


    word2vec = Word2Vec.load("word2vec_model_file")
    # Fungsi untuk mendapatkan vektor kata dari model
    def get_word_vector(word):
        try:
            return word2vec.wv[word]
        except KeyError:
            return None

    inp = st.text_area(label="inp", label_visibility="hidden", placeholder="Please input text...", height=350)
    btn_submit = st.button("submit")


    if btn_submit:
        if inp == "":
            st.write("error")
        else:
            prep = preprocessing(inp)
            prep = " ".join([str(elem) for elem in prep])

            def translate_text(text, src_lang, dest_lang):
                translator = Translator()
                translated = translator.translate(text, src=src_lang, dest=dest_lang)
                return translated.text

            # Contoh penggunaan
            translated_text = translate_text(prep, "id", "en")
            clean = re.sub("[^a-zA-Zï ]+"," ", translated_text)
            tekons = nltk.word_tokenize(clean.lower())
            data = pd.DataFrame([[tekons]],columns=["prep"])
            st.write(data)
            test_features_word2vec = []
            for index, row in data.iterrows():
                model_vector = np.mean([word2vec.wv.get_vector(token) for token in row["prep"]], axis=0)
                if model_vector.dtype == 'float32':
                    test_features_word2vec.append(model_vector)
                else:
                    test_features_word2vec.append(np.array([0 for i in range(1000)]))

            # Contoh penggunaan
            # word = "parkir"
            # vector = get_word_vector(word)
            # if vector is not None:
            #     st.write(f"Vektor kata '{word}': {vector}")
            # else:
            #     st.write(f"Kata '{word}' tidak ditemukan dalam model.")

            label = model.predict(test_features_word2vec)
            st.write(label)
            