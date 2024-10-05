'''

PROBLEM:
Wikipedia metinleri içeren veri setine metin ön işleme ve görselleştirme yapınız.

Proje Görevleri:

'''

import string
from warnings import filterwarnings
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ImagePalette import random
from nltk.corpus import stopwords, words
from nltk.sentiment import SentimentIntensityAnalyzer
from pandas import DataFrame

from textblob import Word, TextBlob
from wordcloud import WordCloud
filterwarnings('ignore')
pd.set_option('display.max_columns',None)
pd.set_option('display.width',200)
pd.set_option('display.float_format',lambda x: '%.2f' % x)

df = pd.read_csv("C:/Users/SALİH KARAYILAN/OneDrive/Desktop/wiki-221126-161428/wiki_data.csv")
print('\nHAM DATASET :\n\n',df.head())




def clean_text(data, Barplot=False, Wordcloud=False):

    '''
    :param data: DataFrame'deki textlerin olduğu değişken
    :param Barplot: Barplot görselleştirme
    :param Wordcloud: Wordcloud görselleştirme
    :return: dat
    '''
    # Büyük küçük harf dönüşümü,
    data = data.str.lower()

    #Noktalama işaretlerini çıkarma,
    data = data.str.replace(r'[^\w\s]', '', regex=True)

    #Numerik ifadeleri çıkarma Işlemlerini gerçekleştirmeli.
    data = data.str.replace(r'\d', '',regex=True)

    #Metin içinde öznitelik çıkarımı yaparken önemli olmayan kelimeleri çıkartılmalı.
    #nltk.download('stopwords')
    sw = stopwords.words('english')
    data = data.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

    #Metinleri tokenize edip sonuçları gözlemleyiniz.
    temp_df = pd.Series(' '.join(data).split()).value_counts()
    drops = temp_df[temp_df<2000]
    data = data.apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))
    #nltk.download("punkt_tab")
    print(data.apply(lambda x : TextBlob(x).words).head())

    #Lemmatization işlemi yapınız
    #nltk.download('wordnet')
    data = data.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


    '''
    Task 2: Veriyi Görselleştiriniz
        Step 1: Metindeki terimlerin frekanslarını hesaplayınız.

        Step 2: Bir önceki Stepda bulduğunuz terim frekanslarının Barplot grafiğini oluşturunuz.

        Step 3: Kelimeleri WordCloud ile görselleştiriniz.  
    '''

    if Barplot:
        tf = data.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        tf.columns = ["words", "tf"]
        tf[tf['tf']>5000].plot.bar(x = 'words', y ='tf')
        plt.xlabel('WORDS')
        plt.ylabel('NUMBER OF WORDS')
        plt.title('Number of Each Word in the Wikipedia Dataset')
        plt.show()

    if Wordcloud:
        text = ' '.join(i for i in data)
        wordcloud = WordCloud(max_font_size=500,
                          max_words=100,
                          background_color='black').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        wordcloud.to_file('wordcloud.png')

    return data

print(clean_text(df['text']))
print(clean_text(df['text']),False,True)










