# coding:"utf_8"
#前処理用の関数をまとめる
import sys
import MeCab
import pandas as pd
import pprint
import numpy as np

tagger = MeCab.Tagger("-Owakati")

class News:
    def __init__(self,news,category):
        self.news = news
        self.category = category
    
def Tokenize(filename,imputattribute):
    """imputattributeの属性を形態素分析して List of Lists of Tokens の形に\n\
ファイルはCSVファイルにしておくこと"""
    df = pd.read_csv(filename)
    attributes =  df[imputattribute]
    TitleTokens = []

    for attribute in attributes :
        TitleTokens.append(tagger.parse(attribute).strip().split())


    return TitleTokens

def Tokenize_SplitBySpace(filename,imputattribute):
    """imputattributeの属性を形態素分析して List of Lists of Tokens の形に\n\
ファイルはCSVファイルにしておくこと"""
    df = pd.read_csv(filename)
    attributes =  df[imputattribute]
    TitleTokens = []

    for attribute in attributes:
        text = tagger.parse(attribute).strip().split()
        text = SplitBySpace(text)
        TitleTokens.append(text)


    return TitleTokens

def Tokenize_SingleText(text):
    """textを形態素分析して List of Lists of Tokens の形に"""
    
    TextTokens = []

    TextTokens.append(tagger.parse(text).strip().split())

    return TextTokens

def CategoryToNumber(filename,imputattribute, *categories):
    """カテゴリ変数を整数化\n
データセット内のカテゴリーを全てcategoriesに書き出しておくこと"""
    df = pd.read_csv(filename)
    attributes = df[imputattribute]
    ReturnList = []

    for attribute in attributes :
        if attribute in categories:
            ReturnList.append(categories.index(attribute))
        else: raise ValueError(u"カテゴリー外の項目があります。")

    return ReturnList


def SplitBySpace(textlists):
    """リスト内の文字列を半角スペースで分けられた一つの文字列に"""
    returntext = ''
    for textlist in textlists:
        returntext = returntext + textlist + " "
    
    return returntext