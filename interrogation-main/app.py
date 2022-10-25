from __future__ import unicode_literals
from email import message
from inspect import trace
import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, TemplateSendMessage, ButtonsTemplate, MessageTemplateAction
import configparser
import random
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
from sklearn import tree
import jieba
from ArticutAPI import Articut
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np 
from glob import glob
import os
import jieba.analyse
import schedule
import time
import threading
from datetime import datetime
import requests


jieba.load_userdict('./jieba.txt')

# 引用私密金鑰
cred = credentials.Certificate('clinic-smart-firebase-adminsdk-nfmbm-464950a5fa.json') # 為了firebase把原本的serviceAccount.json換掉

# 初始化firebase，注意不能重複初始化
firebase_admin.initialize_app(cred)

# 初始化firestore
db = firestore.client()

# 
data = pd.read_csv('特徵矩陣.csv')
data = data.sample(frac=1, random_state=1).reset_index(drop=True) # 將資料順序打亂，以便後續進行cross validation
p0 = pd.read_csv('p0.csv', index_col="Disease")
p1 = pd.read_csv('p1.csv', index_col="Disease")

label = data['Disease']
Diseases = label.unique()
symptoms = data.drop(["Disease"], axis=1).columns

from flask_ngrok import run_with_ngrok
app = Flask(__name__)

#----- stopword -------------------------------------------------------------------------------------------

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
def movestopwords(sentence):
    stopwords = stopwordslist('stopword.txt')  # 這裏加載停用詞的路徑
    outstr = []
    for word in sentence:           #句子中的每一個字，
        if word not in stopwords:   #這裏和英文不一樣，應爲如果這樣用，就是字母了
            if word != '\t'and'\n':
                outstr.append(word)
    return outstr


#----- handle word (word segmentation、fuzzywuzzy、remove common symptom) ----------------------------------

def word_segmentation(input_message,UserId,event): # word segmentation and remove stopword
    seg_word = []

    # articut = Articut(username="", apikey="")
    # result = articut.parse(input_message)
    # contentWordLIST = articut.getContentWordLIST(result)
    content_seg = jieba.cut(input_message)    # jieba分詞
#   print("jieba分詞後：",content_seg)
    contentWordLIST = []
    for i in content_seg:
        contentWordLIST.append(i)

    contentWordLIST = movestopwords(contentWordLIST)
    print("----- word_segmentation -----")
    # for sentence in contentWordLIST:
    #     for word in sentence:
    #         seg_word.append(word[-1])

    after_seg_word = replace_synonym(UserId, contentWordLIST)
    print(after_seg_word)
    temp = ""
    for i in range(len(after_seg_word)):
        temp += after_seg_word[i]
        temp += ' '
    seg_word=temp
    print(seg_word)
    print("----- word_segmentation finish -----")
    

    feature = [i for i in fuzzywuzzy(seg_word,UserId,event)]
    transform_data = pd.DataFrame(data=None, columns=symptoms)
    transform_data.loc[len(transform_data)] = 0
    for i in range(len(feature)):
        if feature[i] == '':
            break
        transform_data.at[len(transform_data)-1, feature[i].replace(' ','')] = 1
    return transform_data

def replace_synonym(UserId, seg_word):
    file = open('同義詞.csv', 'r') # encoding='utf-8-sig'
    words = [line.replace('\n', '').split(',') for line in file]

    storage = []
    storage1 = []
    unknown = []
    for i in range(len(seg_word)):
        for word in words:
            if seg_word[i] in word:
                # if seg_word[i] != word[0]:
                storage.append(seg_word[i])
                storage1.append(word[0])
                seg_word[i] = word[0]
    
    flag = False #抓 小雞雞 等詞
    for i in range(len(seg_word)):
        for j in range(len(storage1)):
            if seg_word[i] == storage1[j]:
                flag = True
        
        if flag == False:
            unknown.append(seg_word[i])

        flag = False
    
    # temp = []
    # for i in range(len(storage)): # 去前面原本去重複的
    #     if storage[i] == storage1[i]:
    #         temp.append(storage[i])
    # for i in range(len(temp)):
    #         storage.remove(temp[i])
    #         storage1.remove(temp[i])



    reply_replace_synonym(UserId, storage, storage1, unknown)
    return seg_word

def reply_replace_synonym(UserId, storage, storage1, unknown):
    sentence = ""
    for i in range(len(storage)):
        if i == (len(storage) - 1):
            sentence += "後台已將\"" + storage[i] + "\"替換成\"" + storage1[i] + "\""
        else:
            sentence += "後台已將\"" + storage[i] + "\"替換成\"" + storage1[i] + "\"\n"

    temp = ""
    for i in range(len(unknown)):
        if i == (len(unknown) - 1):
            temp += "\"" + unknown[i] + "\""
        else:
            temp += "\"" + unknown[i] + "\"" + "、"
    
    if len(temp) != 0:
        sentence += "\n\n"
        sentence += "後台經斷詞後無法辨識" + temp + "，請之後再換其他詞表示。"
    
    if len(sentence) != 0:
        line_bot_api.push_message(UserId, TextSendMessage(text = sentence))


def fuzzywuzzy(seg_word,UserId,event):
    f = open('userdict.txt',"r",encoding="utf-8")
    userdict = []
    for line in f:
        line = line.replace('\n', '')
        userdict.append(line)
    # print(userdict)
    
    print("----- fuzzywuzzy -----")
    result = process.extract(seg_word, userdict)
    
    after_fuzz = []
    delete_symptom = []
    for i in range(len(result)):
        data, grade = result[i]
        if grade >= 90: # completely same
            after_fuzz.append(data)
            # with open('afterfuzz.txt', 'a+') as f:
            #     f.write(data)
        else:
            str_grade = str(grade)
            temp = data + ": " + str_grade
            delete_symptom.append(temp)
    print(after_fuzz)
    for i in range(len(after_fuzz)):
        with open(UserId+'_afterfuzz.txt', 'a',encoding="utf_8") as f:
            f.write(after_fuzz[i])
            f.write("\n")

    if len(after_fuzz) == 0:
        with open(UserId+'_afterfuzz.txt', 'a', encoding="utf_8") as f:
            f.write("")
        with open(UserId+'_check_afterfuzz.txt', 'a',encoding="utf_8") as f:
            f.write("False")
            f.write("\n")
    else:
        with open(UserId+'_check_afterfuzz.txt', 'a',encoding="utf_8") as f:
            f.write("True")
            f.write("\n")
        # delete_nofuzz(UserId,event)

    print("score >= 90 (completely same):")
    for i in range(len(after_fuzz)):
        if (i + 1) != len(after_fuzz):
            print(after_fuzz[i], end = "，")
        else:
            print(after_fuzz[i])
    print(" ")

    print("score < 90 (sympyom: score):")
    for i in range(len(delete_symptom)):
        print(delete_symptom[i])

    print("----- fuzzywuzzy finish -----")
    
    return after_fuzz

# 給後面輸入醫生的判斷結果
# data = pd.read_csv('特徵矩陣.csv')
# # data = data.sample(frac=1, random_state=1).reset_index(drop=True) # 將資料順序打亂，以便後續進行cross validation
# label = data['Disease']
# Diseases = label.unique()
# symptoms = data.drop(["Disease"], axis=1).columns

# def DoctorBayes():
#     Diseases = data["Disease"].unique()
#     symptoms = data.drop(["Disease"], axis=1).columns

#     total_symptoms_count = {sym: data[sym].sum() for sym in symptoms}
#     symptoms_count = {Disease: {sym: data.loc[data["Disease"] == Disease, sym].sum(
#     ) for sym in symptoms} for Disease in Diseases}
#     total_case_number = label.count()
#     case_number = dict(collections.Counter(label))

#     prior = {key: (value/total_case_number) for key, value in case_number.items()}
#     likelihood_1 = {Disease: {key: (value+1)/(case_number[Disease]+len(symptoms)) for key, value in symptoms_count[Disease].items()} for Disease in Diseases}  # +1做平滑處理
#     # evidence_1 = { sym: (total_symptoms_count[sym]/total_case_number) for sym in symptoms}
#     # posterior_1 = { Disease: {sym: (likelihood_1[Disease][sym] * prior[Disease] / evidence_1[sym]) for sym in symptoms} for Disease in Diseases}

#     likelihood_0 = {Disease: {key: ((case_number[Disease]-value+1)/(case_number[Disease]+len(symptoms))) for key, value in symptoms_count[Disease].items()} for Disease in Diseases}
#     # evidence_0 = { sym: 1-(total_symptoms_count[sym]/total_case_number) for sym in symptoms}
#     # posterior_0 = {Disease: {sym: (likelihood_0[Disease][sym] * prior[Disease] / evidence_0[sym]) for sym in symptoms} for Disease in Diseases}

#     pr = pd.DataFrame.from_dict(prior, orient='index')
#     pr.to_csv('pr.csv', index=True, index_label="Disease")
#     p1 = pd.DataFrame.from_dict(likelihood_1, orient='index')
#     p1.to_csv('ll1_ls.csv', index=True, index_label="Disease")
#     p0 = pd.DataFrame.from_dict(likelihood_0, orient='index')
#     p0.to_csv('ll0_ls.csv', index=True, index_label="Disease")

#     # print(evidence_1)


#----- bayesian ------------------------------------------------------------------------------------------
# def predict(mat):
#     # 預測疾病並輸出機率矩陣
#     Diseases = label.unique()
#     symptoms = data.drop(["Disease"], axis=1).columns

#     pr = pd.read_csv('pr.csv', index_col="Disease")
#     p0 = pd.read_csv('ll0_ls.csv', index_col="Disease")
#     p1 = pd.read_csv('ll1_ls.csv', index_col="Disease")

#     probArr = []
#     for index, row in mat.iterrows():
#         # prob = {Disease: math.prod([p1.loc[Disease, sym] if row[sym] == 1 else p0.loc[Disease, sym] for sym in symptoms]) for Disease in Diseases}
#         prob = {Disease: math.prod([p1.loc[Disease, sym] if row[sym] == 1 else 1 for sym in symptoms]) * pr.loc[Disease, 'probability'] for Disease in Diseases}
#         # 按比例縮放
#         s = sum(prob.values())
#         for key, value in prob.items():
#             prob[key] = value / s
#         probArr.append(prob)
#         # predictArr.append(max(probArr, key=probArr.get))
#     prob = probArr[0]
#     prob = sorted(prob.items(), key=lambda x:  x[1], reverse=True)[:2]
#     print(prob)
#     return prob


def predict(mat):
    # 預測疾病並輸出機率矩陣
    Diseases = label.unique()
    symptoms = data.drop(["Disease"], axis=1).columns

    pr = pd.read_csv('pr.csv', index_col="Disease")
    p0 = pd.read_csv('ll0_ls.csv', index_col="Disease")
    p1 = pd.read_csv('ll1_ls.csv', index_col="Disease")

    probArr = []
    for index, row in mat.iterrows():
        prob = {Disease: math.prod([p1.loc[Disease, sym] if row[sym] == 1 else 1 for sym in symptoms]) * pr.loc[Disease, 'probability'] for Disease in Diseases}
        # prob = {Disease: math.prod([p1.loc[Disease, sym] if row[sym] == 1 else 1 for sym in symptoms]) for Disease in Diseases}
        # 按比例縮放
        s = sum(prob.values())
        for key, value in prob.items():
            prob[key] = value / s
        probArr.append(prob)
        # predictArr.append(max(probArr, key=probArr.get))
    prob = probArr[0]
    prob = sorted(prob.items(), key=lambda x:  x[1], reverse=True)[:5]
    # print(prob)
    return prob
#----- decision tree ------------------------------------------------------------------------------------------
def decisiontree(UserId):
    disease = pd.read_csv('特徵矩陣(decisiontree).csv')
    diseases = disease['Disease'].unique()
    # print(diseases)
    # test = pd.read_csv('testing矩陣.csv')
    # tests = test['Disease'].unique()
    # print(tests)
    afpredict = []
    f = open(UserId+'_disease.txt', 'r', encoding="utf_8")
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.replace("\n", "")
        afpredict.append(line)

    after_fuzz = []
    f = open(UserId+'_afterfuzz.txt', 'r', encoding="utf_8")
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.replace("\n", "")
        after_fuzz.append(line)

    linebot_diseases = afpredict
    for ld in linebot_diseases:
        for d in diseases:
            if ld == d:
                new_diseases = disease[disease['Disease'] == d]
                new_diseases.to_csv(f'{UserId}_disease_csv_{d}.csv', index=False, encoding='utf_8_sig')  #索引值不匯出到CSV檔案中
                break



    # for ld in linebot_diseases:
    #     for t in tests:
    #         if ld == t:
    #             new_test = test[test['Disease'] == t]
    #             new_test.to_csv(f'test_csv_{t}.csv', index=False, encoding='utf_8_sig')  #索引值不匯出到CSV檔案中
    #             break
    disease_files = glob(UserId+'_disease_csv*.csv')
    # print(disease_files)
    # test_files = glob('test_csv*.csv')
    # print(test_files)
    # test = pd.read_csv('testing矩陣.csv')
    # list_test = list(test)
    # print(list_test)

    after_concat_disease = pd.concat((pd.read_csv(file, dtype={'Disease': str}) for file in disease_files), axis='rows')
    for ld in linebot_diseases:
        for d in diseases:
            if ld == d:
                file=UserId+'_disease_csv_'+d+'.csv'
                os.remove(file)
                break
    after_concat_disease.to_csv(f'{UserId}_after_concat_disease.csv', index=False, encoding='utf_8_sig')

    df = pd.read_csv(f'{UserId}_after_concat_disease.csv')
    column = [row for row in df]
    column.pop() #  delete 'Disease'
    for a in after_fuzz:
        for i in range(len(column)):  
            if a == column[i]:
                df = df.drop(a, axis=1, inplace=False)

    # test.to_csv(f'{UserId}_after_testing.csv', index=False, encoding='utf_8_sig')

    # training = pd.read_csv(UserId+'_after_concat_disease.csv')
    training = df
    # testing = pd.read_csv(UserId+'_after_testing.csv')
    cols = training.columns
    cols = cols[:-1]
    x = training[cols]
    y = training['Disease']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    clf1 = DecisionTreeClassifier()
    clf = clf1.fit(x_train, y_train)
    scores = cross_val_score(clf, x_test, y_test, cv=2) # cv=5
    # print(scores)
    # print(f'Score: {scores.mean()}')

    with open(UserId+"_disease-tree.dot", 'w', encoding='utf-8') as f:
        f = tree.export_graphviz(clf, out_file=f, class_names=clf.classes_, feature_names=cols)

# ----- binary method -------------------------------------
def binary_method(UserId):
    with open(UserId+'_disease-tree.dot', encoding="utf-8") as f:
        data = f.readlines()[3:-1]
    
    # 2個元素一組
    dict_node = [] # 先node再名稱
    dict_leave = [] # 先node再名稱
    dict_path = [] # 先source 再 dest
    
    for i in range(len(data)):
        if data[i][2] == "[":
            judge = data[i].find("<")
            if judge != -1: # node
                temp_label = ''
                for j in range(10, (judge - 1)):
                    temp_label += data[i][j]
                
                temp_node = data[i][0]
                
                dict_node.append(temp_node)
                dict_node.append(temp_label)
                
            else: # leave
                judge2 = data[i].find("class")
                temp_label = ''
                for j in range((judge2 + 8), (len(data[i]) - 5)):
                    temp_label += data[i][j]
                
                temp_node = data[i][0]
                
                dict_leave.append(temp_node)
                dict_leave.append(temp_label)
                
        elif data[i][2] == "-":
            dict_path.append(data[i][0])
            if data[i][6].isdigit() == True: # node為 >= 10
                temp_node = data[i][5] + data[i][6]
                dict_path.append(temp_node)
    
            else:
                dict_path.append(data[i][5])
                
        else: # node為 >= 10
            judge = data[i].find("<")
            if judge != -1: # node
                temp_label = ''
                for j in range(11, (judge - 1)):
                    temp_label += data[i][j]
    
                temp_node = data[i][0] + data[i][1]
    
                dict_node.append(temp_node)
                dict_node.append(temp_label)
                
            else: # leave
                judge2 = data[i].find("class")
                temp_label = ''
                for j in range((judge2 + 8), (len(data[i]) - 5)):
                    temp_label += data[i][j]
                    
                temp_node = data[i][0] + data[i][1]
    
                dict_leave.append(temp_node)
                dict_leave.append(temp_label)
                
    # print(dict_node, dict_leave, dict_path)
    
    traversal(UserId, dict_node, dict_leave, dict_path)

# ----- traversal -------------------------------------
def traversal(UserId, dict_node, dict_leave, dict_path):
    class node:
      def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None
      def setLeft(self, left):
        self.left = left
      def setRight(self, right):
        self.right = right
    
    p_reg= []
    for i in range(1, int(len(dict_node) / 2) + 1):
        p_reg.append(node(dict_node[(2 * i - 1)]))
    root = p_reg[0]
    
    for i in range(1, int(len(dict_leave) / 2) + 1):
        p_reg.append(node(dict_leave[(2 * i - 1)]))
    
    # print(p_reg)
    
    for i in range(int(len(dict_path) / 2)):
        for j in range((i + 1), int(len(dict_path) / 2)):
            if int(dict_path[(2 * i)]) == int(dict_path[(2 * j)]):
                if int(dict_path[(2 * i + 1)]) < int(dict_path[(2 * j + 1)]):
                    p_reg[int(dict_path[(2 * i)])].setLeft(p_reg[int(dict_path[(2 * i + 1)])])
                    p_reg[int(dict_path[(2 * i)])].setRight(p_reg[int(dict_path[(2 * j + 1)])])

                else:
                    p_reg[int(dict_path[(2 * i)])].setRight(p_reg[int(dict_path[(2 * i + 1)])])
                    p_reg[int(dict_path[(2 * i)])].setLeft(p_reg[int(dict_path[(2 * j + 1)])])
            
                break
            
    # for i in range(int(len(dict_node) / 2)): # trace
    #     print(i)
    #     print(p_reg[int(dict_node[2 * i])].val)
    #     print(p_reg[int(dict_node[2 * i])].left)
    #     print(p_reg[int(dict_node[2 * i])].right)

    traversal = []
    traversal.append("0\n\n")
    for i in range(len(dict_node)):
        traversal.append(dict_node[i])
        traversal.append("\n")
    traversal.append("\n")
    for i in range(len(dict_leave)):
        traversal.append(dict_leave[i])
        traversal.append("\n")
    traversal.append("\n")
    for i in range(len(dict_path)):
        traversal.append(dict_path[i])
        traversal.append("\n")
    traversal.append("\n")
    for i in range(len(p_reg)):
        traversal.append(str(p_reg[i]))
        traversal.append("\n")
    traversal.append("\n")
    
    f = open(UserId+'_traversal.txt', 'w')
    f.writelines(traversal)
    f.close()

# ----- inquiry -----------------------------------------------
def inquiry(UserId, ans):
    traversal = []
    dict_node = []
    dict_leave = []
    dict_path = []
    p_reg = []
    
    f = open(UserId+'_traversal.txt', 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.replace("\n", "")
        traversal.append(line)
    
    count = 0
    for i in range(len(traversal)):
        if traversal[i] == "":
            count = count + 1
            continue
        
        if count == 1:
            dict_node.append(traversal[i])
            
        elif count == 2:
            dict_leave.append(traversal[i]) 
            
        elif count == 3:
            dict_path.append(traversal[i]) 
            
        elif count == 4:
            p_reg.append(traversal[i]) 
            
    print(traversal[0])
    print(dict_node)
    print(dict_leave)
    print(dict_path)
    print(p_reg)
    
    inquiry_sym = "None"
    final_disease = "None"
    check = False
    current_node_num = int(traversal[0])
    next_node_num = -1
    next_node = ""
    
    if ans == "start":
        inquiry_sym = dict_node[1]

    elif ans == True:
        for i in range(int(len(dict_path) / 2)):
            if int((dict_path[(2 * i)])) == current_node_num:
                if next_node_num == -1:
                    next_node_num = int(dict_path[(2 * i + 1)])

                else:
                    if int(dict_path[(2 * i + 1)]) > next_node_num:
                        next_node_num = int(dict_path[(2 * i + 1)])
                        print("next_node_num")
                        print(next_node_num)            

    elif ans == False:
        for i in range(int(len(dict_path) / 2)):
            if int((dict_path[(2 * i)])) == current_node_num:
                if next_node_num == -1:
                    next_node_num = int(dict_path[(2 * i + 1)])

                else:
                    if int(dict_path[(2 * i + 1)]) < next_node_num:
                        next_node_num = int(dict_path[(2 * i + 1)])
                        print("next_node_num")
                        print(next_node_num)    
        
    if str(next_node_num) in dict_node:
        for i in range(int(len(dict_node) / 2)):
            if int((dict_node[(2 * i)])) == next_node_num:
                next_node = dict_node[(2 * i + 1)]
                inquiry_sym = next_node
                print("inquiry_sym")
                print(inquiry_sym)
                
                with open(UserId+'_traversal.txt', 'r') as file:                     
                    lines = file.readlines()
                    lines[0] = str(next_node_num) + "\n"   #new为新参数，记得加换行符\n
                        
                with open(UserId+'_traversal.txt', 'w') as file:
                    file.writelines(lines)
                    
                break
        
    if str(next_node_num) in dict_leave:
        for i in range(int(len(dict_leave) / 2)):
            if int((dict_leave[(2 * i)])) == next_node_num:
                next_node = dict_leave[(2 * i + 1)]
                final_disease = next_node
                print("final_disease")
                print(final_disease)
                check = True
                
                break
    
    return [inquiry_sym, final_disease, check]

def count_binary(UserId, prob):
    disease = []
    rate = []
    for i in range(len(prob)):
        temp_disease, temp_rate = prob[i]
        disease.append(temp_disease)
        rate.append(temp_rate)

    binary_check = False
    dis_arr = []
    for i in range(len(prob) - 1):
        if (rate[i] - rate[(i + 1)]) < 0.3: # 之後要改的誤差
            binary_check = True
            if i == 0:
                dis_arr.append(disease[i])
                dis_arr.append(disease[(i + 1)])
            else:
                dis_arr.append(disease[(i + 1)])
        else:
            break
    # print(binary_check, dis_arr)
    
    for i in range(len(dis_arr)):
        with open(UserId+'_disease.txt', 'a',encoding="utf_8") as f:
            f.write(dis_arr[i])
            f.write("\n")
            
    return binary_check

#----- delete file ------------------------------------------------------------------------------------------
def delete_file(UserId):
    os.remove(UserId+'_afterfuzz.txt')
    os.remove(UserId+'_check_afterfuzz.txt')
    os.remove(UserId+'_disease.txt')
    os.remove(UserId+'_after_concat_disease.csv')
    # os.remove(UserId+'_after_testing.csv')
    os.remove(UserId+'_disease-tree.dot')
    os.remove(UserId+'_traversal.txt')
    os.remove(UserId+'_timeout_event.txt')
    os.remove(UserId+'_user_data.txt')

def delete_afterfuzz(UserId):
    os.remove(UserId+'_afterfuzz.txt')
    os.remove(UserId+'_check_afterfuzz.txt')

def delete_nofuzz(UserId,event):
    delete_afterfuzz(UserId)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "您輸入的文字無法判斷，請換個說法。"))

def nodelete_afterfuzz(event):
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "請輸入\"是\"或\"否\"，謝謝。"))

def timeout(UserId):
    timeout_event_start = []
    f = open(UserId+'_timeout_event.txt', 'r', encoding="utf_8")
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.replace("\n", "")
        timeout_event_start.append(line)

    start = time.process_time() # 先數完再檢查還是不適同一個event

    while True:
        if (time.process_time() - start) == 300:
            break
    
    if os.path.isfile(UserId+'_timeout_event.txt'): # 可能直接跑完結果，已經刪掉檔案了
        timeout_event_end = []
        f = open(UserId+'_timeout_event.txt', 'r', encoding="utf_8")
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.replace("\n", "")
            timeout_event_end.append(line)

        if timeout_event_start[-1] == timeout_event_end[-1]:
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # print(timeout_event_start[-1])
            # print(timeout_event_end[-1])
            line_bot_api.push_message(UserId, TextSendMessage(text = "由於您長時間未進行操作，因此再一分鐘後將取消您此次諮詢"))
    
    while True:
        if (time.process_time() - start) == 310:
            break
    
    if os.path.isfile(UserId+'_timeout_event.txt'): # 可能直接跑完結果，已經刪掉檔案了
        timeout_event_end = []
        f = open(UserId+'_timeout_event.txt', 'r', encoding="utf_8")
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.replace("\n", "")
            timeout_event_end.append(line)

        if timeout_event_start[-1] == timeout_event_end[-1]:
            line_bot_api.push_message(UserId, TextSendMessage(text = "已取消您此次的諮詢，若有新的病症諮詢，請重新輸入，謝謝"))
            delete_file(UserId)

# 寫進FireBase
def write_in_DB(UserId, input_message, after_fuzz, final, binary_check):
    UserId = str(UserId)
    input_message = str(input_message)
    after_fuzz = str(after_fuzz)
    final = str(final)
    binary_check = str(binary_check)

    currentDateAndTime = datetime.now()
    str_currentDateAndTime = str(currentDateAndTime)

    doc = {
        '使用者Line ID': UserId,
        '一開始輸入訊息': input_message,
        '模糊比對後的詞': after_fuzz,
        '最終判斷疾病': final,
        '是否進行二分法': binary_check,
        '問診時間': str_currentDateAndTime
    }
    
    doc_ref = db.collection(str(UserId)).document(str(currentDateAndTime)) # doc_ref = db.collection("集合名稱").document("文件id")
    doc_ref.set(doc) #input必須是dictionary

# 寫回linebot -- for 問診資訊、疾病科普
def write_out_DB(UserId, action):
    UserId = str(UserId)
    path = UserId
    collection_ref = db.collection(path)
    docs = collection_ref.get()
    if len(docs) == 0: # for沒紀錄的
        final_record = False
        return final_record

    record = []
    final_record = []
    for doc in docs:
        doc = doc.to_dict()
        record.append(doc)

    if action == "查詢最新的四個紀錄":
        record.reverse() # 取最後4個
        if len(record) < 4:
            for i in range(len(record)):
                final_record.append(record[i])
        else:
            for i in range(4):
                final_record.append(record[i])
    
    if action == "列出過往全部的紀錄":
        final_record = record

    return final_record

def reply_record(event, final_record_item):
    temp_str = ""
    text1 = "問診時間: " + final_record_item["問診時間"][:19]
    text1 = text1.replace('\n', '')
    text1 += "\n\n"
    text2 = "一開始輸入訊息: " + final_record_item["一開始輸入訊息"]
    text2 = text2.replace('\n', '')
    text2 += "\n\n"
    text3 = "最終判斷疾病: " + final_record_item["最終判斷疾病"]
    text3 = text3.replace('\n', '')
    temp_str = text1 + text2 + text3
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text = temp_str))

def reply_web(event, disease_for_web):
    url = "https://health.udn.com/health/disease/sole/"
    num_arr = []
    check = False
    with open("篩選後的疾病對應表.csv", 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line = line.replace("\ufeff", "").replace("\n", "")
            line = line.split(",")
            if line[0] == disease_for_web:
                num_arr.append(line[2])
                check = True
    
    combine_url = ""
    for i in range(len(num_arr)): # for 多個url
        combine_url += (url + num_arr[i])
        if i != (len(num_arr) - 1):
            combine_url += "\n"

    if check == False:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "在\"篩選後的疾病對應表.csv\"找不到「" + disease_for_web + "」(應該是漏打)"))
    else:
        tmp_str = "➲ " + disease_for_web + "\n" + combine_url
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text = tmp_str))

def reply_all_web(event, disease_arr_for_web):
    final_url_arr = []
    for i in range(len(disease_arr_for_web)):
        url = "https://health.udn.com/health/disease/sole/"
        num_arr = []
        check = False
        with open("篩選後的疾病對應表.csv", 'r', encoding="utf-8") as f:
            for line in f.readlines():
                line = line.replace("\ufeff", "").replace("\n", "")
                line = line.split(",")
                if line[0] == disease_arr_for_web[i]:
                    num_arr.append(line[2])
                    check = True
        
        combine_url = ""
        for j in range(len(num_arr)): # for 多個url
            combine_url += (url + num_arr[j])
            if j != (len(num_arr) - 1):
                combine_url += "\n"
        final_url_arr.append(combine_url)

        if check == False:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "在\"篩選後的疾病對應表.csv\"找不到「" + disease_arr_for_web[i] + "」(應該是漏打)"))
            return
    
    tmp_str = ""
    for i in range(len(disease_arr_for_web)):
        tmp_str += "➲ " + disease_arr_for_web[i] + "\n" + final_url_arr[i]
        if i != (len(disease_arr_for_web) - 1):
            tmp_str += "\n" + "---------------------------------------------------" + "\n"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text = tmp_str))
    
#----- linebot ------------------------------------------------------------------------------------------
config = configparser.ConfigParser()
config.read('config.ini')
line_bot_api = LineBotApi(config.get('line-bot', 'channel_access_token'))
handler = WebhookHandler(config.get('line-bot', 'channel_secret'))

# yourID = 'U2448be13ae578567931bd8c5e5fa51fa'

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    
    try:
        print(body, signature)
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
        
    return 'OK'

#----- handle message ----------------------------------------------------------------------------------
import re
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    UserId = event.source.user_id # for 功能列表

    if event.message.text == "是":
        UserId = event.source.user_id
        inquiry_arr = inquiry(UserId, True)
        # print(inquiry_arr)
        # traversal = []
        # f = open(UserId+'_traversal.txt', 'r')
        # lines = f.readlines()
        # for line in lines:
        #     line = line.replace("\n", "")
        #     traversal.append(line)
        # f.close()
        
        if inquiry_arr[0] != "None":
            line_bot_api.reply_message(event.reply_token, TemplateSendMessage(alt_text='Buttons template', template=ButtonsTemplate(text= "請問是否有" + inquiry_arr[0] + "的症狀?", actions=[MessageTemplateAction(label= "是", text= "是"), MessageTemplateAction(label= "否", text= "否")])))
            with open(UserId+'_timeout_event.txt', 'a',encoding="utf_8") as f:
                f.write(str(event))
                f.write("\n")
            timeout(UserId)
        if inquiry_arr[1] != "None":
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "診斷為" + inquiry_arr[1]))
            
            tmp_data = []
            with open(UserId+'_user_data.txt', 'r', encoding="utf-8") as f:
                for line in f.readlines():
                    tmp_data.append(line)

            write_in_DB(UserId, tmp_data[0], tmp_data[1], inquiry_arr[1], tmp_data[2])
        if inquiry_arr[2] == True:
            delete_file(UserId)
    
    elif event.message.text == "否":
        UserId = event.source.user_id
        inquiry_arr = inquiry(UserId, False)
        
        # traversal = []
        # f = open(UserId+'_traversal.txt', 'r')
        # lines = f.readlines()
        # for line in lines:
        #     line = line.replace("\n", "")
        #     traversal.append(line)
        
        if inquiry_arr[0] != "None":
            line_bot_api.reply_message(event.reply_token, TemplateSendMessage(alt_text='Buttons template', template=ButtonsTemplate(text= "請問是否有" + inquiry_arr[0] + "的症狀?", actions=[MessageTemplateAction(label= "是", text= "是"), MessageTemplateAction(label= "否", text= "否")])))
            with open(UserId+'_timeout_event.txt', 'a',encoding="utf_8") as f:
                f.write(str(event))
                f.write("\n")
            timeout(UserId)
        if inquiry_arr[1] != "None":
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "診斷為" + inquiry_arr[1]))

            tmp_data = []
            with open(UserId+'_user_data.txt', 'r', encoding="utf-8") as f:
                for line in f.readlines():
                    tmp_data.append(line)

            write_in_DB(UserId, tmp_data[0], tmp_data[1], inquiry_arr[1], tmp_data[2])
        if inquiry_arr[2] == True:
            delete_file(UserId)

    elif event.message.text == "問診":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "請在一句話內輸入您目前有的症狀，謝謝。"))
    
    elif event.message.text == "地圖":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "請點選「功能選單」左側的按鈕，再點選「>」按鈕，接著點選「+」按鈕內的『位置資料』，最後跳出「位置資訊」介面後點選右上角的「分享」，謝謝。"))

    elif event.message.text == "疾病科普":
        line_bot_api.reply_message(event.reply_token, TemplateSendMessage(alt_text='Buttons template', template=ButtonsTemplate(text= "請選擇搜尋方式", actions=[MessageTemplateAction(label= "最新的四個問診紀錄(有去重複)", text= "從最新的四個問診紀錄搜尋"), MessageTemplateAction(label= "全部的問診紀錄(有去重複)", text= "從全部的問診紀錄搜尋"),  MessageTemplateAction(label= "網頁(全部家醫科疾病)", text= "從網頁搜尋")])))

    elif event.message.text == "從最新的四個問診紀錄搜尋":
        action = "列出過往全部的紀錄" # for 去重複
        final_record = write_out_DB(UserId, action)
        final_record.reverse()
        if final_record == False: #for 沒紀錄的
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "資料庫內無您的問診紀錄，請先進行問診再使用此功能，謝謝。"))
            return

        text = ""
        search_arr = []
        check_arr = []
        for i in range(len(final_record)):
            text = final_record[i]["最終判斷疾病"]
            text = text.replace('\n', '')
            if text not in check_arr:
                check_arr.append(text)

                search_arr.append(text)
                if search_arr.index(text) == 0:
                    tmp_str = "一: " + text
                    search_arr[search_arr.index(text)] = tmp_str
                elif search_arr.index(text) == 1:
                    tmp_str = "二: " + text
                    search_arr[search_arr.index(text)] = tmp_str
                elif search_arr.index(text) == 2:
                    tmp_str = "三: " + text
                    search_arr[search_arr.index(text)] = tmp_str
                elif search_arr.index(text) == 3:
                    tmp_str = "四: " + text
                    search_arr[search_arr.index(text)] = tmp_str

                if len(search_arr) == 4: # for 去重複
                    break
        
        if len(search_arr) == 1:
            line_bot_api.reply_message(event.reply_token, TemplateSendMessage(alt_text='Buttons template', template=ButtonsTemplate(text= "請選擇欲查詢的疾病", actions=[MessageTemplateAction(label= search_arr[0], text= "第一個紀錄")])))
        elif len(search_arr) == 2:
            line_bot_api.reply_message(event.reply_token, TemplateSendMessage(alt_text='Buttons template', template=ButtonsTemplate(text= "請選擇欲查詢的疾病", actions=[MessageTemplateAction(label= search_arr[0], text= "第一個紀錄"), MessageTemplateAction(label= search_arr[1], text= "第二個紀錄")])))  
        elif len(search_arr) == 3:
            line_bot_api.reply_message(event.reply_token, TemplateSendMessage(alt_text='Buttons template', template=ButtonsTemplate(text= "請選擇欲查詢的疾病", actions=[MessageTemplateAction(label= search_arr[0], text= "第一個紀錄"), MessageTemplateAction(label= search_arr[1], text= "第二個紀錄"), MessageTemplateAction(label= search_arr[2], text= "第三個紀錄")])))
        elif len(search_arr) == 4:
            line_bot_api.reply_message(event.reply_token, TemplateSendMessage(alt_text='Buttons template', template=ButtonsTemplate(text= "請選擇欲查詢的疾病", actions=[MessageTemplateAction(label= search_arr[0], text= "第一個紀錄"), MessageTemplateAction(label= search_arr[1], text= "第二個紀錄"), MessageTemplateAction(label= search_arr[2], text= "第三個紀錄"), MessageTemplateAction(label= search_arr[3], text= "第四個紀錄")])))
    
    elif event.message.text == "第一個紀錄":
        action = "列出過往全部的紀錄" # for 去重複
        final_record = write_out_DB(UserId, action)
        final_record.reverse()
        if final_record == False: #for 沒紀錄的
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "資料庫內無您的問診紀錄，請先進行問診再使用此功能，謝謝。"))
            return

        text = ""
        check_arr = []
        for i in range(len(final_record)):
            text = final_record[i]["最終判斷疾病"]
            text = text.replace('\n', '')
            if text not in check_arr:
                check_arr.append(text)
                if len(check_arr) == 1:
                    break
        reply_web(event, check_arr[-1])
    elif event.message.text == "第二個紀錄":
        action = "列出過往全部的紀錄" # for 去重複
        final_record = write_out_DB(UserId, action)
        final_record.reverse()
        if final_record == False: #for 沒紀錄的
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "資料庫內無您的問診紀錄，請先進行問診再使用此功能，謝謝。"))
            return

        text = ""
        check_arr = []
        for i in range(len(final_record)):
            text = final_record[i]["最終判斷疾病"]
            text = text.replace('\n', '')
            if text not in check_arr:
                check_arr.append(text)
                if len(check_arr) == 2:
                    break
        reply_web(event, check_arr[-1])
    elif event.message.text == "第三個紀錄":
        action = "列出過往全部的紀錄" # for 去重複
        final_record = write_out_DB(UserId, action)
        final_record.reverse()
        if final_record == False: #for 沒紀錄的
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "資料庫內無您的問診紀錄，請先進行問診再使用此功能，謝謝。"))
            return

        text = ""
        check_arr = []
        for i in range(len(final_record)):
            text = final_record[i]["最終判斷疾病"]
            text = text.replace('\n', '')
            if text not in check_arr:
                check_arr.append(text)
                if len(check_arr) == 3:
                    break
        reply_web(event, check_arr[-1])
    elif event.message.text == "第四個紀錄":
        action = "列出過往全部的紀錄" # for 去重複
        final_record = write_out_DB(UserId, action)
        final_record.reverse()
        if final_record == False: #for 沒紀錄的
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "資料庫內無您的問診紀錄，請先進行問診再使用此功能，謝謝。"))
            return

        text = ""
        check_arr = []
        for i in range(len(final_record)):
            text = final_record[i]["最終判斷疾病"]
            text = text.replace('\n', '')
            if text not in check_arr:
                check_arr.append(text)
                if len(check_arr) == 4:
                    break
        reply_web(event, check_arr[-1])

    elif event.message.text == "從全部的問診紀錄搜尋":
        action = "列出過往全部的紀錄" # for 去重複
        final_record = write_out_DB(UserId, action)
        final_record.reverse()
        if final_record == False: #for 沒紀錄的
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "資料庫內無您的問診紀錄，請先進行問診再使用此功能，謝謝。"))
            return

        text = ""
        check_arr = []
        for i in range(len(final_record)):
            text = final_record[i]["最終判斷疾病"]
            text = text.replace('\n', '')
            if text not in check_arr:
                check_arr.append(text)

        reply_all_web(event, check_arr)

    elif event.message.text == "從網頁搜尋":
        temp_str = ""
        separate_line = "\n" + "---------------------------------------------------" + "\n"
        text0 = " ➲ 心血管: " + "\n"
        text0_url = "https://health.udn.com/health/disease/group/48"
        text1 = " ➲ 新陳代謝與內分泌: " + "\n"
        text1_url = "https://health.udn.com/health/disease/group/33"
        text2 = " ➲ 過敏・免疫與血液疾病: " + "\n"
        text2_url = "https://health.udn.com/health/disease/group/51"
        text3 = " ➲ 肝膽腸胃: " + "\n"
        text3_url = "https://health.udn.com/health/disease/group/28"
        text4 = " ➲ 耳鼻喉科: " + "\n"
        text4_url = "https://health.udn.com/health/disease/group/31"
        text5 = " ➲ 呼吸胸腔: " + "\n"
        text5_url = "https://health.udn.com/health/disease/group/34"
        text6 = " ➲ 泌尿腎臟: " + "\n"
        text6_url = "https://health.udn.com/health/disease/group/49"

        temp_str += (text0 + text0_url + separate_line + text1 + text1_url + separate_line + text2 + text2_url + separate_line + text3 + text3_url + separate_line + text4 + text4_url + separate_line + text5 + text5_url + separate_line + text6 + text6_url)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text = temp_str))

    elif event.message.text == "問診紀錄":
        line_bot_api.reply_message(event.reply_token, TemplateSendMessage(alt_text='Buttons template', template=ButtonsTemplate(text= "請選擇查詢方式", actions=[MessageTemplateAction(label= "查詢最新的四個紀錄", text= "查詢最新的四個紀錄"), MessageTemplateAction(label= "列出過往全部的紀錄", text= "列出過往全部的紀錄")])))
        
    elif event.message.text == "查詢最新的四個紀錄":
        action = "查詢最新的四個紀錄"
        final_record = write_out_DB(UserId, action)
        if final_record == False: #for 沒紀錄的
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "資料庫內無您的問診紀錄，請先進行問診再使用此功能，謝謝。"))
            return

        time_arr = []
        for i in range(len(final_record)):
            if i == 0:
                text = "一: "
            elif i == 1:
                text = "二: "
            elif i == 2:
                text = "三: "
            elif i == 3:
                text = "四: "
            text += final_record[i]["問診時間"][2:19]
            text = text.replace('\n', '')
            time_arr.append(text)
        
        if len(time_arr) == 1:
            line_bot_api.reply_message(event.reply_token, TemplateSendMessage(alt_text='Buttons template', template=ButtonsTemplate(text= "請選擇問診時間", actions=[MessageTemplateAction(label= time_arr[0], text= "第一個問診紀錄")])))
        elif len(time_arr) == 2:
            line_bot_api.reply_message(event.reply_token, TemplateSendMessage(alt_text='Buttons template', template=ButtonsTemplate(text= "請選擇問診時間", actions=[MessageTemplateAction(label= time_arr[0], text= "第一個問診紀錄"), MessageTemplateAction(label= time_arr[1], text= "第二個問診紀錄")])))  
        elif len(time_arr) == 3:
            line_bot_api.reply_message(event.reply_token, TemplateSendMessage(alt_text='Buttons template', template=ButtonsTemplate(text= "請選擇問診時間", actions=[MessageTemplateAction(label= time_arr[0], text= "第一個問診紀錄"), MessageTemplateAction(label= time_arr[1], text= "第二個問診紀錄"), MessageTemplateAction(label= time_arr[2], text= "第三個問診紀錄")])))
        elif len(time_arr) == 4:
            line_bot_api.reply_message(event.reply_token, TemplateSendMessage(alt_text='Buttons template', template=ButtonsTemplate(text= "請選擇問診時間", actions=[MessageTemplateAction(label= time_arr[0], text= "第一個問診紀錄"), MessageTemplateAction(label= time_arr[1], text= "第二個問診紀錄"), MessageTemplateAction(label= time_arr[2], text= "第三個問診紀錄"), MessageTemplateAction(label= time_arr[3], text= "第四個問診紀錄")])))

    elif event.message.text == "第一個問診紀錄":
        action = "查詢最新的四個紀錄"
        final_record = write_out_DB(UserId, action)
        reply_record(event, final_record[0])
    elif event.message.text == "第二個問診紀錄":
        action = "查詢最新的四個紀錄"
        final_record = write_out_DB(UserId, action)
        reply_record(event, final_record[1])
    elif event.message.text == "第三個問診紀錄":
        action = "查詢最新的四個紀錄"
        final_record = write_out_DB(UserId, action)
        reply_record(event, final_record[2])
    elif event.message.text == "第四個問診紀錄":
        action = "查詢最新的四個紀錄"
        final_record = write_out_DB(UserId, action)
        reply_record(event, final_record[3])

    elif event.message.text == "列出過往全部的紀錄":
        action = "列出過往全部的紀錄"
        final_record = write_out_DB(UserId, action)
        if final_record == False: #for 沒紀錄的
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "資料庫內無您的問診紀錄，請先進行問診再使用此功能，謝謝。"))
            return

        temp_str = ""
        for i in range(len(final_record)):
            text0 = " ➲ 第" + str(i + 1) + "筆問診記錄: " + "\n\n"
            text1 = "問診時間: " + final_record[i]["問診時間"][:19]
            text1 = text1.replace('\n', '')
            text1 += "\n\n"
            text2 = "一開始輸入訊息: " + final_record[i]["一開始輸入訊息"]
            text2 = text2.replace('\n', '')
            text2 += "\n\n"
            text3 = "最終判斷疾病: " + final_record[i]["最終判斷疾病"]
            text3 = text3.replace('\n', '')
            temp_str += (text0 + text1 + text2 + text3)

            if i != (len(final_record) - 1):
                temp_str += ("\n" + "---------------------------------------------------" + "\n")

        line_bot_api.reply_message(event.reply_token, TextSendMessage(text = temp_str))

    else:
        input_message = event.message.text
        UserId = event.source.user_id

        # 檢查檔案是否存在
        # filepath = "./" + UserId + '_afterfuzz.txt'
        # print(filepath)
        # if os.path.isfile(filepath):
        #     print(os.path.isfile(filepath))
        #     nodelete_afterfuzz(event)
        #     return
        # else:
        #     sym = word_segmentation(input_message,UserId,event)
        sym = word_segmentation(input_message,UserId,event)
        
        
        after_fuzz = []
        f = open(UserId+'_afterfuzz.txt', 'r', encoding="utf_8")
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.replace("\n", "")
            after_fuzz.append(line)

        check_afterfuzz = []
        f = open(UserId+'_check_afterfuzz.txt', 'r', encoding="utf_8")
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.replace("\n", "")
            check_afterfuzz.append(line)

        if check_afterfuzz[-1] == "False" and len(after_fuzz) == 0: # 一開始直接輸錯
            delete_nofuzz(UserId,event)
            return
        elif check_afterfuzz[-1] == "False" and len(after_fuzz) != 0: # 在二分法輸錯
            nodelete_afterfuzz(event)
            with open(UserId+'_timeout_event.txt', 'a',encoding="utf_8") as f:
                f.write(str(event))
                f.write("\n")
            timeout(UserId)
            return

        prob = predict(sym)

        reply_arr=[]
        disease1 = ""
        for i in range(len(prob)):
            disease, rate = prob[i]
            rate = str(rate)
            if i == 0:
                reply_arr.append(TextSendMessage(text= "目前初步" + disease + " " + rate))
                disease1 = disease
            else:
                reply_arr.append(TextSendMessage(text= "第" + str(i + 1) + "可能的" + disease + " " + rate))
        line_bot_api.reply_message(event.reply_token, reply_arr)
        
        binary_check = count_binary(UserId, prob)  # 是否要做二分法的判斷條件

        with open(UserId+'_user_data.txt', 'a',encoding="utf_8") as f:
                f.write(input_message)
                f.write("\n")
                f.write(str(after_fuzz))
                f.write("\n")
                f.write(str(binary_check))
                f.write("\n")

        if binary_check == True:
            decisiontree(UserId)

            binary_method(UserId)

            inquiry_arr = inquiry(UserId, "start")
            line_bot_api.push_message(UserId, TemplateSendMessage(alt_text='Buttons template', template=ButtonsTemplate(title= "發現有症狀機率相似，做二分法", text= "請問是否有" + inquiry_arr[0] + "的症狀?", actions=[MessageTemplateAction(label= "是", text= "是"), MessageTemplateAction(label= "否", text= "否")])))
        
            with open(UserId+'_timeout_event.txt', 'a',encoding="utf_8") as f:
                f.write(str(event))
                f.write("\n")
            timeout(UserId)   

        else:
            line_bot_api.push_message(UserId, TextSendMessage(text = "故診斷為" + disease1))
            write_in_DB(UserId, input_message, after_fuzz, disease1, binary_check)
            delete_afterfuzz(UserId)

if __name__ == "__main__":
    # port = int(os.environ.get('PORT', 5000))
    # app.run(host='0.0.0.0', port=port)

    run_with_ngrok(app)
    app.run()

    # UI網址: https://manager.line.biz/account/@767ykkih/richmenu