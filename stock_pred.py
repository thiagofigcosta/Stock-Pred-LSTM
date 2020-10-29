#!/bin/python3
# -*- coding: utf-8 -*-

import time
import urllib
import urllib.request
import os
import codecs
import json
import pandas as pd
import numpy as np
import random as rd
import re
import math
import shutil
import sys, getopt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import joblib
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping


DATASET_PATH='datasets/'
MODELS_PATH='saved_models/'
DATE_FORMAT='%d/%m/%Y'
DATETIME_FORMAT='"%d/%m/%Y %H:%M:%S"'


def createFolderIfNotExists(path){
    if not os.path.exists(path){
        os.makedirs(path, exist_ok=True)
    }
}

def stringToSTimestamp(string,include_time=False,date_format=DATE_FORMAT){
    if not include_time {
        return int(time.mktime(datetime.strptime(string,date_format).timetuple()))
    }else{
        return int(time.mktime(datetime.strptime(string,DATETIME_FORMAT).timetuple()))
    }
}

def sTimestampToString(timestamp,include_time=False,date_format=DATE_FORMAT){
    if not include_time {
        return datetime.fromtimestamp(timestamp).strftime(date_format)
    }else{
        return datetime.fromtimestamp(timestamp).strftime(DATETIME_FORMAT)
    }
}

def filterNullLines(content){
    lines=content.split('\n')
    content=''
    for line in lines{
        if not re.match(r'((.*null|nan),.*|.*,(null|nan).*)',line, flags=re.IGNORECASE){
            content+=line+'\n'
        }
    }
    return content[:-1]
}

def removeStrPrefix(text, prefix){
    if text.startswith(prefix){
        return text[len(prefix):]
    }
    return text 
}

def jsonToCSV(json_str){
    parsed_json=json.loads(json_str)
    timestamps=parsed_json['spark']['result'][0]['response'][0]['timestamp']
    close_values=parsed_json['spark']['result'][0]['response'][0]['indicators']['quote'][0]['close']

    if len(timestamps)!=len(close_values){
        raise Exception('Stock timestamp array({}) with different size from stock values array({})'.format(len(timestamps),len(close_values)))
    }

    CSV='timestamp,Date,Close\n'
    for i in range(len(timestamps)){
        CSV+='{},{},{}\n'.format(timestamps[i],sTimestampToString(timestamps[i],True),close_values[i])
    }
    return CSV[:-1]
}

def getStockHistoryOneDay(stock_name,filename,start_date=sTimestampToString(0),end_date=sTimestampToString(int(time.time()))){
    base_url='https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events=history&includeAdjustedClose=true'.format(stock_name,stringToSTimestamp(start_date),stringToSTimestamp(end_date))
    print(base_url)
    with urllib.request.urlopen(base_url) as response{ 
        if response.code == 200{
            content=response.read().decode('utf-8')
            createFolderIfNotExists(DATASET_PATH)
            if filename.startswith(DATASET_PATH){
                path=filename
            }else{
                path=DATASET_PATH+filename
            }
            content=filterNullLines(content)
            with codecs.open(path, "w", "utf-8") as file{
                file.write(content)
            }
        }else{
            raise Exception('Response code {}'.format(response.code))
        }
    }
}

def getStockOnlineHistoryOneHour(stock_name,filename,data_range='730d',start_timestamp='',end_timestamp=''){
    # data_range maximum value is 730d
    # data_range minimum value is 1m
    if not start_timestamp or not end_timestamp{
        base_url='https://query1.finance.yahoo.com/v7/finance/spark?symbols={}&range={}&interval=60m&indicators=close&includeTimestamps=false&includePrePost=false&corsDomain=finance.yahoo.com&.tsrc=finance'.format(stock_name,data_range)
    }else{
        base_url='https://query1.finance.yahoo.com/v7/finance/spark?symbols={}&period1={}&period2={}&interval=60m&indicators=close&includeTimestamps=false&includePrePost=false&corsDomain=finance.yahoo.com&.tsrc=finance'.format(stock_name,start_timestamp,end_timestamp)
    } 
    print(base_url)
    with urllib.request.urlopen(base_url) as response{
        if response.code == 200{
            content=response.read().decode('utf-8')
            createFolderIfNotExists(DATASET_PATH)
            if filename.startswith(DATASET_PATH){
                path=filename
            }else{
                path=DATASET_PATH+filename
            }
            content=jsonToCSV(content)
            content=filterNullLines(content)
            with codecs.open(path, "w", "utf-8") as file{
                file.write(content)
            }
        }else{
            raise Exception('Response code {} - {}'.format(response.code))
        }
    }
}

def filenameFromPath(path,get_extension=False){
    if get_extension {
        re_result=re.search(r'.*\/(.*\..+)', path)
        return re_result.group(1) if re_result is not None else path
    }else{
        re_result=re.search(r'.*\/(.*)\..+', path)
        return re_result.group(1) if re_result is not None else path
    }
} 

def extractFromStrDate(str_date){
    re_result=re.search(r'([0-9][0-9])\/([0-9][0-9])\/([0-9][0-9][0-9][0-9]|[0-9][0-9])', str_date)
    return re_result.group(1),re_result.group(2),re_result.group(3)
}   

def saveObj(obj,path){ 
    joblib.dump(obj, path)
}

def loadObj(path){
    return joblib.load(path)
}

def parseHist(history){
    new_hist = {}
    for key in list(history.history.keys()){
        new_hist[key]=history.history[key]
        if type(history.history[key]) == np.ndarray{
            new_hist[key] = history.history[key].tolist()
        }elif type(history.history[key]) == list{
           if  type(history.history[key][0]) == np.float64{
                new_hist[key] = list(map(float, history.history[key]))
            }
        }
    }    
    return new_hist
}

def saveHist(parsed_history,path){
    with codecs.open(path, 'w', encoding='utf-8') as file{
        json.dump(parsed_history, file, separators=(',', ':'), sort_keys=True, indent=4) 
    }
}

def loadHist(path){
    with codecs.open(path, 'r', encoding='utf-8') as file{
        n = json.loads(file.read())
    }
    return n
}

def modeNextDifference(array){
    diff=[]
    for i in range(len(array)-1){
        if str(type(array[i]))=="<class 'pandas._libs.tslibs.timestamps.Timestamp'>"{
            diff.append(float((array[i+1]-array[i]) / np.timedelta64(1, 'h')))
        }else{
            diff.append(array[i+1]-array[i])
        }
    }
    return max(set(diff), key=diff.count)
}

def unwrapFoldedArray(array,use_last=False,use_mean=False,magic_offset=0){
    fold_size=len(array[0])
    array_size=len(array)
    unwraped_size=array_size+fold_size-1
    if use_mean{
        aux_sum_array_tuple=([0]*unwraped_size,[0]*unwraped_size)
        for i in range(magic_offset,array_size){
            for j in range(fold_size){
                aux_sum_array_tuple[0][i+j]+=array[i][j]
                aux_sum_array_tuple[1][i+j]+=1
            }
        }
        unwraped=[]
        for i in range(magic_offset,unwraped_size){
            unwraped.append(aux_sum_array_tuple[0][i]/aux_sum_array_tuple[1][i])
        }
    }else{
        position=0
        if use_last{
            #then use last
            position=fold_size-1
        }
        unwraped=[array[i][position] for i in range(magic_offset,array_size)]
        for i in range(1,fold_size){
            unwraped.append(array[array_size-1][i])
        }
    }
    return unwraped    
}

def extractIfList(y,last_instead_of_all_but_last=False){
    new_y=[]
    for x in y{
        if isinstance(x, (list,pd.core.series.Series,np.ndarray)){
            if last_instead_of_all_but_last{
                new_y.append(x[-1])
            }else{
                new_y.append(x[:-1])
            }
        }else{
            new_y.append(x)
        }
    }
    return new_y
}

def binarySearch(lis,el){ # list must be sorted
    low=0
    high=len(lis)-1
    ret=None 
    while low<=high{
        mid=(low+high)//2
        if el<lis[mid]{
            high=mid-1
        }elif el>lis[mid]{
            low=mid+1
        }else{
            ret=mid
            break
        }
    }
    return ret
}

def alignAndCropTwoArrays(first,second,reverse=False){
    sorted_second=second
    sorted_second.sort()
    used_first=first.copy()
    if reverse{
        used_first.reverse()
    }
    common=None
    for el in used_first{
        ind=binarySearch(sorted_second,el)
        if ind is not None{
            common=el
            break
        }
    }
    if common is None{
        raise Exception('No common element between arrays')
    }else{
        if reverse{
            return first[:first.index(common)+1], second[:second.index(common)+1], common
        }else{
            return first[first.index(common):], second[second.index(common):], common
        }
    }
}

def alignIndexesOnFirstCommonValue(array_of_indexes,reverse=False){
    start=0
    last_common=None
    limit=len(array_of_indexes)-1
    while start<limit{
        f_array,s_array,common=alignAndCropTwoArrays(array_of_indexes[start],array_of_indexes[start+1],reverse=reverse)
        array_of_indexes[start]=f_array
        array_of_indexes[start+1]=s_array
        if common != last_common{
            if last_common is not None{
                start=-1
            }
            last_common=common
        }
        start+=1
    }
    if reverse{
        [el.reverse() for el in array_of_indexes]
    }
    return array_of_indexes
}

def alignOnFirstAndLastCommonIndexes(array_of_data){
    array_of_indexes=[]
    for data in array_of_data{
        array_of_indexes.append(data.index.tolist())
    }
    aligned_array_of_indexes=alignIndexesOnFirstCommonValue(array_of_indexes)
    for i in range(len(array_of_data)){
        array_of_data[i]=array_of_data[i][aligned_array_of_indexes[i][0]:]
    }
    aligned_array_of_indexes=alignIndexesOnFirstCommonValue(array_of_indexes,reverse=True)
    for i in range(len(array_of_data)){
        array_of_data[i]=array_of_data[i][:aligned_array_of_indexes[i][0]]
    }
    return array_of_data
}

# input_size is the amount of data points taken into account to predict output_size amount of data points
def loadDataset(paths,input_size,output_size,company_index_array=[0],train_fields=['Close'],result_field='Close',index_field=None,normalize=False,plot_dataset=False,train_percent=1,val_percent=0,from_date=None){
    if val_percent>1 or train_percent>1 or val_percent<0 or train_percent<0{
        raise Exception('Train + validation percent must be smaller than 1 and bigger than 0')
    }
    if not isinstance(paths, list){
        paths=[paths]
    }

    full_data=[]
    amount_of_companies=len(set(company_index_array))
    if amount_of_companies>1{
        if len(company_index_array)!=len(paths){
            raise Exception('Company index array ({}) must have the same lenght than Paths array ({}) '.format(len(company_index_arary),len(paths)))
        }
        dataset_name=[]
        frames=[]
        last_company=None
        for i in range(len(company_index_array)){
            company=company_index_array[i]
            path=paths[i]
            if last_company != company{
                last_company=company
                current_filename=filenameFromPath(path)
                if i!=len(company_index_array)-1{
                    current_filename=current_filename.split('_')[0]
                }
                dataset_name.append(current_filename)
                if len(frames)>0{
                    full_data.append(pd.concat(frames))
                    frames=[]
                }
            }
            frames.append(pd.read_csv(path))
        }
        dataset_name='+'.join(dataset_name)
        if len(frames)>0{
            full_data.append(pd.concat(frames))
            frames=[]
        }
    }else{
        dataset_name=filenameFromPath(paths[0])
        if len(paths)>1{
            dataset_name+=str(len(paths))
        }
        frames=[]
        for path in paths{
            frames.append(pd.read_csv(path))
        }
        full_data.append(pd.concat(frames))
    }

    fields=train_fields
    if result_field not in fields{
        fields.append(result_field)
    }
    fields.remove(result_field)
    fields.append(result_field) # ensure that the last one is the result field
    
    for i in range(len(full_data)){
        date_index_array=None
        if index_field is not None{
            date_index_array = pd.to_datetime(full_data[i][index_field])
            if from_date is not None{
                from_date_formated=sTimestampToString(stringToSTimestamp(from_date),date_format='%Y-%m-%d')
                date_index_array=date_index_array[date_index_array >= from_date_formated]
            }
            full_data[i][index_field] = date_index_array
            full_data[i].set_index(index_field, inplace=True)
        }
        full_data[i]=full_data[i][fields]
        if from_date is not None{
            full_data[i]= full_data[i][pd.to_datetime(from_date,format=DATE_FORMAT):]
            d,m,y=extractFromStrDate(from_date)
            dataset_name+='trunc{}{}{}'.format(y,m,d)
        }
        if plot_dataset{ 
            if amount_of_companies==1 {
                label='Stock Values of {}'.format(dataset_name)
            }else{
                label='Stock Values Company {} from {}'.format(i+1,dataset_name)
            }
            plt.plot(full_data[i], label=label)
            plt.legend(loc='best')
            plt.show() 
        }
    }

    if amount_of_companies==1 {
        full_data=full_data[0].values.reshape(full_data[0].shape[0],len(fields))
    }else{
        full_data=alignOnFirstAndLastCommonIndexes(full_data)
        date_index_array = full_data[0].index
        for i in range(len(full_data)){
            full_data[i]=full_data[i].values.reshape(full_data[i].shape[0],len(fields))
        }
        tuple_of_companies=tuple(full_data)
        full_data = np.concatenate((tuple_of_companies), axis=1)
    }

    unitedScaler = MinMaxScaler(feature_range=(0, 1))
    if normalize{
        unitedScaler = unitedScaler.fit(full_data)
        print('Min: {}, Max: {}'.format(unitedScaler.data_min_, unitedScaler.data_max_))
        full_plot_data = unitedScaler.transform(full_data)
        if len(train_fields) == 1{ 
            full_data=full_plot_data
        }
        if plot_dataset{ 
            plt.plot(full_plot_data, label='Stock Values of {} normalized'.format(dataset_name))
            plt.legend(loc='best')
            plt.show() 
        }
    }

    data_size=len(full_data)

    X_full_data=[]
    Y_full_data=[]
    if amount_of_companies==1 {
        for i in range(input_size,data_size-output_size+1){
            X_full_data.append(np.array([x[:len(train_fields)] for x in full_data[i-input_size:i]]))
            Y_full_data.append(np.array([x[-1] for x in full_data[i:i+output_size]]))
        }
        # going beyond labels
        for i in range(data_size-output_size+1,data_size+1){
            X_full_data.append(np.array([x[:len(train_fields)] for x in full_data[i-input_size:i]]))
        }
    }else{
         for i in range(input_size,data_size-output_size+1){
            X_full_data.append(np.array([extractIfList(x) for x in full_data[i-input_size:i]]))
            Y_full_data.append(np.array([extractIfList(x,last_instead_of_all_but_last=True) for x in full_data[i:i+output_size]]))
        }
        # going beyond labels
        for i in range(data_size-output_size+1,data_size+1){
            X_full_data.append(np.array([extractIfList(x) for x in full_data[i-input_size:i]]))
        }
    }
    X_full_data=np.array(X_full_data)
    Y_full_data=np.array(Y_full_data)

    if amount_of_companies>1{
        Y_shape=Y_full_data.shape
        Y_full_data = Y_full_data.reshape(Y_shape[0], Y_shape[1]*Y_shape[2])
    }
    
    if len(train_fields) > 1 or amount_of_companies>1{
        scalerX = MinMaxScaler(feature_range=(0, 1))
        scalerY = MinMaxScaler(feature_range=(0, 1))
        if normalize{
            instances, time_steps, features = X_full_data.shape
            X_full_data = np.reshape(X_full_data, newshape=(-1, features))
            scalerX = scalerX.fit(X_full_data)
            scalerY = scalerY.fit(Y_full_data)
            X_full_data = scalerX.transform(X_full_data)
            Y_full_data = scalerY.transform(Y_full_data)
            X_full_data = np.reshape(X_full_data, newshape=(instances, time_steps, features))
        }
        firstScaler=scalerX
        secondScaler=scalerY
    }else{
        firstScaler=unitedScaler
        secondScaler=None
    }

    train_idx=int(len(X_full_data)*train_percent)

    X_train_full=X_full_data[:train_idx]
    Y_train_full=Y_full_data[:train_idx]
    X_test=X_full_data[train_idx:]
    Y_test=Y_full_data[train_idx:]
    X_val=[]
    Y_val=[] 

    date_index_array=date_index_array.tolist()
    train_date_index=date_index_array[:train_idx+input_size+1]
    test_date_index=date_index_array[train_idx:]
    minutes_step=int(modeNextDifference(date_index_array)*60)
    last_date=date_index_array[len(date_index_array)-1].to_numpy()
    for i in range(1,output_size){
        new_date=pd.to_datetime(last_date+np.timedelta64(minutes_step*i,'m'))
        test_date_index.append(new_date)
    }

    if val_percent>0{
        X_train,X_val,Y_train,Y_val = train_test_split(X_train_full, Y_train_full, test_size=val_percent)
    }else{
        X_train=X_train_full
        Y_train=Y_train_full
    }

    return X_train,Y_train,X_val,Y_val,X_test,Y_test,firstScaler,secondScaler,X_train_full,Y_train_full,train_date_index,test_date_index,dataset_name
}

def processPredictedArray(Y_pred){
    magic_offset=1 # align pred with real
    Y_pred_first_val=unwrapFoldedArray(Y_pred,magic_offset=0)
    Y_pred_last_val=unwrapFoldedArray(Y_pred,use_last=True,magic_offset=0)
    Y_pred_mean_val=unwrapFoldedArray(Y_pred,use_mean=True,magic_offset=0)

    Y_pred_first_val=Y_pred_first_val[magic_offset:]
    Y_pred_last_val=Y_pred_last_val[magic_offset:]
    Y_pred_mean_val=Y_pred_mean_val[magic_offset:]
    Y_pred_fl_mean_val=[(Y_pred_first_val[i]+Y_pred_last_val[i])/2 for i in range(len(Y_pred_first_val))]

    return Y_pred_first_val, Y_pred_last_val, Y_pred_mean_val, Y_pred_fl_mean_val
}

def getStockReturn(stock_array){
    shifted_stock_array=stock_array[1:]+[0]
    stock_delta=((np.array(shifted_stock_array)-np.array(stock_array))[:-1]).tolist()#+[None]
    return stock_delta
}

def sum(array){
    sum=0
    for el in array{
        sum+=el
    }
    return sum
}

def mean(array){
    return sum(array)/len(array)
}

def printDict(dictionary,name=None){
    start=''
    if name is not None{
        print('{}:'.format(name))
        start='\t'
    }
    for key,value in dictionary.items(){
        print('{}{}: {}'.format(start,key,value))
    }
}

def analyzeStrategiesAndClassMetrics(stock_real_array,stock_pred_array){
    real_stock_delta=getStockReturn(stock_real_array)
    pred_stock_delta=getStockReturn(stock_pred_array)

    real_movement_encoded=[ 1 if r>0 else 0 for r in real_stock_delta]
    pred_movement_encoded=[ 1 if pred_stock_delta[i]>0 else 0 for i in range(len(real_stock_delta))]
    real_movement=[ 'Up' if r>0 else 'Down' for r in real_stock_delta]
    pred_movement=[ 'Up' if r>0 else 'Down' for r in pred_stock_delta]
    correct_movement=[ 1 if real_movement[i]==pred_movement[i] else 0 for i in range(len(real_stock_delta))]
    accuracy=mean(correct_movement)

    swing_return=[real_stock_delta[i] if pred_movement[i] == 'Up' else 0 for i in range(len(real_stock_delta))]
    swing_return=sum(swing_return) # se a ação subir investe, caso contrario não faz nada
    buy_and_hold_return=sum(real_stock_delta) # compra a ação e segura durante todo o periodo

    class_metrics={'f1_monark':f1_score(real_movement_encoded,pred_movement_encoded),'accuracy':accuracy,'precision':precision_score(real_movement_encoded,pred_movement_encoded),'recall':recall_score(real_movement_encoded,pred_movement_encoded),'roc auc':roc_auc_score(real_movement_encoded,pred_movement_encoded)}
    return swing_return, buy_and_hold_return, class_metrics
}

def autoBuy13(total_money_to_invest,stock_real_array,stock_pred_array,saving_percentage=0.13){
    real_stock_delta=getStockReturn(stock_real_array)
    pred_stock_delta=getStockReturn(stock_pred_array)
    e=math.e
    corret_predicts_in_a_row=0
    savings_money=0
    current_money=total_money_to_invest
    for i in range(len(real_stock_delta)){
        if pred_stock_delta[i] > 0 and stock_real_array[i-1]>0{
            stocks_to_buy=0
            try{
                stock_buy_price=stock_real_array[i-1]
                stock_predicted_sell_price=stock_pred_array[i]
                predicted_valuing=stock_predicted_sell_price/stock_buy_price
                max_stocks_possible=math.floor(current_money/stock_buy_price)
                if (max_stocks_possible<0){
                    max_stocks_possible=0
                }
                if corret_predicts_in_a_row ==0{
                    lucky_randomness=rd.uniform(.02,.07)
                }elif corret_predicts_in_a_row ==1{
                    lucky_randomness=rd.uniform(.04,.09)
                }elif corret_predicts_in_a_row >=2{
                    extra=(corret_predicts_in_a_row/7)
                    if extra>1{
                        extra=1
                    }
                    lucky_randomness=rd.uniform(.07,.13)+.13*extra
                }
                confidence=(-1+(e**(predicted_valuing**.6/1.13))**0.5)/5
                multiplier=(lucky_randomness+confidence)/2
                if multiplier > 0.23{
                    multiplier=0.23
                }
                stocks_to_buy=math.ceil(max_stocks_possible*multiplier)
                if stocks_to_buy<2{
                    stocks_to_buy=2
                }
            }except Exception as e{
                print("Error on auto13")
                print(type(e))  
                print(e.args)     
                print(e)
            }
            if real_stock_delta[i]<0{
                corret_predicts_in_a_row=0
                current_money+=real_stock_delta[i]*stocks_to_buy
            }else{
                corret_predicts_in_a_row+=1
                current_money+=real_stock_delta[i]*stocks_to_buy*(1-saving_percentage)
                savings_money+=real_stock_delta[i]*stocks_to_buy*saving_percentage
            }
        }
    }
    return current_money+savings_money-total_money_to_invest
}

def calculateLayerOutputSize(layer_input_size,network_output_size,train_data_size=0,a=2,second_formula=False){
    if not second_formula{
        return int(math.ceil(train_data_size/(a*(layer_input_size+network_output_size))))
    }else{
        return int(math.ceil(2/3*(layer_input_size+network_output_size)))
    }
}

def modeler(id=0,train_size=13666,input_features=['Close'],amount_companies=1){ # default train_size based on GE data
    model_base_name='model_id-{}'.format(id)
    input_features=len(input_features)
    hyperparameters={}
    if id==0{
        pass # final model will be here
        # since the best models are 2 and 11, there is no final model
    }elif id==1{
        a=2 # from 2 to 8
        hyperparameters['backwards_samples']=40
        hyperparameters['forward_samples']=15
        hyperparameters['lstm_layers']=2
        hyperparameters['max_epochs']=200
        hyperparameters['patience_epochs']=10
        hyperparameters['batch_size']=5
        hyperparameters['stateful']=False
        hyperparameters['dropout_values']=[0,0]
        hyperparameters['lstm_l1_size']=60
        hyperparameters['lstm_l2_size']=35
        hyperparameters['normalize']=True
        hyperparameters['optimizer']='adam'
        hyperparameters['model_metrics']=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
        hyperparameters['loss']='mean_squared_error'
        hyperparameters['train_percent']=.8
        hyperparameters['val_percent']=.2
    }elif id==2{
        a=2 # from 2 to 8
        hyperparameters['backwards_samples']=40
        hyperparameters['forward_samples']=15
        hyperparameters['lstm_layers']=2
        hyperparameters['max_epochs']=200
        hyperparameters['patience_epochs']=10
        hyperparameters['batch_size']=1
        hyperparameters['stateful']=True
        hyperparameters['dropout_values']=[0,0]
        hyperparameters['lstm_l1_size']=60
        hyperparameters['lstm_l2_size']=35
        hyperparameters['normalize']=True
        hyperparameters['optimizer']='adam'
        hyperparameters['model_metrics']=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
        hyperparameters['loss']='mean_squared_error'
        hyperparameters['train_percent']=.8
        hyperparameters['val_percent']=.2
    }elif id==3{ #cams
        a=4 # from 2 to 8
        hyperparameters['backwards_samples']=30
        hyperparameters['forward_samples']=15
        hyperparameters['lstm_layers']=3
        hyperparameters['max_epochs']=200
        hyperparameters['patience_epochs']=10
        hyperparameters['batch_size']=1
        hyperparameters['stateful']=True
        hyperparameters['dropout_values']=[.2,.2,.2]
        hyperparameters['lstm_l1_size']=calculateLayerOutputSize(hyperparameters['backwards_samples'],hyperparameters['forward_samples'],train_data_size=train_size,a=a,second_formula=False)
        hyperparameters['lstm_l2_size']=calculateLayerOutputSize(hyperparameters['lstm_l1_size'],hyperparameters['forward_samples'],train_data_size=train_size,a=a,second_formula=False)
        hyperparameters['lstm_l3_size']=calculateLayerOutputSize(hyperparameters['lstm_l2_size'],hyperparameters['forward_samples'],train_data_size=train_size,a=a,second_formula=False)
        hyperparameters['normalize']=True
        hyperparameters['optimizer']='adam'
        hyperparameters['model_metrics']=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
        hyperparameters['loss']='mean_squared_error'
        hyperparameters['train_percent']=.8
        hyperparameters['val_percent']=.2
    }elif id==4 {
        a=2 # from 2 to 8
        hyperparameters['backwards_samples']=59
        hyperparameters['forward_samples']=15
        hyperparameters['lstm_layers']=3
        hyperparameters['max_epochs']=200
        hyperparameters['patience_epochs']=10
        hyperparameters['batch_size']=5
        hyperparameters['stateful']=False
        hyperparameters['dropout_values']=[.2,.2,.2]
        hyperparameters['lstm_l1_size']=calculateLayerOutputSize(hyperparameters['backwards_samples'],hyperparameters['forward_samples'],train_data_size=train_size,a=a,second_formula=False)
        hyperparameters['lstm_l2_size']=calculateLayerOutputSize(hyperparameters['lstm_l1_size'],hyperparameters['forward_samples'],train_data_size=train_size,a=a,second_formula=False)
        hyperparameters['lstm_l3_size']=calculateLayerOutputSize(hyperparameters['lstm_l2_size'],hyperparameters['forward_samples'],train_data_size=train_size,a=a,second_formula=False)
        hyperparameters['normalize']=True
        hyperparameters['optimizer']='adam'
        hyperparameters['model_metrics']=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
        hyperparameters['loss']='mean_squared_error'
        hyperparameters['train_percent']=.8
        hyperparameters['val_percent']=.2
    }elif id==5 {
        a=2 # from 2 to 8
        hyperparameters['backwards_samples']=30
        hyperparameters['forward_samples']=5
        hyperparameters['lstm_layers']=2
        hyperparameters['max_epochs']=200
        hyperparameters['patience_epochs']=15
        hyperparameters['batch_size']=5
        hyperparameters['stateful']=False
        hyperparameters['dropout_values']=[.14,0]
        hyperparameters['lstm_l1_size']=40
        hyperparameters['lstm_l2_size']=20
        hyperparameters['normalize']=True
        hyperparameters['optimizer']='adam'
        hyperparameters['model_metrics']=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
        hyperparameters['loss']='mean_squared_error'
        hyperparameters['train_percent']=.8
        hyperparameters['val_percent']=.2
    }elif id==6 {
        a=2 # from 2 to 8
        hyperparameters['backwards_samples']=10
        hyperparameters['forward_samples']=3
        hyperparameters['lstm_layers']=2
        hyperparameters['max_epochs']=200
        hyperparameters['patience_epochs']=10
        hyperparameters['batch_size']=1
        hyperparameters['stateful']=True
        hyperparameters['dropout_values']=[.6,.1]
        hyperparameters['lstm_l1_size']=200
        hyperparameters['lstm_l2_size']=20
        hyperparameters['normalize']=True
        hyperparameters['optimizer']='adam'
        hyperparameters['model_metrics']=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
        hyperparameters['loss']='mean_squared_error'
        hyperparameters['train_percent']=.8
        hyperparameters['val_percent']=.2
    }elif id==7 {
        a=2 # from 2 to 8
        hyperparameters['backwards_samples']=90
        hyperparameters['forward_samples']=5
        hyperparameters['lstm_layers']=3
        hyperparameters['max_epochs']=200
        hyperparameters['patience_epochs']=10
        hyperparameters['batch_size']=10
        hyperparameters['stateful']=False
        hyperparameters['dropout_values']=[.3,.5,.1]
        hyperparameters['lstm_l1_size']=80
        hyperparameters['lstm_l2_size']=150
        hyperparameters['lstm_l3_size']=20
        hyperparameters['normalize']=True
        hyperparameters['optimizer']='adam'
        hyperparameters['model_metrics']=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
        hyperparameters['loss']='mean_squared_error'
        hyperparameters['train_percent']=.8
        hyperparameters['val_percent']=.2
    }elif id==8 {
        a=2 # from 2 to 8
        hyperparameters['backwards_samples']=7
        hyperparameters['forward_samples']=3
        hyperparameters['lstm_layers']=2
        hyperparameters['max_epochs']=200
        hyperparameters['patience_epochs']=15
        hyperparameters['batch_size']=7
        hyperparameters['stateful']=False
        hyperparameters['dropout_values']=[.4,.2]
        hyperparameters['lstm_l1_size']=120
        hyperparameters['lstm_l2_size']=40
        hyperparameters['normalize']=True
        hyperparameters['optimizer']='adam'
        hyperparameters['model_metrics']=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
        hyperparameters['loss']='mean_squared_error'
        hyperparameters['train_percent']=.8
        hyperparameters['val_percent']=.2
    }elif id==9 {
        a=2 # from 2 to 8
        hyperparameters['backwards_samples']=7
        hyperparameters['forward_samples']=3
        hyperparameters['lstm_layers']=2
        hyperparameters['max_epochs']=200
        hyperparameters['patience_epochs']=15
        hyperparameters['batch_size']=1
        hyperparameters['stateful']=True
        hyperparameters['dropout_values']=[.4,.2]
        hyperparameters['lstm_l1_size']=120
        hyperparameters['lstm_l2_size']=40
        hyperparameters['normalize']=True
        hyperparameters['optimizer']='adam'
        hyperparameters['model_metrics']=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
        hyperparameters['loss']='mean_squared_error'
        hyperparameters['train_percent']=.8
        hyperparameters['val_percent']=.2
    }elif id==10 {
        a=2 # from 2 to 8
        hyperparameters['backwards_samples']=15
        hyperparameters['forward_samples']=5
        hyperparameters['lstm_layers']=2
        hyperparameters['max_epochs']=200
        hyperparameters['patience_epochs']=15
        hyperparameters['batch_size']=7
        hyperparameters['stateful']=False
        hyperparameters['dropout_values']=[.2,.2]
        hyperparameters['lstm_l1_size']=50
        hyperparameters['lstm_l2_size']=10
        hyperparameters['normalize']=True
        hyperparameters['optimizer']='adam'
        hyperparameters['model_metrics']=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
        hyperparameters['loss']='mean_squared_error'
        hyperparameters['train_percent']=.8
        hyperparameters['val_percent']=.2
    }elif id==11 {
        a=2 # from 2 to 8
        hyperparameters['backwards_samples']=66
        hyperparameters['forward_samples']=6
        hyperparameters['lstm_layers']=2
        hyperparameters['max_epochs']=200
        hyperparameters['patience_epochs']=15
        hyperparameters['batch_size']=6
        hyperparameters['stateful']=False
        hyperparameters['dropout_values']=[.3,.3]
        hyperparameters['lstm_l1_size']=66
        hyperparameters['lstm_l2_size']=33
        hyperparameters['normalize']=True
        hyperparameters['optimizer']='adam'
        hyperparameters['model_metrics']=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
        hyperparameters['loss']='mean_squared_error'
        hyperparameters['train_percent']=.7
        hyperparameters['val_percent']=.2
    }elif id==12 {
        a=2 # from 2 to 8
        hyperparameters['backwards_samples']=20
        hyperparameters['forward_samples']=10
        hyperparameters['lstm_layers']=2
        hyperparameters['max_epochs']=200
        hyperparameters['patience_epochs']=6
        hyperparameters['batch_size']=7
        hyperparameters['stateful']=False
        hyperparameters['dropout_values']=[.6,.3]
        hyperparameters['lstm_l1_size']=70
        hyperparameters['lstm_l2_size']=30
        hyperparameters['normalize']=True
        hyperparameters['optimizer']='adam'
        hyperparameters['model_metrics']=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
        hyperparameters['loss']='mean_squared_error'
        hyperparameters['train_percent']=.8
        hyperparameters['val_percent']=.2
    }elif id==13 {
        a=2 # from 2 to 8
        hyperparameters['backwards_samples']=33
        hyperparameters['forward_samples']=15
        hyperparameters['lstm_layers']=2
        hyperparameters['max_epochs']=200
        hyperparameters['patience_epochs']=10
        hyperparameters['batch_size']=8
        hyperparameters['stateful']=False
        hyperparameters['dropout_values']=[.2,.2]
        hyperparameters['lstm_l1_size']=calculateLayerOutputSize(hyperparameters['backwards_samples'],hyperparameters['forward_samples'],train_data_size=train_size,a=a,second_formula=True)
        hyperparameters['lstm_l2_size']=calculateLayerOutputSize(hyperparameters['lstm_l1_size'],hyperparameters['forward_samples'],train_data_size=train_size,a=a,second_formula=True)
        hyperparameters['normalize']=True
        hyperparameters['optimizer']='adam'
        hyperparameters['model_metrics']=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
        hyperparameters['loss']='mean_squared_error'
        hyperparameters['train_percent']=.8
        hyperparameters['val_percent']=.2
    }elif id==14 {
        a=2 # from 2 to 8
        hyperparameters['backwards_samples']=33
        hyperparameters['forward_samples']=15
        hyperparameters['lstm_layers']=2
        hyperparameters['max_epochs']=200
        hyperparameters['patience_epochs']=10
        hyperparameters['batch_size']=1
        hyperparameters['stateful']=True
        hyperparameters['dropout_values']=[.2,.2]
        hyperparameters['lstm_l1_size']=calculateLayerOutputSize(hyperparameters['backwards_samples'],hyperparameters['forward_samples'],train_data_size=train_size,a=a,second_formula=True)
        hyperparameters['lstm_l2_size']=calculateLayerOutputSize(hyperparameters['lstm_l1_size'],hyperparameters['forward_samples'],train_data_size=train_size,a=a,second_formula=True)
        hyperparameters['normalize']=True
        hyperparameters['optimizer']='adam'
        hyperparameters['model_metrics']=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
        hyperparameters['loss']='mean_squared_error'
        hyperparameters['train_percent']=.8
        hyperparameters['val_percent']=.2
    }elif id==15 {
        a=2 # from 2 to 8
        hyperparameters['backwards_samples']=33
        hyperparameters['forward_samples']=5
        hyperparameters['lstm_layers']=2
        hyperparameters['max_epochs']=200
        hyperparameters['patience_epochs']=15
        hyperparameters['batch_size']=2
        hyperparameters['stateful']=False
        hyperparameters['dropout_values']=[.2,.2]
        hyperparameters['lstm_l1_size']=calculateLayerOutputSize(hyperparameters['backwards_samples'],hyperparameters['forward_samples'],train_data_size=train_size,a=a,second_formula=True)
        hyperparameters['lstm_l2_size']=calculateLayerOutputSize(hyperparameters['lstm_l1_size'],hyperparameters['forward_samples'],train_data_size=train_size,a=a,second_formula=True)
        hyperparameters['normalize']=True
        hyperparameters['optimizer']='adam'
        hyperparameters['model_metrics']=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
        hyperparameters['loss']='mean_squared_error'
        hyperparameters['train_percent']=.8
        hyperparameters['val_percent']=.2
    }else{
        return None
    }

    hyperparameters['id']=id
    hyperparameters['base_name']=model_base_name
    hyperparameters['input_features']=input_features
    hyperparameters['amount_companies']=amount_companies
    if hyperparameters['stateful']{
        hyperparameters['batch_size']=1 # batch size must be one for stateful
    }
    return hyperparameters
}

def modelBuilder(hyperparameters,stock_name,print_summary=True){
    model = Sequential()
    hyperparameters['lstm_l0_size']=hyperparameters['backwards_samples']
    for l in range(hyperparameters['lstm_layers']){
        if hyperparameters['input_features']>1 and hyperparameters['amount_companies']>1{
            raise Exception('Only input_features or amount_companies must be greater than 1')
        }
        deepness=1
        if hyperparameters['input_features']>1{
            deepness=hyperparameters['input_features']
        }elif hyperparameters['amount_companies']>1{
            deepness=hyperparameters['amount_companies']
        }
        input_shape=(hyperparameters['lstm_l{}_size'.format(l)],deepness)
        if hyperparameters['stateful'] {
            if l==0{
                batch_input_shape=(hyperparameters['batch_size'],hyperparameters['lstm_l{}_size'.format(l)],deepness)
                model.add(LSTM(hyperparameters['lstm_l{}_size'.format(l+1)],batch_input_shape=batch_input_shape, stateful=hyperparameters['stateful'], return_sequences=True if l+1<hyperparameters['lstm_layers'] else False))
            }else{
                model.add(LSTM(hyperparameters['lstm_l{}_size'.format(l+1)],input_shape=input_shape, stateful=hyperparameters['stateful'], return_sequences=True if l+1<hyperparameters['lstm_layers'] else False))
            }
        }else{
            model.add(LSTM(hyperparameters['lstm_l{}_size'.format(l+1)],input_shape=input_shape, stateful=hyperparameters['stateful'], return_sequences=True if l+1<hyperparameters['lstm_layers'] else False))
        }
        if hyperparameters['dropout_values'][l]>0{
            model.add(Dropout(hyperparameters['dropout_values'][l]))
        }
    }
    model.add(Dense(hyperparameters['forward_samples']*hyperparameters['amount_companies']))
    if print_summary {
        print(model.summary())
    }
    model.compile(loss=hyperparameters['loss'], optimizer=hyperparameters['optimizer'],metrics=hyperparameters['model_metrics'])
    early_stopping=EarlyStopping(monitor='val_loss', mode='min', patience=hyperparameters['patience_epochs'], verbose=1)
    checkpoint = ModelCheckpoint(MODELS_PATH+hyperparameters['base_name']+'_'+stock_name+'_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks=[early_stopping,checkpoint]
    return model, callbacks
}

def fixIfStatefulModel(hyperparameters,model,stock_name){
    new_model=model
    if hyperparameters['stateful']{ # workaround because model.predict was not working for trained stateful models
        hyperparameters['stateful']=False
        new_model,_=modelBuilder(hyperparameters,stock_name,print_summary=False)
        new_model.set_weights(model.get_weights()) 
    }
    return new_model
}


def uncompactMultiCompanyArray(compacted_array,amount_companies){
    shape=compacted_array.shape
    newshape=(shape[0], int(shape[1]/amount_companies), amount_companies)
    return np.reshape(compacted_array, newshape=newshape)
}

def isolateMultiCompanyArray(uncompacted_array,amount_companies){
    isolated_array=[]
    for i in range(amount_companies){
        isolated_array.append([])
    }
    for samples in uncompacted_array{
        for i in range(amount_companies){
            company_sample=[]
            for j in range(len(samples)){
                company_sample.append(samples[j][i])
            }
            isolated_array[i].append(company_sample)
        }
    }
    return np.array(isolated_array)
}

def loadTrainAndSaveModel(model_id,dataset_paths=[],load_instead_of_training=False,plot_graphs=True,train_fields=['Close'],company_index_array=[0],from_date=None){
    hyperparameters=modeler(id=model_id,input_features=train_fields,amount_companies=len(set(company_index_array))) 
    investiment=22000  
    
    X_train,Y_train,X_val,Y_val,X_test,Y_test,firstScaler,secondScaler,_,Y_train_full,train_date_index,test_date_index,stock_name = loadDataset(
        dataset_paths,hyperparameters['backwards_samples'],hyperparameters['forward_samples'],index_field='Date',train_fields=train_fields,company_index_array=company_index_array,
            normalize=hyperparameters['normalize'],plot_dataset=plot_graphs,train_percent=hyperparameters['train_percent'],val_percent=hyperparameters['val_percent'],from_date=from_date)
    
    if len(train_fields)>1{
        hyperparameters['base_name']+='-IFs{}'.format(len(train_fields))
    }
    model_path=MODELS_PATH+hyperparameters['base_name']+'_'+stock_name
    model_model_path=model_path+'.h5'
    model_hyperparam_path=model_path+'_hyperparams.json'
    model_metrics_path=model_path+'_metrics.json'
    if secondScaler is not None{
        model_scalerX_path=model_path+'_scalerX.bin'
        model_scalerY_path=model_path+'_scalerY.bin'
    }else{
        model_scaler_path=model_path+'_scaler.bin'
    }
    model_history_path=model_path+'_history.json'
    createFolderIfNotExists(MODELS_PATH)

    if load_instead_of_training{
        model = load_model(model_model_path)
        model = fixIfStatefulModel(hyperparameters,model,stock_name)
        with open(model_hyperparam_path, 'r') as fp {
            hyperparameters=json.load(fp)
        }
        if secondScaler is not None{
            scalerX=loadObj(model_scalerX_path)
            scalerY=loadObj(model_scalerY_path)
        }else{
            scaler=loadObj(model_scaler_path)
        }
        history=loadHist(model_history_path)
    }else{
        model,callbacks=modelBuilder(hyperparameters,stock_name)  
        history=model.fit(X_train,Y_train,epochs=hyperparameters['max_epochs'],validation_data=(X_val,Y_val),batch_size=hyperparameters['batch_size'],callbacks=callbacks,shuffle=True,verbose=2)
        history=parseHist(history)
        model=fixIfStatefulModel(hyperparameters,model,stock_name)
        model.save(model_model_path) 
        print('Model saved at {};'.format(model_model_path))
        with open(model_hyperparam_path, 'w') as fp{
            json.dump(hyperparameters, fp)
        }
        print('Model Hyperparameters saved at {};'.format(model_hyperparam_path))   
        if secondScaler is not None{
            scalerX=firstScaler
            scalerY=secondScaler
            saveObj(scalerX,model_scalerX_path)  
            print('Model ScalerX saved at {};'.format(model_scalerX_path))
            saveObj(scalerY,model_scalerY_path)  
            print('Model ScalerY saved at {};'.format(model_scalerY_path))
        }else{
            scaler=firstScaler
            saveObj(scaler,model_scaler_path)  
            print('Model ScalerY saved at {};'.format(model_scaler_path))
        }
        saveHist(history,model_history_path)  
        print('Model History saved at {};'.format(model_history_path)) 
    }

    Y_test_predicted = model.predict(X_test)
    Y_train_predicted = model.predict(X_train)

    if hyperparameters['normalize'] {
        if secondScaler is not None{
            Y_test_predicted=scalerY.inverse_transform(Y_test_predicted)
            Y_train_predicted=scalerY.inverse_transform(Y_train_full)
            Y_train_full=scalerY.inverse_transform(Y_train_full)
            Y_test=scalerY.inverse_transform(Y_test)
        }else{
            Y_test_predicted=scaler.inverse_transform(Y_test_predicted)
            Y_train_predicted=scaler.inverse_transform(Y_train_full)
            Y_train_full=scaler.inverse_transform(Y_train_full)
            Y_test=scaler.inverse_transform(Y_test)
        }
    }

    if plot_graphs{
        plt.plot(history['loss'], label='loss')
        plt.plot(history['val_loss'], label='val_loss')
        plt.legend(loc='best')
        plt.title('Training loss of {}'.format(stock_name))
        plt.show() 
    }

    if hyperparameters['amount_companies']>1{
        model_metrics=model.evaluate(X_test[:len(Y_test)],Y_test)
        aux={}
        for i in range(len(model_metrics)){
            aux[model.metrics_names[i]] = model_metrics[i]
        }
        model_metrics=aux

        Y_test_predicted=uncompactMultiCompanyArray(Y_test_predicted,hyperparameters['amount_companies'])
        Y_train_predicted=uncompactMultiCompanyArray(Y_train_predicted,hyperparameters['amount_companies'])
        Y_train_full=uncompactMultiCompanyArray(Y_train_full,hyperparameters['amount_companies'])
        Y_test=uncompactMultiCompanyArray(Y_test,hyperparameters['amount_companies'])

        Y_test_predicted=isolateMultiCompanyArray(Y_test_predicted,hyperparameters['amount_companies'])
        Y_train_predicted=isolateMultiCompanyArray(Y_train_predicted,hyperparameters['amount_companies'])
        Y_train_full=isolateMultiCompanyArray(Y_train_full,hyperparameters['amount_companies'])
        Y_test=isolateMultiCompanyArray(Y_test,hyperparameters['amount_companies'])

        metrics={'Strategy Metrics':[],'Model Metrics':model_metrics,'Class Metrics':[]}

        for i in range(hyperparameters['amount_companies']){
            Y_test_unwraped=unwrapFoldedArray(Y_test[i])
            Y_test_pred_first_val, Y_test_pred_last_val,\
                Y_test_pred_mean_val, Y_test_pred_fl_mean_val=processPredictedArray(Y_test_predicted[i])
            Y_full=unwrapFoldedArray(np.vstack((Y_train_full[i],Y_test[i])))
            Y_train_pred_first_val, Y_train_pred_last_val,\
                Y_train_pred_mean_val, Y_train_pred_fl_mean_val=processPredictedArray(Y_train_predicted[i])

            swing_return,buy_hold_return,class_metrics_tmp=analyzeStrategiesAndClassMetrics(Y_test_unwraped,Y_test_pred_fl_mean_val)
            viniccius13_return=autoBuy13(investiment,Y_test_unwraped,Y_test_pred_fl_mean_val)
            strategy_metrics={'Company':'{} of {}'.format(i+1,hyperparameters['amount_companies']),'Daily Swing Trade Return':swing_return,'Buy & Hold Return':buy_hold_return,'Auto13(${}) Return'.format(investiment):viniccius13_return}

            class_metrics={'Company':'{} of {}'.format(i+1,hyperparameters['amount_companies'])}
            for key, value in class_metrics_tmp.items(){
                class_metrics[key]=value
            }

            metrics['Strategy Metrics'].append(strategy_metrics)
            metrics['Class Metrics'].append(class_metrics)

            printDict(model_metrics,'Model metrics')
            printDict(class_metrics,'Class metrics')
            printDict(strategy_metrics,'Strategy metrics')

            if plot_graphs{
                try {
                    magic=0 # hyperparameters['amount_companies']-1 #?
                    input_size=hyperparameters['backwards_samples']
                    output_size=hyperparameters['forward_samples']

                    plt.plot(test_date_index[input_size+magic:-output_size+1],Y_test_unwraped, label='Real')
                    plt.plot(test_date_index[input_size+magic:],Y_test_pred_first_val, color='r', label='Predicted F')
                    plt.plot(test_date_index[input_size+magic:],Y_test_pred_last_val, label='Predicted L')
                    plt.plot(test_date_index[input_size+magic:],Y_test_pred_mean_val, label='Predicted Mean')
                    plt.plot(test_date_index[input_size+magic:],Y_test_pred_fl_mean_val, label='Predicted FL Mean')
                    plt.title('Testing stock values {} - Company {} of {}'.format(stock_name,i+1,hyperparameters['amount_companies']))
                    plt.legend(loc='best')
                    plt.show() 

                    full_date_index=train_date_index[1:-input_size]+test_date_index
                    plt.plot(full_date_index[input_size+magic:-(output_size-1)],Y_full, label='Real Full data')
                    plt.plot(train_date_index[input_size+2+magic:],Y_train_pred_mean_val[:-(output_size-1)], label='Train Predicted Mean')
                    plt.plot(train_date_index[input_size+2+magic:],Y_train_pred_fl_mean_val[:-(output_size-1)], label='Train Predicted FL Mean')
                    plt.plot(test_date_index[input_size+magic:],Y_test_pred_mean_val, label='Test Predicted Mean')
                    plt.plot(test_date_index[input_size+magic:],Y_test_pred_fl_mean_val, label='Test Predicted FL Mean')
                    plt.title('Complete stock values {} - Company {} of {}'.format(stock_name,i+1,hyperparameters['amount_companies']))
                    plt.legend(loc='best')
                    plt.show() 
                }except Exception as e {
                    print("Error on plot (single company)")
                    print(type(e))  
                    print(e.args)     
                    print(e)
                }
            }
        }

        with open(model_metrics_path, 'w') as fp{
            json.dump(metrics, fp)
        }
        print('Model Metrics saved at {};'.format(model_metrics_path))
    }else{
        Y_test_unwraped=unwrapFoldedArray(Y_test)
        Y_test_pred_first_val, Y_test_pred_last_val,\
            Y_test_pred_mean_val, Y_test_pred_fl_mean_val=processPredictedArray(Y_test_predicted)
        Y_full=unwrapFoldedArray(np.vstack((Y_train_full,Y_test)))
        Y_train_pred_first_val, Y_train_pred_last_val,\
            Y_train_pred_mean_val, Y_train_pred_fl_mean_val=processPredictedArray(Y_train_predicted)
        

        model_metrics=model.evaluate(X_test[:len(Y_test)],Y_test)
        aux={}
        for i in range(len(model_metrics)){
            aux[model.metrics_names[i]] = model_metrics[i]
        }
        model_metrics=aux
        swing_return,buy_hold_return,class_metrics=analyzeStrategiesAndClassMetrics(Y_test_unwraped,Y_test_pred_fl_mean_val)
        viniccius13_return=autoBuy13(investiment,Y_test_unwraped,Y_test_pred_fl_mean_val)
        strategy_metrics={'Daily Swing Trade Return':swing_return,'Buy & Hold Return':buy_hold_return,'Auto13(${}) Return'.format(investiment):viniccius13_return}
        metrics={'Strategy Metrics':strategy_metrics,'Model Metrics':model_metrics,'Class Metrics':class_metrics}
        with open(model_metrics_path, 'w') as fp{
            json.dump(metrics, fp)
        }
        print('Model Metrics saved at {};'.format(model_metrics_path))

        printDict(model_metrics,'Model metrics')
        printDict(class_metrics,'Class metrics')
        printDict(strategy_metrics,'Strategy metrics')

        if plot_graphs{
            try{
                input_size=hyperparameters['backwards_samples']
                output_size=hyperparameters['forward_samples']
                plt.plot(test_date_index[input_size:-output_size+1],Y_test_unwraped, label='Real')
                plt.plot(test_date_index[input_size:],Y_test_pred_first_val, color='r', label='Predicted F')
                plt.plot(test_date_index[input_size:],Y_test_pred_last_val, label='Predicted L')
                plt.plot(test_date_index[input_size:],Y_test_pred_mean_val, label='Predicted Mean')
                plt.plot(test_date_index[input_size:],Y_test_pred_fl_mean_val, label='Predicted FL Mean')
                plt.title('Testing stock values {}'.format(stock_name))
                plt.legend(loc='best')
                plt.show() 

                full_date_index=train_date_index[1:-input_size]+test_date_index
                plt.plot(full_date_index[input_size:-(output_size-1)],Y_full, label='Real Full data')
                plt.plot(train_date_index[input_size+2:],Y_train_pred_mean_val[:-(output_size-1)], label='Train Predicted Mean')
                plt.plot(train_date_index[input_size+2:],Y_train_pred_fl_mean_val[:-(output_size-1)], label='Train Predicted FL Mean')
                plt.plot(test_date_index[input_size:],Y_test_pred_mean_val, label='Test Predicted Mean')
                plt.plot(test_date_index[input_size:],Y_test_pred_fl_mean_val, label='Test Predicted FL Mean')
                plt.title('Complete stock values {}'.format(stock_name))
                plt.legend(loc='best')
                plt.show() 
            }except Exception as e {
                print("Error on plot (single company)")
                print(type(e))  
                print(e.args)     
                print(e)
            }
        }
    }
}

def downloadAllReferenceDatasets(){
    limit_date_2020='21/10/2020'

    QP1_3_6_start_date=sTimestampToString(0)
    QP1_3_6_end_date=limit_date_2020
    QP1_3_6_stocks=['AAPL','AZUL','BTC-USD','GE','PBR','TSLA','VALE','MSFT','GOGL','AMZN','IBM','T','FB','YOJ.SG','KO']
    for stock in QP1_3_6_stocks{
        _,month,year=extractFromStrDate(QP1_3_6_end_date)
        filename=DATASET_PATH+'{}_I1d_F0_T{}-{}.csv'.format(stock,year,month)
        getStockHistoryOneDay(stock,filename,start_date=QP1_3_6_start_date,end_date=QP1_3_6_end_date)
    }

    QP4_start_dates=['01/01/2010','01/01/2011','01/01/2012','01/01/2013','01/01/2014','01/01/2015','01/01/2016','01/01/2017','01/01/2018','01/01/2019','01/01/2020']
    QP4_end_dates=  ['31/12/2010','31/12/2011','31/12/2012','31/12/2013','31/12/2014','31/12/2015','31/12/2016','31/12/2017','31/12/2018','31/12/2019',limit_date_2020]
    QP4_stocks=['PBR','TSLA','BTC-USD','VALE']
    for stock in QP4_stocks{
        for i in range(len(QP4_start_dates)){
            filename=DATASET_PATH+'{}_I1h_R1y_S{}.csv'.format(stock,extractFromStrDate(QP4_start_dates[i])[2])
            getStockOnlineHistoryOneHour(stock,filename,start_timestamp=stringToSTimestamp(QP4_start_dates[i]),end_timestamp=stringToSTimestamp(QP4_end_dates[i]))
        }
    }
}

def restoreBestModelCheckpoint(){
    print_models=False
    models={}
    for file_str in os.listdir(MODELS_PATH){
        re_result=re.search(r'model_id-([0-9]+(?:-?[a-zA-Z]*?[0-9]*?)_.*?(?=_)_I[0-9]+[a-zA-Z]+).*\.(h5|json)', file_str)
        if re_result{
            model_id=re_result.group(1)
            if model_id not in models{
                models[model_id]=[file_str]
            }else{
                models[model_id].append(file_str)
            }
        }
    }
    if print_models{
        models_list = list(models.keys())
        models_list.sort()
        for key in models_list{
            print('Keys: {} len: {}'.format(key,len(models[key])))
        }
    }
    for _,files in models.items(){
        checkpoint_filename=None
        model_filename=None
        metrics_filename=None
        last_patience_filename=None
        for file in files{
            if re.search(r'model_id-[0-9]+.*_checkpoint\.h5', file){
                checkpoint_filename=file
            }elif re.search(r'model_id-[0-9]+.*(?<![_checkpoint|_last_patience])\.h5', file){
                model_filename=file
            }elif re.search(r'model_id-[0-9]+.*(?<!_last_patience)_metrics\.json', file){
                metrics_filename=file
            }elif re.search(r'model_id-[0-9]+.*_last_patience\.h5', file){
                last_patience_filename=file
            }      
        }
        if checkpoint_filename is not None and model_filename is not None and last_patience_filename is None{
            print('Restoring checkpoint {}'.format(checkpoint_filename))
            shutil.move(MODELS_PATH+model_filename,MODELS_PATH+model_filename.split('.')[0]+'_last_patience.h5')
            shutil.move(MODELS_PATH+checkpoint_filename,MODELS_PATH+model_filename)
            if metrics_filename is not None{
                shutil.move(MODELS_PATH+metrics_filename,MODELS_PATH+metrics_filename.split('_metrics')[0]+'_last_patience_metrics.json')
            }
        }
    }
}

def trainAllProposedTestModels(dataset_paths,start_at=0,plot_and_load=False){
    last_test_model_id=None
    test_id=1
    while last_test_model_id is None{
        if modeler(test_id) is None{
            last_test_model_id=test_id
        }else{
         test_id+=1
        }
    }
    for i in range(start_at+1,last_test_model_id){
        print("Model {}".format(i))
        loadTrainAndSaveModel(model_id=i,dataset_paths=dataset_paths,load_instead_of_training=plot_and_load,plot_graphs=plot_and_load)
    }
}


def QP1(plot_and_load=True){
    dataset_paths=['datasets/GE_I1d_F0_T2020-10.csv','datasets/AAPL_I1d_F0_T2020-10.csv','datasets/AZUL_I1d_F0_T2020-10.csv','datasets/BTC-USD_I1d_F0_T2020-10.csv','datasets/PBR_I1d_F0_T2020-10.csv','datasets/TSLA_I1d_F0_T2020-10.csv','datasets/VALE_I1d_F0_T2020-10.csv']
    for dataset in dataset_paths{
        trainAllProposedTestModels(dataset,plot_and_load=plot_and_load)
    }
}

def QP2(plot_and_load=True){
    dataset_paths=['datasets/GE_I1d_F0_T2020-10.csv','datasets/AAPL_I1d_F0_T2020-10.csv','datasets/AZUL_I1d_F0_T2020-10.csv','datasets/BTC-USD_I1d_F0_T2020-10.csv','datasets/PBR_I1d_F0_T2020-10.csv','datasets/TSLA_I1d_F0_T2020-10.csv','datasets/VALE_I1d_F0_T2020-10.csv']
    for dataset in dataset_paths{
        loadTrainAndSaveModel(model_id=2,train_fields=['Open','High','Low','Close','Adj Close','Volume'],dataset_paths=dataset,load_instead_of_training=plot_and_load,plot_graphs=plot_and_load)
        loadTrainAndSaveModel(model_id=11,train_fields=['Open','High','Low','Close','Adj Close','Volume'],dataset_paths=dataset,load_instead_of_training=plot_and_load,plot_graphs=plot_and_load)
    }
}

def QP3(plot_and_load=True){
    dataset_paths=['datasets/AAPL_I1d_F0_T2020-10.csv','datasets/MSFT_I1d_F0_T2020-10.csv','datasets/GOGL_I1d_F0_T2020-10.csv','datasets/AMZN_I1d_F0_T2020-10.csv','datasets/IBM_I1d_F0_T2020-10.csv','datasets/T_I1d_F0_T2020-10.csv','datasets/FB_I1d_F0_T2020-10.csv','datasets/YOJ.SG_I1d_F0_T2020-10.csv']
    company_index_array=[0,1,2,3,4,5,6,7]
    loadTrainAndSaveModel(model_id=2,dataset_paths=dataset_paths,company_index_array=company_index_array,load_instead_of_training=plot_and_load,plot_graphs=plot_and_load)
    loadTrainAndSaveModel(model_id=11,dataset_paths=dataset_paths,company_index_array=company_index_array,load_instead_of_training=plot_and_load,plot_graphs=plot_and_load)
    for dataset in dataset_paths{
        loadTrainAndSaveModel(model_id=2,dataset_paths=dataset,from_date='29/04/2013',load_instead_of_training=plot_and_load,plot_graphs=plot_and_load)
        loadTrainAndSaveModel(model_id=11,dataset_paths=dataset,from_date='29/04/2013',load_instead_of_training=plot_and_load,plot_graphs=plot_and_load)
    }
}

def QP4(plot_and_load=True){
    dataset_paths=[['datasets/PBR_I1h_R1y_S2010.csv','datasets/PBR_I1h_R1y_S2011.csv',
    'datasets/PBR_I1h_R1y_S2012.csv','datasets/PBR_I1h_R1y_S2013.csv','datasets/PBR_I1h_R1y_S2014.csv',
    'datasets/PBR_I1h_R1y_S2015.csv','datasets/PBR_I1h_R1y_S2016.csv','datasets/PBR_I1h_R1y_S2017.csv',
    'datasets/PBR_I1h_R1y_S2018.csv','datasets/PBR_I1h_R1y_S2019.csv','datasets/PBR_I1h_R1y_S2020.csv'],
    ['datasets/TSLA_I1h_R1y_S2010.csv','datasets/TSLA_I1h_R1y_S2011.csv','datasets/TSLA_I1h_R1y_S2012.csv',
    'datasets/TSLA_I1h_R1y_S2013.csv','datasets/TSLA_I1h_R1y_S2014.csv','datasets/TSLA_I1h_R1y_S2015.csv',
    'datasets/TSLA_I1h_R1y_S2016.csv','datasets/TSLA_I1h_R1y_S2017.csv','datasets/TSLA_I1h_R1y_S2018.csv',
    'datasets/TSLA_I1h_R1y_S2019.csv','datasets/TSLA_I1h_R1y_S2020.csv'],
    ['datasets/BTC-USD_I1h_R1y_S2010.csv','datasets/BTC-USD_I1h_R1y_S2011.csv','datasets/BTC-USD_I1h_R1y_S2012.csv',
    'datasets/BTC-USD_I1h_R1y_S2013.csv','datasets/BTC-USD_I1h_R1y_S2014.csv','datasets/BTC-USD_I1h_R1y_S2015.csv',
    'datasets/BTC-USD_I1h_R1y_S2016.csv','datasets/BTC-USD_I1h_R1y_S2017.csv','datasets/BTC-USD_I1h_R1y_S2018.csv',
    'datasets/BTC-USD_I1h_R1y_S2019.csv','datasets/BTC-USD_I1h_R1y_S2020.csv'],
    ['datasets/VALE_I1h_R1y_S2010.csv','datasets/VALE_I1h_R1y_S2011.csv','datasets/VALE_I1h_R1y_S2012.csv',
    'datasets/VALE_I1h_R1y_S2013.csv','datasets/VALE_I1h_R1y_S2014.csv','datasets/VALE_I1h_R1y_S2015.csv',
    'datasets/VALE_I1h_R1y_S2016.csv','datasets/VALE_I1h_R1y_S2017.csv','datasets/VALE_I1h_R1y_S2018.csv',
    'datasets/VALE_I1h_R1y_S2019.csv','datasets/VALE_I1h_R1y_S2020.csv']] 
    for dataset in dataset_paths{
        loadTrainAndSaveModel(model_id=2,dataset_paths=dataset,load_instead_of_training=plot_and_load,plot_graphs=plot_and_load)
        loadTrainAndSaveModel(model_id=11,dataset_paths=dataset,load_instead_of_training=plot_and_load,plot_graphs=plot_and_load)
    }
}

def QP6(plot_and_load=True){
    dataset_paths=['datasets/IBM_I1d_F0_T2020-10.csv','KO_I1d_F0_T2020-10.csv']
    for dataset in dataset_paths{
        loadTrainAndSaveModel(model_id=2,dataset_paths=dataset,load_instead_of_training=plot_and_load,plot_graphs=plot_and_load)
        loadTrainAndSaveModel(model_id=11,dataset_paths=dataset,load_instead_of_training=plot_and_load,plot_graphs=plot_and_load)
    }
}

def main(argv){
    HELP_STR=r'Pytho{N}.py stock_pred.py [-d|--download-datasets] [[--qp1 | --qp2 | --qp3 | --qp4 | --qp6] [--train_without_plot]] [--test-all-test-trained-models [--start-at <value>] --dataset-paths <values>] [--restore-best-checkpoints] [--train-all-test-models [--start-at <value>] --dataset-paths <values>] [--download-stock [--use-hour-interval] --stock-name <value> --start-date <value> --end-date <value>]'
    modules=["download-datasets","download-stock","train-all-test-models","test-all-test-trained-models","restore-best-checkpoints","qp1","qp2","qp3","qp4","qp5"]
    modules_to_run=[]
    args=[]
    
    use_hour_interval=False
    stock_name=''
    start_date=''
    end_date=''
    start_at=0
    plot_and_load=True
    dataset_paths=[]

    try{
        opts, args = getopt.getopt(argv,"hd",["use-hour-interval","stock-name=","start-date=","end-date=","start-at=","dataset-paths=","train_without_plot"]+modules)
    }except getopt.GetoptError{
        print (HELP_STR)
        sys.exit(2)
    }
    for opt, arg in opts{
        if opt == '-h'{
            print (HELP_STR)
            sys.exit()
        }elif opt == "--use-hour-interval"{
            use_hour_interval=True
        }elif opt == "--stock-name"{
            stock_name=arg
        }elif opt == "--start-date"{
            try{
                extractFromStrDate(arg)
            }except{
                raise Exception('Date must be in format {}'.format(DATE_FORMAT))
            }
            start_date=arg
        }elif opt == "--end-date"{
            try{
                extractFromStrDate(arg)
            }except{
                raise Exception('Date must be in format {}'.format(DATE_FORMAT))
            }
            end_date=arg
        }elif opt == "--start-at"{
            start_at=int(arg)
        }elif opt == "--dataset-paths"{
            dataset_paths=arg.split(',')
        }elif opt == "--train_without_plot"{
            plot_and_load=False
        }else{
            modules_to_run.append(opt)
        }
    }
    for module in modules_to_run{
        module=removeStrPrefix(module,'--')
        if module == "download-datasets"{
            downloadAllReferenceDatasets()
        }elif module == "train-all-test-models"{
            trainAllProposedTestModels(dataset_paths,start_at=start_at)
        }elif module == "test-all-test-trained-models"{ 
            trainAllProposedTestModels(dataset_paths,start_at=start_at,plot_and_load=True)
        }elif module == "restore-best-checkpoints"{
            restoreBestModelCheckpoint()
        }elif module == "qp1"{
            QP1(plot_and_load=plot_and_load)
        }elif module == "qp2"{
            QP2(plot_and_load=plot_and_load)
        }elif module == "qp3"{
            QP3(plot_and_load=plot_and_load)
        }elif module == "qp4"{
            QP4(plot_and_load=plot_and_load)
        }elif module == "qp6"{
            QP6(plot_and_load=plot_and_load)
        }elif module == "download-stock"{
            start_day,start_month,start_year=extractFromStrDate(start_date)
            end_day,end_month,end_year=extractFromStrDate(end_date)
            if use_hour_interval {
                filename=DATASET_PATH+'{}_I1h_F{}{}{}_T{}{}{}.csv'.format(stock,start_year,start_month,start_day,end_year,end_month,end_day)
                getStockOnlineHistoryOneHour(stock,filename,start_timestamp=stringToSTimestamp(start_date),end_timestamp=stringToSTimestamp(end_date))
            }else{
                filename=DATASET_PATH+'{}_I1d_F{}{}{}_T{}{}{}.csv'.format(stock,start_year,start_month,start_day,end_year,end_month,end_day)
                getStockHistoryOneDay(stock,filename,start_date=start_date,end_date=end_date)
            }
        }else{
            print("Unkown argument {}".format(module))
            print(HELP_STR)
            sys.exit(2)
        }
    }
}

if __name__ == "__main__"{
    delta=-time.time()
    main(sys.argv[1:])
    delta+=time.time()
    print("\n\nTotal delta time is {} s".format(delta))
}