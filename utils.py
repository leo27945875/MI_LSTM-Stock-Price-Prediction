import args
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
import copy
import json
import time
from collections import defaultdict
from shutil import copyfile
from torch.utils.data import Dataset


# 訓練模型時用來產生batch的Dataset類別:
class TimeSeriesDataset(Dataset):
    def __init__(self, timeSeriesDataList, timeSeriesTargetData, isRegression):
        self.timeSeriesDataList = [torch.from_numpy(data) for data in timeSeriesDataList]
        self.timeSeriesTargetData = torch.from_numpy(timeSeriesTargetData)
        self.isRegression = isRegression
    
    
    def __getitem__(self, i):
        if self.isRegression:
            return ([data[i].float() for data in self.timeSeriesDataList], 
                    self.timeSeriesTargetData[i].unsqueeze(-1).float())
        else:
            return ([data[i].float() for data in self.timeSeriesDataList], 
                    JudgeStockTommorowTrend(self.timeSeriesDataList[0][i], 
                                            self.timeSeriesTargetData[i]))
    
    def __len__(self):
        return self.timeSeriesTargetData.shape[0]


# 取得股票走勢(漲: 0; 跌: 2; 打平: 1):
def JudgeStockTommorowTrend(pastYs, tommorowY, checkRate=args.JUDGE_STOCK_TREND_RATE):
    nowY = pastYs.ravel()[-1]
    rate = (tommorowY - nowY) / nowY
    checkRate = abs(checkRate)
    if rate > checkRate:    
        return torch.tensor(0, dtype=torch.long)
    elif rate < -checkRate: 
        return torch.tensor(2, dtype=torch.long)
    else:                   
        return torch.tensor(1, dtype=torch.long)


# 讀取存有所有股票資料的pickle檔案:
def GetOriginData(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# 將每個企業股票DataFrame同樣的column取出來並接再一起:
def ConcatSameColumn(stocks, column):
    sameColData = {}
    for sym, stock in stocks.items():
        sameColData[sym] = stock[column].values
    
    sameColData = pd.DataFrame(sameColData)
    sameColData["Date"] = stock["Date"]
    return sameColData


# 計算出MI-LSTM所需要的[Y], [Xi], [Xp], [Xn] 。
def GetFactors(df, yCol, indexCol):
    df = df.drop("Date", axis=1)
    
    # Self:
    y = df.pop(yCol).values.reshape(-1, 1)
    
    # Index:
    index = df.pop(indexCol).values.reshape(-1, 1)
    
    # Positive & Negative:
    l = df.shape[1]
    corrList = [[df.iloc[:, i].name, np.corrcoef(y.ravel(), df.iloc[:, i].values)[0, 1]] for i in range(l)]
    corrList = sorted(corrList, key=lambda x: x[1])
    splitIndex = len(corrList) // 2
    posColumn  = [c[0] for c in corrList[:splitIndex]]
    negColumn  = [c[0] for c in corrList[splitIndex:]]
    pos = df.loc[:, posColumn].values
    neg = df.loc[:, negColumn].values
    
    # Output result:
    return y, index, pos, neg


# 取得[Y], [Xi], [Xp], [Xn]後，轉換成時間序列資料集(即Dataset物件):
def MakeTimeSeriesData(dataList, window, isRegression):
    n = dataList[0].shape[0] - window
    timeSeriesDataList = []
    for data in dataList:
        timeSeriesData = np.zeros([n, window, data.shape[1]])
        for i in range(n):
            timeSeriesData[i] = data[i: i + window]
        
        timeSeriesDataList.append(timeSeriesData)
    
    timeSeriesTargetData = np.zeros([n])
    targetColumn = dataList[0]
    for i in range(n):
        timeSeriesTargetData[i] = targetColumn[i + window]
    
    return TimeSeriesDataset(timeSeriesDataList, timeSeriesTargetData, isRegression)


# 生成訓練模型時用來產生batch的Dataset物件，參數<symbol>是指要預測的企業代號:
def MakeDataset(symbol, index, window=30, isRegression=True, feature="Close", path=args.USED_DATA_FILE_PATH):
    stocks = GetOriginData(path)
    sameColData = ConcatSameColumn(stocks, feature)
    factors = GetFactors(sameColData, symbol, index)
    return MakeTimeSeriesData(factors, window, isRegression)


# 視覺化learning rate scheduler:
def PlotScheduler(scheduler, n_iter=2000, title="LR Scheduler"):
    scheduler = copy.deepcopy(scheduler)
    scheduler.verbose = False
    for i in range(n_iter):
        lr = scheduler.get_last_lr()
        plt.plot(i, lr, 'bo')
        scheduler.step()
    
    plt.title(title)
    plt.show()
    scheduler.last_epoch = -1


# 將所有tensor轉換至特定device(如: CPU, GPU ... 等):
def ToDevice(device, tensors):
    return [tensor.to(device) for tensor in tensors]


# 儲存模型參數:
def SaveModel(model, path):
    modelPySrc = os.path.join(os.getcwd(), 'model.py')
    modelPyDst = os.path.splitext(path)[0] + '.py'
    copyfile(modelPySrc, modelPyDst)
    torch.save(model, path)
    

# 輸入每個參數的多個選項，產生grid search用的所有參數組合: 
def CreateGrid(params):
    grid = []
    for key, values in params.items():
        if not grid:
            grid = [{key: value} for value in values]
        else:
            newGrid = []
            for point in grid:
                for value in values:
                    newPoint = copy.deepcopy(point)
                    newPoint.update({key: value})
                    newGrid.append(newPoint)
            
            grid = newGrid
    
    return grid


# 將dict物件儲存成.json檔:
def SaveJSON(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
        

# 尋找所有在[folder]中屬於[symbol]的模型:
def FindModelInfo(symbol, folder):
    filenames = os.listdir(folder)
    filenames = [filename for filename in filenames 
                 if symbol    in os.path.split(filename)[1] and 
                    ".json"   in os.path.split(filename)[1] ]
    for filename in filenames:
        filename = os.path.join(folder, filename)
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                params = json.load(f)
                model  = torch.load(os.path.splitext(filename)[0] + ".pth")
            except Exception as e:
                print("\n" + "-" * 50 + f"\n {e} \n" + "-" * 50 + "\n")
                continue
        
        yield params, model


# 從list中取眾數:
def FindMode(l):
    count = defaultdict(int)
    for num in l:
        count[num] += 1
    
    maxKey, maxValue = None, -1
    for key, value in count.items():
        if value > maxValue:
            maxKey, maxValue = key, value
    
    return maxKey


# 取平均值:
def FindAverage(l):
    return sum(l) / len(l)


# 找出從今天往回算[nDay]天的日期範圍:
def GetDateRange(nDay, endDate=None):
    if endDate:
        endDate = pd.Timestamp(endDate)
    else:
        endDate = pd.Timestamp(pd.to_datetime(time.time(), unit="s").date())
    
    timeDelta = pd.Timedelta(nDay - 1, unit="d")
    startDate = endDate - timeDelta
    return startDate, endDate
    


if __name__ == '__main__':
    datasets = MakeDataset("AMD", "SPY", 10, False)
    
    







