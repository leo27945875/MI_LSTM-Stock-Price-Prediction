import torch
import pickle
from utils import FindModelInfo, ConcatSameColumn, GetFactors, FindMode, FindAverage, JudgeStockTommorowTrend


class Ensembler:
    def __init__(self, symbols, feature, modelFolder, dataPath, savePath, isRegression):
        self.symbols = symbols
        self.feature = feature
        self.modelFolder = modelFolder
        self.dataPath = dataPath
        self.savePath = savePath
        self.isRegression = isRegression
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lastResult = None
    
    
    def Predict(self):
        sameColData, result = self.GetSameColumnData(), {}
        print(sameColData)
        for sym, index in self.symbols:
            print(f"Predicting [{sym}] ... ")
            predList = []
            for params, model in FindModelInfo(sym, self.modelFolder):
                model.to(self.device)
                data = self.GetInput(sameColData, sym, index, params["window"])
                pred = self.PredictOne(model, data)
                predList.append(pred)
            
            if self.isRegression:
                result[sym] = JudgeStockTommorowTrend(data[0], FindAverage(predList)).item()
            else:
                result[sym] = FindMode(predList)
                
            print(f"{predList}\n  => {result[sym]}\n")
        
        self.lastResult = result
        self.Output(result)
        return result
    
    
    def PredictOne(self, model, data):
        with torch.no_grad():
            model.eval()
            output = model(*data)
            if self.isRegression:
                output = output.squeeze().item()
            else:
                output = torch.argmax(output, axis=1).item()
                
            return output
    
    
    def GetSameColumnData(self):
        with open(self.dataPath, 'rb') as f:
            data = pickle.load(f)
        
        return ConcatSameColumn(data, self.feature)
    
    
    def GetInput(self, sameColData, symbol, index, nDay):
        factors = GetFactors(sameColData, symbol, index)
        return [torch.from_numpy(factor[-nDay:, :]).unsqueeze(0).to(self.device).float()
                for factor in factors]
    
    
    def Output(self, result):
        with open(self.savePath, 'w') as f:
            for sym, _ in self.symbols:
                f.write(f"{result[sym]}\n")
    
    
    