import args
import os
import copy
import torch
import torch.nn as nn
import torch_optimizer as optim
import torch.optim.lr_scheduler as schedule
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from warmup_scheduler import GradualWarmupScheduler
from pprint import pprint
from model import Model, MAPELoss, Accuracy
from utils import (MakeDataset, PlotScheduler, ToDevice, SaveModel, CreateGrid,
                   SaveJSON, FindModelInfo)


class Trainer:
    def __init__(self, modelClass, params, symbols, trainFunc, trainParmas):
        self.modelClass = modelClass
        self.grid = CreateGrid(params)
        self.symbols = symbols
        self.trainParmas = copy.deepcopy(trainParmas)
        self.trainFunc = trainFunc
        self.availableSymbols = set([sym for sym, _ in args.SYMBOLS])
    
    
    def Init(self):
        dropParams = ["isSaveWeights", "modelCount"]
        for dropParam in dropParams:
            if dropParam in self.trainParmas:
                print(f"Force to change [{dropParam}] !")
                self.trainParmas.pop(dropParam)
    
    
    def Train(self):
        self.Init()
        for sym, index in self.symbols:
            if sym in self.availableSymbols:
                for i, point in enumerate(self.grid):
                    print('=' * 30 + f' {sym} ' + '=' * 30)
                    pprint(point)
                    print('-' * 50)
                    model = self.modelClass(**point)
                    _, savePath, _ = self.trainFunc(model, 
                                                    sym, 
                                                    index,
                                                    isSaveWeights=True,
                                                    modelCount=i,
                                                    **self.trainParmas)
                    SaveJSON(point, os.path.splitext(savePath)[0] + '.json')
            else:
                print('=' * 30 + f' {sym} ' + '=' * 30)
                print(f"[{sym}] is not in SYMBOLS !")


class Finetuner:
    def __init__(self, modelClass, symbols, trainFunc, trainParmas):
        self.modelClass = modelClass
        self.symbols = symbols
        self.trainParmas = trainParmas
        self.trainFunc = trainFunc
        self.availableSymbols = set([sym for sym, _ in args.SYMBOLS])
    
    
    def Init(self):
        dropParams = ["isSaveWeights", "epochs", "modelCount"]
        for dropParam in dropParams:
            if dropParam in self.trainParmas:
                print(f"Force to change [{dropParam}] !")
                self.trainParmas.pop(dropParam)
    
    
    def Train(self, modelInfoFolder, epochs, epochsMLP=0, newMLP_Class=None, 
              newMLP_ModelParams={}, newMLP_trainParmas={}):
        self.Init()
        isFinetune = newMLP_Class and epochsMLP
        for sym, index in self.symbols:
            for i, (params, model) in enumerate(FindModelInfo(sym, modelInfoFolder)):
                print('=' * 30 + f' {sym} ' + '=' * 30)
                pprint(params)
                print('-' * 50)
                if isFinetune:
                    print('\n' + '*' * 50 + "\nTraining new MLP ...\n" + '*' * 50 + '\n')
                    classifier = newMLP_Class(model.regressor.regressor.in_features, 
                                              **newMLP_ModelParams)
                    model.requires_grad_(False)
                    model.regressor = classifier
                    model.regressor.requires_grad_()
                    _, _, datasets = self.trainFunc(model, 
                                                    sym, 
                                                    index,
                                                    epochs=epochsMLP,
                                                    isSaveWeights=False,
                                                    saveFolder="",
                                                    **newMLP_trainParmas)
                    
                print('\n' + '*' * 50 + "\nTraining the whole model ...\n" + '*' * 50 + '\n')
                model.requires_grad_()
                _, savePath, _ = self.trainFunc(model, 
                                                sym, 
                                                index,
                                                epochs=epochs,
                                                isSaveWeights=True,
                                                modelCount=i,
                                                datasets=datasets if isFinetune else None,
                                                **self.trainParmas)
                SaveJSON(params, os.path.splitext(savePath)[0] + '.json')


def TrainEpoch(model, metric, optimizer, scheduler, gradClip, trainLoader, device, 
               epoch, epochs, nBatch, isRegression):
    model.train()
    lossEpoch, totalAcc, nIter = 0., 0., 0
    for (y, index, pos, neg), target in trainLoader:
        optimizer.zero_grad()
        y, index, pos, neg, target = ToDevice(device, [y, index, pos, neg, target])
        
        output = model(y, index, pos, neg)
        loss = metric(output, target)
        loss.backward()
        
        clip_grad_norm_(model.parameters(), gradClip)
        optimizer.step()
        scheduler.step()
        
        lossEpoch += loss.item()
        totalAcc  += MAPELoss(output, target) if isRegression else 1 - Accuracy(output, target)
        nIter += 1
        print(f"\r| Epoch {epoch + 1}/{epochs} | Batch {nIter}/{nBatch} | Training Loss = {round(lossEpoch / nIter, 4)} |", end="")
    
    return totalAcc / nIter



def TestModel(model, metric, dataLoader, testType, isRegression):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        model.eval()
        totalLoss, totalMAPE, nIter = 0., 0., 0
        for (y, index, pos, neg), target in dataLoader:
            y, index, pos, neg, target = ToDevice(device, [y, index, pos, neg, target])
            output = model(y, index, pos, neg)
            loss = metric(output, target)
            totalLoss += loss.item()
            totalMAPE += MAPELoss(output, target) if isRegression else 1 - Accuracy(output, target)
            nIter += 1
        
        print(f"| {testType} Loss = {round(totalLoss / nIter, 4)} | Accuracy = {round(1 - totalMAPE / nIter, 4)}")
    
    return totalMAPE / nIter


def TrainModel(model, symbol, index, saveFolder,
               gradClip=1.,
               epochs=1000,
               batchSize=512,
               lr=1e-2,
               isOnlyTrain=False,
               trainRate=0.8,
               validRate=0.1,
               testRate=0.1,
               baselineAcc=0.98,
               isRegression=True,
               isPlotScheduler=False,
               isSaveWeights=True,
               modelCount=None,
               datasets=None):
    
    if trainRate + validRate + testRate != 1:
        raise ValueError("trainRate + validRate + testRate must equal to 1 .")
    
    if datasets:
        trainSet, validSet, testSet = datasets
        trainSize, validSize, testSize = len(trainSet), len(validSet), len(testSet)
    else:
        dataset   = MakeDataset(symbol, index, model.window, isRegression, path=args.USED_DATA_FILE_PATH)
        trainSize = round(len(dataset) * trainRate)
        validSize = round(len(dataset) * validRate)
        testSize  = len(dataset) - trainSize - validSize
        trainSet, validSet, testSet = random_split(dataset, [trainSize, validSize, testSize])
        
    if isOnlyTrain:
        trainLoader = DataLoader(trainSet, batchSize, shuffle=True)
        validLoader = None
        testLoader  = None
    else:
        trainLoader = DataLoader(trainSet, batchSize, shuffle=True) 
        validLoader = DataLoader(validSet, batchSize, shuffle=True) if validRate else None
        testLoader  = DataLoader(testSet , batchSize, shuffle=True) if testRate  else None
    
    nBatch    = (len(trainSet) // batchSize) + 1
    totalIter = epochs * len(trainLoader)
    
    metric    = nn.SmoothL1Loss() if isRegression else nn.CrossEntropyLoss()
    optimizer = optim.AdaBound(model.parameters(), lr)
    scheduler = GradualWarmupScheduler(optimizer,
                                       multiplier=1,
                                       total_epoch=totalIter // 20,
                                       after_scheduler=schedule.StepLR(optimizer, totalIter // 5))
    if isPlotScheduler: 
        PlotScheduler(scheduler, totalIter)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metric.to(device)
    
    baselineEpoch = round(epochs * 0.6)
    minMAPE, bestModel, bestEpoch = float("inf"), None, 0
    for epoch in range(epochs):
        print("-" * 50)
        trainMAPE = TrainEpoch(model, metric, optimizer, scheduler, 
                               gradClip, trainLoader, device, epoch, 
                               epochs, nBatch, isRegression)
        
        print(" => ", end="")
        if not isOnlyTrain and validRate:
            nowMAPE = TestModel(model, metric, validLoader, "Validation", isRegression)
        else:
            nowMAPE = trainMAPE
            print(f"Training Accracy = {1 - nowMAPE}")
        
        if epoch >= baselineEpoch:
            if nowMAPE < minMAPE:
                minMAPE = nowMAPE
                bestModel = model.state_dict()
                bestEpoch = epoch
                
            if epoch - bestEpoch > 100 and 1 - minMAPE > baselineAcc:
                print(f'\n[Early Stopping (Best Accuracy = {1 - minMAPE}) !]\n')
                break
        
    print("=" * 50)
    modelCount = "" if modelCount is None else "_" + str(modelCount)
    model.load_state_dict(bestModel)
    if not isOnlyTrain and testRate:
        testMAPE = TestModel(model, metric, testLoader, "Testing", isRegression)
        if isRegression:
            savePath = os.path.join(saveFolder, f"{symbol}{modelCount}_Acc={1 - testMAPE}.pth")
        else:
            savePath = os.path.join(saveFolder, f"{symbol}{modelCount}_Cls_Acc={1 - testMAPE}.pth")
    else:
        if isRegression:
            savePath = os.path.join(saveFolder, f"{symbol}{modelCount}_Acc={1 - minMAPE}.pth")
        else:
            savePath = os.path.join(saveFolder, f"{symbol}{modelCount}_Cls_Acc={1 - minMAPE}.pth")
    
    if isSaveWeights:
        SaveModel(model, savePath)
        return model, savePath, (trainSet, validSet, testSet)
    else:
        return model, None, (trainSet, validSet, testSet)


if __name__ == "__main__":
    WINDOW                = 20
    INPUT_SZIE            = 1
    EMBED_SIZE_0          = 64
    EMBED_SIZE_1          = 128
    EMBED_SIZE_2          = 128
    ENCODER_LAYERS        = 2
    ENCODER_DROPOUT       = 0.1
    ENCODER_BIDIRECTIONAL = True
    ENCODER_ATTENTION     = True
    ATTENTION_HEADS       = 4
    ATTENTION_LAYERS      = 1
    REGRESSION_LAYERS     = 4
    
    SYMBOL            = "GOOG"
    INDEX             = "SPY"
    EPOCHS            = 1000
    BATCH_SIZE        = 512
    LR                = 1e-2
    GRAD_CLIP         = 1.
    SAVE_FLODER       = r"D:\Downloads\BigData\雲端運算與巨量資料分析\HW\HW2\Save"
    IS_ONLY_TRAIN     = True
    TRAIN_RATE        = 1
    VALID_RATE        = 0
    TEST_RATE         = 0
    BASELINE_ACC      = 0.98
    IS_RERESSION      = True
    IS_PLOT_SCHEDULER = False
    IS_SAVE_WEIGHTS   = True
    MODEL_NUMBER      = None
    DATA_SETS         = None
    
    model = Model(WINDOW,
                  INPUT_SZIE,
                  EMBED_SIZE_0,
                  EMBED_SIZE_1,
                  EMBED_SIZE_2,
                  ENCODER_LAYERS,
                  ENCODER_DROPOUT,
                  ENCODER_BIDIRECTIONAL,
                  ENCODER_ATTENTION,
                  ATTENTION_HEADS,
                  ATTENTION_LAYERS,
                  REGRESSION_LAYERS)
    model, savePath = TrainModel(model,
                                 SYMBOL,
                                 INDEX,
                                 SAVE_FLODER,
                                 GRAD_CLIP,
                                 EPOCHS,
                                 BATCH_SIZE,
                                 LR,
                                 IS_ONLY_TRAIN,
                                 TRAIN_RATE,
                                 VALID_RATE,
                                 TEST_RATE,
                                 BASELINE_ACC,
                                 IS_RERESSION,
                                 IS_PLOT_SCHEDULER,
                                 IS_SAVE_WEIGHTS,
                                 MODEL_NUMBER,
                                 DATA_SETS)







    