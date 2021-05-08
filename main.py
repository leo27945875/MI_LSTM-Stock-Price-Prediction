import args
from train import Trainer, Finetuner, TrainModel
from model import Model, ClassifierModel, Classifier
from ensemble import Ensembler


def Train():
    
    isRegression = True
    layersParamName = f"{'regressor' if isRegression else 'classification'}Layers"
    
    trainParameters = {
        "lr":           1e-3,
        "isOnlyTrain":  False,
        "trainRate":    0.8,
        "validRate":    0.1,
        "testRate":     0.1,
        "saveFolder":   args.REGRESSION_MODEL_SAVE_FOLDER,
        "isRegression": isRegression
    }
    
    modelParameters = {
        "window":               [10, 20, 30],
        "inputSize":            [1],
        "embedSize0":           [64],
        "embedSize1":           [128],
        "embedSize2":           [128],
        "encoderLayers":        [2],
        "encoderDropout":       [0.1],
        "encoderBidirectional": [True],
        "encoderAttention":     [True],
        "attentionHeads":       [4],
        "attentionLayers":      [0, 1],
        layersParamName:        [4]
    }
    
    trainer = Trainer(Model if isRegression else ClassifierModel, 
                      modelParameters, 
                      args.SYMBOLS, 
                      TrainModel, 
                      trainParameters)
    trainer.Train()
    return trainer


def Finetune():
    
    isOnlyTrain = True
    trainRate = 1
    validRate = 0
    testRate  = 0
    isRegression = True
    
    trainParameters = {
        "lr":          1e-4,
        "isOnlyTrain": isOnlyTrain,
        "trainRate":   trainRate,
        "validRate":   validRate,
        "testRate":    testRate,
        "isRegression": isRegression,
        "saveFolder":  args.FINETUNE_MODEL_SAVE_FOLDER
    }
    
    # newMLP_trainParmas = {
    #     "lr":           1e-2,
    #     "isOnlyTrain":  isOnlyTrain,
    #     "trainRate":    trainRate,
    #     "validRate":    validRate,
    #     "testRate":     testRate,
    #     "isRegression": isRegression
    # }
    
    # newMLP_ModelParams = {
    #     "layers": 4
    # }
    
    finetuner = Finetuner(Model,
                          args.SYMBOLS,
                          TrainModel,
                          trainParameters)
    finetuner.Train(args.REGRESSION_MODEL_SAVE_FOLDER,
                    500,
                    # 300,
                    # Classifier,
                    # newMLP_ModelParams,
                    # newMLP_trainParmas
                    )
    return finetuner


def Predict():
    isRegression = True
    predictor = Ensembler(args.SYMBOLS,
                          args.TARGET_FEATURE,
                          args.REGRESSION_MODEL_SAVE_FOLDER,
                          args.USED_DATA_FILE_PATH,
                          args.PREDICTION_FILE_SAVE_PATH,
                          isRegression)
    return predictor.Predict()


if __name__ == '__main__':
    # trainer = Train()
    # finetuner = Finetune()
    result = Predict()
    
    
    
    
    
    
    
    
    
    
    
    