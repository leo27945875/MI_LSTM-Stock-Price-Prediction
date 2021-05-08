INDEX = "SPY"
SYMBOLS = [["INTC", INDEX],
           ["AMD" , INDEX], 
           ["CSCO", INDEX], 
           ["AAPL", INDEX], 
           ["MU"  , INDEX], 
           ["NVDA", INDEX], 
           ["QCOM", INDEX], 
           ["AMZN", INDEX], 
           ["NFLX", INDEX],
           ["FB"  , INDEX], 
           ["GOOG", INDEX], 
           ["BABA", INDEX], 
           ["EBAY", INDEX], 
           ["IBM" , INDEX], 
           ["XLNX", INDEX], 
           ["TXN" , INDEX], 
           ["NOK" , INDEX], 
           ["TSLA", INDEX], 
           ["MSFT", INDEX], 
           ["SNPS", INDEX]]

TARGET_FEATURE         = "Close"
JUDGE_STOCK_TREND_RATE = 0.015

REGRESSION_MODEL_SAVE_FOLDER     =  "D:/Downloads/BigData/雲端運算與巨量資料分析/HW/HW2/Code/Save"
FINETUNE_MODEL_SAVE_FOLDER       =  "D:/Downloads/BigData/雲端運算與巨量資料分析/HW/HW2/Code/SaveFinetune"
PREDICTION_FILE_SAVE_PATH        =  "D:/Downloads/BigData/雲端運算與巨量資料分析/HW/HW2/Code/Prediction/result.txt"
USED_DATA_FILE_PATH              = f"D:/Downloads/BigData/雲端運算與巨量資料分析/HW/HW2/Code/Data/index_{INDEX}.pkl"


