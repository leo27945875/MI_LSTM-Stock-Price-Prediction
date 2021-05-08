import pickle
import pandas as pd
import yfinance as yf


SYMBOLS   = ["INTC", "AMD", "CSCO", "AAPL", "MU", "NVDA", "QCOM", "AMZN", "NFLX",
             "FB", "GOOG", "BABA", "EBAY", "IBM", "XLNX", "TXN", "NOK", "TSLA",
             "MSFT", "SNPS"]
INDEX     = "SPY"
SAVE_PATH = f"D:/Downloads/BigData/雲端運算與巨量資料分析/HW/HW2/Data/index_{INDEX}.pkl"


# 填補DataFrame中的缺失值:
def FillNan(df):
    df = df.interpolate(method='polynomial', order=2)
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    return df


# 取得單一股票資料:
def GetStock(symbol):
    n = 0
    while n < 5:
        try:
            print(f"Getting stock of {symbol} ...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="max")
            break
        except:
            print(f"There is error when getting the stock of {symbol} !")
            n += 1
    
    return df.reset_index()


# 取得所有股票資料:
def GetAllStocks(symbols):
    stocks = {}
    minDate, maxDate = None, None  # 確保每個股票歷史資料的時間都在相同範圍。
    for sym in symbols:
        stock = GetStock(sym)
        stocks[sym] = FillNan(stock)
        if not minDate or stock["Date"].iloc[0] > minDate:
            minDate = stock["Date"].iloc[0]
        
        if not maxDate or stock["Date"].iloc[-1] < maxDate:
            maxDate = stock["Date"].iloc[-1]
    
    dateRange = pd.date_range(minDate, maxDate).to_frame(False, "Date")
    for sym in stocks:
        stocks[sym] = pd.merge(stocks[sym], dateRange, on="Date")
    
    return stocks # {"企業代號": 股票歷史DataFrame}


# 檢查每個企業的股票資料是否還有沒處理到的缺失值:
def CheckNan(stocks):
    print("")
    print('-' * 50)
    print('Check nan-value:')
    print('-' * 50)
    for sym in stocks:
        hasNan = stocks[sym].isnull().values.any()
        print(f"{sym}: {hasNan}")


# 儲存所有股票資料至一pickle檔案:
def SaveData(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


# 取得並儲存所有資料:
def GetData(symbols=SYMBOLS + [INDEX], isSave=True, path=SAVE_PATH):
    stocks = GetAllStocks(symbols)
    CheckNan(stocks)
    if isSave:
        SaveData(stocks, path)
        
    return stocks


# 讀取儲存的pickle檔:
def ReadData(path=SAVE_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)
    

# 視覺化股票走勢:
def VisualizeStock(df, columns=[]):
    df = df[columns] if columns else df
    df = df.drop("Date"        , axis=1) if "Date"         in df.columns else df
    df = df.drop("Volume"      , axis=1) if "Volume"       in df.columns else df
    df = df.drop("Dividends"   , axis=1) if "Dividends"    in df.columns else df
    df = df.drop("Stock Splits", axis=1) if "Stock Splits" in df.columns else df
    df.plot()
    

if __name__ == '__main__':
    stocks = GetData()
    
    
    
    
    
    
    
    
    