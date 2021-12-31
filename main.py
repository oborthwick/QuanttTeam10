from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.stattools import adfuller
import numpy as np

class Team10Algo(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2014,8,1)
        self.SetEndDate(2019,8,1)
        self.SetCash(100000000)
        self.historical = dict()
        self.tickers = ["CHTR", "MCO", "ODFL", "MKC", "MKTX", "YUM", "JKHY", "HD", "MKTX", "ILMN", "EL",
            "CPRT", "NDAQ", "SHW"]
        # Schedule for adf scores (new stock to trade) once per week
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday), self.TimeRules.At(9, 0), Action(self.StartofWeek))
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(TimeSpan.FromMinutes(181)), Action(self.EveryThreeHours))
        
        # first order difference, then get adf scores
    def AdfScores(self):
        #find lowest adf score- this is stock selected to trade for week
        adfs = []
        for symbol, historical in self.historical.items():
            first_diff = self.historical[symbol]['close'].diff() #assume data will need to be differenced once
            first_diff = first_diff.iloc[1:] #remove NaNs
            adfs.append(adfuller(first_diff)[1])
        self.min_adf = adfs[0]
        for score in adfs:
            if score < self.min_adf:
                self.min_adf = score
        self.stock_to_trade = self.tickers[adfs.index(self.min_adf)] #get ticker associated with min adf
        
    def StartofWeek(self):
        for i in self.tickers:
            symbol = self.AddEquity(i, Resolution.Hour).Symbol
            self.historical[symbol] = self.History(symbol, 21, Resolution.Daily)
        self.AdfScores()
        self.Liquidate()
        
    def EveryThreeHours(self):
        self.InitializeCount()
        
    def InitializeCount(self):
        self.postrend = 0
        self.negtrend = 0
        
    #run arima model on stock to trade, use forecast vs current price to make trades
    def OnData(self, data):
        trading_stock = self.AddEquity(self.stock_to_trade, Resolution.Hour).Symbol
        price = self.Securities[self.stock_to_trade].Price
        p_val = ar_select_order(np.array(self.historical[trading_stock]['close']), maxlag=6)
        p_val.ar_lags
        model = ARIMA(np.array(self.historical[trading_stock]['close']), order=(p_val.ar_lags,1,0))
        result = model.fit()
        forecast = result.forecast()
        if (self.postrend <= 2) and (forecast > price*1.1):
            self.SetHoldings(self.stock_to_trade, (forecast/price)*1.1)
            self.postrend += 1
        elif (self.postrend <= 2) and (forecast < price*1.1) and (forecast > price):
            self.SetHoldings(self.stock_to_trade, (forecast/price)*1.05)
            self.postrend += 1
        elif (self.postrend <= 2) and (forecast < price) and (forecast > price*0.975):
            self.SetHoldings(self.stock_to_trade, -(forecast/price)*0.05)
            self.negtrend += 1
        elif (self.postrend <= 2) and (forecast < price*0.975):
            self.SetHoldings(self.stock_to_trade, -(forecast/price)*0.2)
            self.negtrend += 1
        #use momentum to influence decision when forecast/price~1
        elif (self.postrend > 2) and (forecast > price):
            self.SetHoldings(self.stock_to_trade, 1.1)
        elif (self.negtrend > 2) and (forecast < price):
            self.SetHoldings(self.stock_to_trade, -0.1)
        else:
            self.Liquidate()
