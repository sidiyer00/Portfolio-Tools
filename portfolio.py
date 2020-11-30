import pandas as pd
import numpy as np
import math
import pandas_datareader.data as pdr
import datetime as dt
import plotly.express as px


class Portfolio:
    """
    Constant weighted portfolio w/rebalancing and fractional shares
    """

    folio = []                  # list of stocks
    weights = []                # list of weights
    stocks = pd.DataFrame()     # pd.DataFrame of historical prices and net_value
    net_value = 0               # current portfolio value
    start_dt = 0
    end_dt = 0

    def __init__(self, stocks: dict, initial_investment: int, start: str, end: str):
        # Start and End dates in datetime format
        self.start_dt = dt.datetime.strptime(start, "%Y-%m-%d")
        self.end_dt = dt.datetime.strptime(end, "%Y-%m-%d")

        # list of stocks and weights in folio
        self.folio = list(stocks.keys())
        self.weights = list(stocks.values())

        # gets historical data and loads into DataFrame self.stocks
        for stock in stocks:
            self.stocks[stock] = pdr.DataReader(stock, "yahoo", self.start_dt, self.end_dt)["Adj Close"]

        # initialize net_value w/initial investment amount
        self.net_value = initial_investment
        # initial net_value numpy array w/initial investment
        net_value_np = np.array([self.net_value])

        # get dataframe of p2/p1 - drop first row of NaNs
        stocks_change_df = self.stocks/self.stocks.shift(1)
        stocks_change_df = stocks_change_df.drop(stocks_change_df.index[0])
        # calculate net value of portfolio over time & update self.stocks
        for row in stocks_change_df.itertuples(index=False):
            self.net_value = self.net_value * np.dot(np.asarray(row), np.asarray(self.weights))
            net_value_np = np.append(net_value_np, self.net_value)

        self.stocks["net_value"] = net_value_np.tolist()

    def efficient_frontier(self, trials: int):
        # number of stocks in the portfolio
        num_stocks = len(self.folio)
        # stock prices historical df
        portfolio = self.stocks.iloc[:, 0:num_stocks]
        # log returns of portfolio
        log_rets = np.log(portfolio / portfolio.shift(1))

        # all test weights for simulation
        all_weights = np.zeros((trials, num_stocks))

        ret_arr = np.zeros(trials)
        vol_arr = np.zeros(trials)
        sharpe_arr = np.zeros(trials)

        for x in range(trials):
            # weights
            weights = np.array(np.random.random(num_stocks))
            weights = weights / np.sum(weights)

            # save weights
            all_weights[x, :] = weights
            # Expected return
            ret_arr[x] = np.sum((log_rets.mean() * weights * 252))       # 252 days in a trading year
            # Expected volatility
            vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(log_rets.cov() * 252, weights)))
            # Sharpe Ratio
            sharpe_arr[x] = ret_arr[x] / vol_arr[x]

        max_sr_ret = ret_arr[sharpe_arr.argmax()]
        max_sr_vol = vol_arr[sharpe_arr.argmax()]

        fig = px.scatter(x=vol_arr, y=ret_arr, title= "Markowitz Efficient-Frontier", labels={"x":"Volatility (Risk)", "y":"Return"})
        fig.show()

        best_port_index = sharpe_arr.argmax()
        return all_weights[best_port_index, :]

    def monte_carlo_sim(self, stock: str, trials: int):
        prices = self.stocks[stock]
        returns = prices.pct_change()[1:]

        last_price = prices[-1]
        num_days = 252
        simulation_df = pd.DataFrame()

        for i in range(trials):
            count = 0
            daily_vol = returns.std()

            price_series = []
            price = last_price * (1 + np.random.normal(0, daily_vol))
            price_series.append(price)

            for j in range(num_days):
                price = price_series[count] * (1 + np.random.normal(0, daily_vol))
                price_series.append(price)
                count += 1

            simulation_df[i] = price_series

        head = "Monte Carlo Simulation: " + stock
        fig = px.line(simulation_df, title=head, labels={"index": "Days into Future", "value": "Value ($)"})
        fig.layout.update(showlegend=False)
        fig.show()

        return simulation_df

    def getPortfolioBeta(self):
        gspc = pd.DataFrame()
        gspc["gspc"] = pdr.DataReader("^GSPC", "yahoo", self.start_dt, self.end_dt)['Adj Close']
        gspc_rets = gspc.pct_change()[1:]
        port_rets = self.getReturns()
        beta_dict = {}

        for label, col in port_rets.items():
            col_var = col.var()
            cat = pd.concat([col, gspc_rets], axis=1)
            beta_dict[label] = cat.cov().iloc[1,1]/col_var

        net_beta = 0
        i = 0
        for key in beta_dict.keys():
            if i == len(self.folio):
                break;
            net_beta += beta_dict.get(key) * self.weights[i]
            i+=1

        beta_dict["net_value"] = net_beta
        return beta_dict

    def portfolioRisk(self):
        corr_matrix = self.stocks.loc[:, self.stocks.columns != "net_value"].corr()
        weights_arr = np.asarray(self.weights)
        return np.dot(np.dot(weights_arr, corr_matrix), weights_arr)

    def graphFolio(self):
        fig1 = px.line(self.stocks/self.stocks.iloc[0], title="Portfolio Growth over Time", labels={"Date": "Date", "value": "Growth"})
        fig1.show()
        fig2 = px.line(self.stocks["net_value"], title="Portfolio Growth over Time (Initial Investment $100,000)", labels={"Date": "Date", "value": "Net Value ($)"})
        fig2.show()

    def getReturns(self):
        daily_rets = self.stocks.pct_change()[1:]
        return daily_rets

    def graphReturns(self):
        daily_rets = self.getReturns()['net_value']
        fig1 = px.line(daily_rets, title = "Daily Returns of Net Value")
        fig1.show()

    def getWeights(self):
        return weights

    def getPricesFrame(self):
        return stocks

    def getFolio(self):
        return self.folio

    def __str__(self):
        return str(self.folio)



