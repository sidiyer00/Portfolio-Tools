import pandas_datareader.data as pdr
import pandas as pd
import datetime as dt
import numpy as np
import plotly.express as px


def monte_carlo_sim(stock: str, start: str, end: str, trials: int):
    """
    Based on past returns and volatility, conducts a Monte Carlo Simulation of future price trends (1y).
    Plotly graphical output

    :param stock: str
    :param start: str
    :param end: str
    :param trials: int
    :return:
    """
    start_dt = dt.datetime.strptime(start, "%Y-%m-%d")
    end_dt = dt.datetime.strptime(end, "%Y-%m-%d")

    prices = pdr.DataReader(stock, "yahoo", start_dt, end_dt)['Adj Close']
    returns = prices.pct_change()

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
    fig = px.line(simulation_df, title=head)
    fig.layout.update(showlegend=False)
    fig.show()

    return simulation_df
