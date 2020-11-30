# Portfolio-Tools
These are a few portfolio analysis tools for personal use. Currently, only constant weight rebalancing and fractional shares are supported.

Included:
1) Risk Calculations, Beta calculations 
2) Charting tools 
3) Markowitz Efficient Frontier charting 
4) Monte Carlo simulations 

Historical pricing data is sourced from Python pandas_datareader.

# Usage
More tools such as Value at Risk and Bond tools will be added soon.
```
from portfolio import Portfolio

test_folio = {"TSLA": .3, "AAPL": .4, "AMZN": .3}
investment = 100000

folio = Portfolio(test_folio, investment, "2012-01-01", "2021-01-01")
```

```
# Monte Carlo w/ 500 trials charting - can run on any stocks ticker in the portfolio
# Example output (optimal weights): [0.23862722 0.27578674 0.48558605]
folio.monte_carlo_sim("net_value", 500)
```

```
# Efficinet Fronter charting - returns optimal weights based on trials
folio.efficient_frontier(500)
```

```
# Calculates beta via Cov(r_stock, r_market)/Var(r_market) - returns dictionary of betas
folio.getPortfolioBeta()
```

```
# Calculates weighted risk of portfolio via correlation matrix 
folio.portfolioRisk()
```

```
# Charts portfolio - call values scaled down to reflect percentage changes and uniform axes
folio.graphFolio()
```
# Charts

![Efficient Frontier Charting](https://github.com/sidiyer00/Portfolio-Tools/blob/main/pics/efficinet-frontier.PNG)
![Monte Carlo Charting](https://github.com/sidiyer00/Portfolio-Tools/blob/main/pics/monte-carlo.PNG)

# Dependencies
* pandas
* pandas_datareader
* numpy 
* datetime
* plotly
