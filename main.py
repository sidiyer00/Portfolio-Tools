from portfolio import Portfolio

test_folio = {"TSLA": .3, "AAPL": .4, "AMZN": .3}
investment = 100000

folio = Portfolio(test_folio, investment, "2012-01-01", "2021-01-01")
folio.monte_carlo_sim("net_value", 500)
