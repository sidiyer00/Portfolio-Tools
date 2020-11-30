require(TTR)
require(writexl)

stocks = TTR::stockSymbols()
stocks = stocks[c("Symbol", "Name", "Exchange")]

write_xlsx(stocks, "C:\\Users\\sidiy\\Documents\\Projects\\Quant\\Datasets\\all_public_stocks.xlsx")