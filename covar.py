import pandas as pd
import plotly.express as px


df = pd.read_excel("C:/Users/sidiy/Documents/Projects/Quant/Portfolio Tools/subset1.xlsx")
print(df)

fig = px.scatter(df)
fig.show()
