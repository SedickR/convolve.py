from motion_class import CustomMotion
import yaml
import numpy as np
import csv
import plotly.graph_objects as go

with open(r'motion\scales.yaml', 'r') as file:
    scales = yaml.safe_load(file)


key = []
value = []

for keyx, valuex in scales.items():
    key.append(float(keyx))
    value.append(valuex)

#order key and value
key.sort()
value.sort(reverse=True)


#plot the keys as x and values as y in a scatter plot
fig = go.Figure(data=[go.Scatter(x=key, y=value)])
fig.show()

