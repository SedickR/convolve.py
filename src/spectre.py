import plotly.graph_objects as go
import numpy as np
from tkinter.filedialog import askdirectory
import pathlib
from scipy.signal import find_peaks
import time

directory = pathlib.Path(askdirectory())

def unstack(a, axis=1):
    return np.moveaxis(a, axis, 0)


# Loop on all spectrums
for file in directory.iterdir():
    #get only the name of the file
    name = file.name
    data = np.genfromtxt(file, skip_header = 17, delimiter = '\t', skip_footer = 1)
    data = unstack(data)
    #apply a smoothing on intensity data (data[1])
    data_s = np.convolve(data[1], np.ones((20,))/20, mode='same')

    print(name)

    indices = find_peaks(data_s, prominence=1000)[0]
    
    print(indices)

    
    # graph the data
    fig = go.Figure(data=[go.Scatter(x=data[0], y=data[1], line = dict(width=1))])
    # add vertical lines at the peaks
    for i in indices:
        #add the line
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=data[0][i],
                y0=0,
                x1=data[0][i],
                y1=data[1][i],
                line=dict(color="grey", width=1)
            ))
        #add the value of the line above the
        fig.add_annotation(
            go.layout.Annotation(
                x=data[0][i],
                y=data[1][i]+2000,
                text=str(data[0][i]),
                xref="x",
                yref="y",
                showarrow=False
            )
        )
    
    fig.update_xaxes(title_text='Wavelength (nm)')
    fig.update_yaxes(title_text='Intensity (a.u.)')
    fig.update_layout(title=f'{name[:-4]}', template="simple_white")
    fig.write_image(f'spectres\\{name[:-4]}.pdf')
    time.sleep(0.5)
    fig.write_image(f'spectres\\{name[:-4]}.pdf')
