import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import tkinter as Tk
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror

# Open text file
root = Tk.Tk()
root.withdraw()
filepath = askopenfilename(title="Open text file", filetypes=[('TXT', '.txt'), ('all files', '.*')])
data = np.genfromtxt(filepath, skip_header=17)
#print(data)
# axis = 0 for x and axis = 1 for y 
profil = np.sum(data, axis = 0)
profil = profil-np.min(profil)
profil = profil/np.max(profil)
#print(profil)

nx = data.shape[1]
x = np.linspace(0, nx, num=nx) - nx/2

# Fit the data using a Gaussian fit
g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
fitter = fitting.LevMarLSQFitter()
g = fitter(g_init, x, profil)
fwhm_g = g.stddev*2.3548
pix_size_sensor_µm = 2.2                #pixel size of the sensor used in lab CMOS 1/2.5" 2.2 µm pixel pitch
fwhm_µm = fwhm_g * pix_size_sensor_µm   #conversion in µm

# Print results
print(g, '\n')
print('FWHM using a Gaussian fit = ', fwhm_g, 'pixels', '\n')
print('FWHM in µm = ', fwhm_µm, 'µm')

# Plot the data with the best-fit model
plt.figure(figsize=(8, 5))
plt.plot(x, profil, 'ko')
plt.plot(x, g(x), 'g')
plt.xlabel('X (pixels)', fontsize=14)
plt.ylabel('Intensity (A.U.)', fontsize=14)
plt.grid(b=True, which='both', color='0.65', linestyle=':')
plt.show()
