# todohighlight.include
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import time, datetime, os, pathlib
from rms_spot import image_analysis

# TODO:
# - 0.75217, pas de rotation cas idéal
# - centre à centre deux pixel dans la matrice diagonale, 39.5 um
# - Tester code avec un truc déjà connue et facile à faire à la main - DONE
# - intégrer dans l'affichage
# - Clarifier les intrants et extrants de chaque fonction, formats des arguments, etc.


class Simulation:
    """Zemax optical simulation data analysis and plotting"""

    def __init__(self):
        self.df = None
        self.matrix_s = None
        self.matrix_r = None
        self.matrix_g = None
        self.matrix_b = None
        self.steps = {'r': 0.75217, 'g': 0.75217, 'b':0.75217}

    def load_data(self, filename):
        """Load data from a Zemax .txt file

        Loads data from a single text file and stores it in a pandas dataframe."""
        self.df = pd.read_csv(filename, sep=";")
        self.matrix_s = (
            np.resize(np.array(self.df.loc[:, "Spot Size"]), (12, 12)),
            "Combined",
        )
        self.matrix_r = (
            np.resize(
                np.array(self.df.loc[:, "R Spot Size"]),
                (
                    int(np.sqrt(len(self.df.loc[:, "R Spot Size"]))),
                    int(np.sqrt(len(self.df.loc[:, "R Spot Size"]))),
                ),
            ),
            "Red",
        )
        self.matrix_g = (
            np.resize(
                np.array(self.df.loc[:, "G Spot Size"]),
                (
                    int(np.sqrt(len(self.df.loc[:, "G Spot Size"]))),
                    int(np.sqrt(len(self.df.loc[:, "G Spot Size"]))),
                ),
            ),
            "Green",
        )
        self.matrix_b = (
            np.resize(
                np.array(self.df.loc[:, "B Spot Size"]),
                (
                    int(np.sqrt(len(self.df.loc[:, "B Spot Size"]))),
                    int(np.sqrt(len(self.df.loc[:, "B Spot Size"]))),
                ),
            ),
            "Blue",
        )

    def heatmap_maker(self, matrix):
        """Create figure from matrix

        Takes a matrix of spot size and returns a plotly Heatmap figure
        with degrees as x and y axis and microns as z axis"""
        steps = {'r': 0.75217*2, 'g': 0.75217, 'b':0.75217*2}
        fig = go.Figure(
            data=go.Heatmap(z=matrix[0], colorbar={"title": "Spot Size [μm]"})
        )
        fig.update_yaxes(title_text="Row index")
        fig.update_xaxes(title_text="Column index")
        fig.update_layout(title=f"Spot Size RSRH")
        return fig, matrix[1]

    def plot_matrix(self, rgb="r", save=False):
        """Plot the Spot Size matrix

        Takes a series of boolean arguments to specify which matrix to plot and
        whether to save the figure or not. The function does not return anything,
        but shows or saves the figure."""
        plots = []
        if "r" in rgb and "g" in rgb and "b" in rgb:
            plots.append(self.heatmap_maker(self.matrix_r))
            plots.append(self.heatmap_maker(self.matrix_g))
            plots.append(self.heatmap_maker(self.matrix_b))
        if rgb == "r":
            plots.append(self.heatmap_maker(self.matrix_r))
        if rgb == "g":
            plots.append(self.heatmap_maker(self.matrix_g))
        if rgb == "b":
            plots.append(self.heatmap_maker(self.matrix_b))
        if save:
            for fig, name in plots:
                fig.write_image(
                    f'matrix_heatmap\\{name}_heatmap_{datetime.datetime.now().isoformat("_", "minutes").replace(":", "")}.pdf'
                )
                time.sleep(0.5)
                fig.write_image(
                    f'matrix_heatmap\\{name}_heatmap_{datetime.datetime.now().isoformat("_", "minutes").replace(":", "")}.pdf'
                )
        else:
            for fig, name in plots:
                fig.show()

    def plot_diagonal(self, rgb="r", save=False):
        """Plot the gradient of the Spot Size matrix

        Plots the gradient of the three axis of interest of spot size of the loaded data.
        Saves the figure if save is True."""
        
        if "r" in rgb and "g" in rgb and "b" in rgb:
            self.plot_diagonal(rgb="r", save=save)
            self.plot_diagonal(rgb="g", save=save)
            self.plot_diagonal(rgb="b", save=save)
        elif rgb == "r":
            matrix = self.matrix_r[0]
        elif rgb == "g":
            matrix = self.matrix_g[0]
        elif rgb == "b":
            matrix = self.matrix_b[0]
        else:
            raise ValueError("rgb must be 'r', 'g' or 'b' or a combination of those")

        # compute gradient of diagonal and plot
        #diag = matrix.diagonal()
        #grad = np.gradient(diag)
        #fig = go.Figure(
        #    data=go.Scatter(
        #        x=np.arange(0, len(grad)*steps[rgb], step=steps[rgb]),
        #        y=grad,
        #        mode="lines",
        #        name="diagonal",
        #        line=dict(shape="linear", color="black"),
        #    )
        #)
        # Compute gradient of vertical line and plot
        grad_v = np.gradient(matrix[:, 0])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(0, len(grad_v)*self.steps[rgb], step=self.steps[rgb]),
                y=grad_v,
                mode="lines",
                name="vertical",
                line=dict(shape="linear", color="black", dash="dash"),
            )
        )
        #compute gradient of horizontal line and plot
        grad_h = np.gradient(matrix[0, :])
        fig.add_trace(
            go.Scatter(
                x=np.arange(0, len(grad_h)*self.steps[rgb], step=self.steps[rgb]),
                y=grad_h,
                mode="lines",
                name="horizontal",
                line=dict(shape="linear", color="black", dash="dot"),
            )
        )
        #add plot of horizontal line
        fig.add_trace(
            go.Scatter(
                x=np.arange(0, len(matrix[:, 0])+3, step=self.steps[rgb]),
                y=matrix[:, 0],
                mode="lines",
                name="Horizontal",
            )
        )
        #add plot of vertical line
        fig.add_trace(
            go.Scatter(
                x=np.arange(0, (len(matrix[0, :])+2)*self.steps[rgb], step=self.steps[rgb]),
                y=matrix[0, :],
                mode="lines",
                name="Vertical",
            )
        )

        #Compute a fourth degree polynomial fit to the vertical and horizontal lines
        fit_v = np.polyfit(np.arange(0, len(grad_v)*self.steps[rgb], step=self.steps[rgb]), grad_v, 4)
        fit_h = np.polyfit(np.arange(0, len(grad_h)*self.steps[rgb], step=self.steps[rgb]), grad_h, 4)

        # Compute the polynomial values of the fit
        poly_v = np.polyval(fit_v, np.arange(0, len(grad_v)*self.steps[rgb], step=self.steps[rgb]))
        poly_h = np.polyval(fit_h, np.arange(0, len(grad_h)*self.steps[rgb], step=self.steps[rgb]))

        # Add the polyline to the plot
        fig.add_trace(
            go.Scatter(
                x=np.arange(0, len(poly_v)*self.steps[rgb], step=self.steps[rgb]),
                y=poly_v,
                mode="lines",
                name="Vertical fit",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(0, len(poly_h)*self.steps[rgb], step=self.steps[rgb]),
                y=poly_h,
                mode="lines",
                name="Horizontal fit",
            )
        )

        # Print the fit equation
        print(f"Vertical fit: {fit_v}")
        print(f"Horizontal fit: {fit_h}")

        # Add axis labels
        fig.update_yaxes(title_text="Spot Size grandient")
        fig.update_xaxes(title_text="Rotation step", dtick=self.steps[rgb])
        fig.update_layout(title=f"Spot Size Gradient RSRH")
        fig.update_layout(template="simple_white", xaxis_tickformat = '°')

        # Save figure if specified
        if save:
            fig.write_image(
                f'diagonal_gradient\\gradient_{datetime.datetime.now().isoformat("_", "minutes").replace(":", "")}.pdf'
            )
            time.sleep(1)
            fig.write_image(
                f'diagonal_gradient\\gradient_{datetime.datetime.now().isoformat("_", "minutes").replace(":", "")}.pdf'
            )
        else:
            fig.show()

    def plot_all_files(
        self, directory, matrix=True, diagonal=True, rgb="r", save=False
    ):
        """Plot all text file in the directory
        
        Takes a directory and plots all the text files or csv files contained.
        Transfers the rgb argument to the class functions called"""
        directory = pathlib.Path(directory)
        for file in directory.iterdir():
            if file.suffix == ".txt" or file.suffix == ".csv":
                self.load_data(directory + "\\" + file)
                if matrix:
                    self.plot_matrix(rgb=rgb, save=save)
                if diagonal:
                    self.plot_diagonal(rgb=rgb, save=save)

    def plot_line(self, directory, rgb="r", save=False):
        """Takes a directory or a csv file and plots the horizontal or vertical lines."""

        file = pathlib.Path(directory)

        if file.suffix == ".txt" or file.suffix == ".csv":
            data = np.loadtxt(file, delimiter=";", skiprows=1)
            rows = {"b":1, "g":2, "r":3}
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=np.arange(0, len(data)*self.steps[rgb], step=self.steps[rgb]),
                    y=data[:, rows[rgb]],
                    mode="lines",
                    name='spot size',
                    line=dict(shape="linear", color="black", dash="dash"),
                )
            )
            grad = np.gradient(data[:, rows[rgb]])
            fig.add_trace(
                go.Scatter(
                    x=np.arange(0, len(grad)*self.steps[rgb], step=self.steps[rgb]),
                    y=grad,
                    mode="lines",
                    name="gradient",
                    line=dict(shape="linear", color="black"),
                )
            )
        if file.is_dir():
            directory = image_analysis(file)
            data = directory.compute_rms()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=np.arange(0, len(data)*self.steps[rgb], step=self.steps[rgb]),
                    y=data[:, 1],
                    mode="lines",
                    line=dict(shape="linear", color="black", dash="dash"),
                )
            )
            grad = np.gradient(data[:, 1])
            fig.add_trace(
                go.Scatter(
                    x=np.arange(0, len(grad)*self.steps[rgb], step=self.steps[rgb]),
                    y=grad,
                    mode="lines",
                    line=dict(shape="linear", color="black"),
                )
            )
        fig.update_xaxes(title_text="Rotation step", dtick=self.steps[rgb])
        if save:
            fig.write_image(
                f'line\\line_{datetime.datetime.now().isoformat("_", "minutes").replace(":", "")}.pdf'
            )
            time.sleep(1)
            fig.write_image(
                f'line\\line_{datetime.datetime.now().isoformat("_", "minutes").replace(":", "")}.pdf'
            )
        else:
            fig.show()


