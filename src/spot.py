from importlib.resources import path
from traitement import Simulation
from rms_spot import image_analysis
from argparse import ArgumentParser
import pathlib
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from tkinter.filedialog import asksaveasfilename
from tkinter.messagebox import askyesno
import tkinter as Tk
import cv2
import numpy as np


def command_line_parser():
    """Command line interface"""
    parser = ArgumentParser(description="Analyse spot size")
    parser.add_argument(
        "-f", "--folder", help="Load a folder instead of a file", action="store_true"
    )
    parser.add_argument(
        "-a", "--analyse", help="Analyse rms spot size from image", action="store_true"
    )
    parser.add_argument("-p", "--plot", help="Plot spot size", action="store_true")
    parser.add_argument(
        "-s", "--save", help="Save data to csv format", action="store_true"
    )
    parser.add_argument(
        "-v", "--verbose", help="Show steps in details", action="store_true"
    )
    parser.add_argument(
        "-ar",
        "--array",
        help="Compute full array of elemental image (diagonal, horizontal, vertical)",
        action="store_true",
    )
    parser.add_argument("-rgb", help="Select a color channel", action="store_true")

    parser.add_argument(
        "-l",
        "--line_plot",
        help="Compute and graph the plot size of series of images",
        action="store_true",
    )
    parser.add_argument(
        "-gen",
        "--generate",
        help="Generate an image from a zemax text file",
        action="store_true",
    )
    parser.add_argument(
        "-lc",
        "--lateral_color",
        help="Compute the lateral color of a series of images contained in a folder",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":

    terminal = command_line_parser()

    if terminal.lateral_color:
        terminal.folder = True

    if terminal.folder:  # Ask for folder if selected
        root = Tk.Tk()
        root.withdraw()
        filepath: pathlib.Path = pathlib.Path(askdirectory(title="Select a directory"))

    else:  # Ask for file otherwise
        root = Tk.Tk()
        root.withdraw()
        filepath: pathlib.Path = pathlib.Path(
            askopenfilename(
                title="Open text file",
                filetypes=[
                    ("all files", ".*"),
                    ("TXT", ".txt"),
                    ("CSV", ".csv"),
                    ("BMP", ".bmp"),
                ],
            )
        )

    if terminal.verbose:
        print("Command line arguments:")
        print(terminal)

    if terminal.save:
        save = True
    else:
        save = False

    if terminal.rgb:
        # ask user for rgb channel specification
        rgb = input("Which color channel do you want to analyse? (r, g, b)")
    else:
        rgb = "r"

    if terminal.analyse:
        analyse = image_analysis(filepath)  # Initialise image_analysis class
        results = analyse.compute_rms(channel=rgb, show=terminal.verbose)
        # ask to save if not already done
        if save is False:
            if askyesno("Save results", "Do you want to save the results?"):
                save = True

    if terminal.generate:
        """Generate an image from a zemax text file and compute the spot size"""
        sim = Simulation()
        results = []
        if terminal.folder is False:
            img = sim.generate_image(
                filepath, rgb
            )  # Generate image from zemax text file
            analyse = image_analysis(img=img)  # Initialise image_analysis class
            results = analyse.compute_rms(
                channel=rgb, show=terminal.verbose
            )  # Compute rms
        else:
            for file in filepath.iterdir():
                if file.suffix == ".txt":
                    img = sim.generate_image(
                        file, rgb
                    )  # Generate image from zemax text file
                    analyse = image_analysis(img=img)  # Initialise image_analysis class
                    results.append(
                        (None, analyse.compute_rms(rgb, terminal.verbose))
                    )  # Compute rms

        # ask to save if not already done
        if askyesno("Save results", "Do you want to save the results?"):
            save = True

    if terminal.array:
        """Compute full array of elemental image (diagonal, horizontal, vertical)"""
        analyse = image_analysis(filepath)
        results = analyse.complete_ei(channel=rgb, show=terminal.verbose)
        # ask to save if not already done
        if save is False:
            if askyesno("Save results", "Do you want to save the results?"):
                save = True

    if terminal.lateral_color:
        results = []
        for dir in filepath.iterdir():  # Find red and blue channel folders
            if "red" in dir.name:
                red_l = [i for i in dir.iterdir()]
            elif "blue" in dir.name:
                blue_l = [i for i in dir.iterdir()]

        for index, file in enumerate(
            red_l
        ):  # Compute lateral color for each image pair
            if file.suffix == ".txt":
                sim = Simulation()
                red = sim.generate_image(file, "r")
                blue = sim.generate_image(blue_l[index], "b")
                analyse = image_analysis()
                results.append(
                    (
                        f"{file.name} + {blue_l[index].name}",
                        analyse.compute_lateral_color(red, blue, terminal.verbose),
                    )
                )
            elif file.suffix == ".bmp":
                analyse = image_analysis()
                red = cv2.imdecode(
                    np.fromfile(file, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                )
                blue = cv2.imdecode(
                    np.fromfile(blue_l[index], dtype=np.uint8), cv2.IMREAD_UNCHANGED
                )
                results.append(
                    (
                        f"{file.name} + {blue_l[index].name}",
                        analyse.compute_lateral_color(red, blue, terminal.verbose),
                    )
                )
        print(results)
        if save is False:
            if askyesno("Save results", "Do you want to save the results?"):
                save = True

    if save:
        # ask for file name
        filename = asksaveasfilename(
            title="Save results", filetypes=[("CSV", ".csv"), ("all files", ".*")]
        )
        extension = (
            lambda x: ".csv" if x[-4:] != ".csv" else ""
        )  # Add extension if not already done
        analyse.save_to_csv(results, filename + extension(filename))

    if terminal.plot:
        if askyesno("Save results", "Do you want to save the results?"):
            save = True
        plotting = Simulation()
        plotting.load_data(filepath)
        plotting.plot_diagonal(rgb=rgb, save=save)
        # plotting.plot_matrix(rgb=rgb, save=save)

    if terminal.line_plot:
        if askyesno("Save results", "Do you want to save the results?"):
            save = True
        plotting = Simulation()
        plotting.plot_line(filepath, rgb=rgb, save=save)
