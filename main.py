from interface import analyser_la_ligne_de_commande
from traitement import Simulation
from rms_spot import image_analysis

if __name__ == "__main__":

    terminal = analyser_la_ligne_de_commande()

    if terminal.verbose:
        print("Command line arguments:")
        print(terminal)
        verbose = True
    else:
        verbose = False
    
    if terminal.save:
        save = True
    else:
        save = False
    
    if terminal.rgb:
        # ask user for rgb channel specification
        rgb = input("Which color channel do you want to analyse? (r, g, b or all) ")
    else:
        rgb = "r"
    
    if terminal.analyse:
        analyse = image_analysis(terminal.load[0])
        results = analyse.compute_rms(channel=rgb, show=verbose)
        # ask to save if not already done
        if save is False:
            save = input("Do you want to save the results? (y/n) ")
            if save == "y":
                save = True
            else:
                save = False
    
    if terminal.save:
        # ask for file name
        filename = input("Enter a filename: ")
        analyse.save_data(results, filename)
