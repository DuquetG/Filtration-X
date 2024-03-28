import re
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sigfig import round

## Traitement des données brutes sous format .mca ##
## Prend en argument le fichier .mca et retourne une liste de valeurs filtrées ##
## Détecte automatiquement le nombre de canaux ##

def traitement(fichier):

    with open(fichier, 'r') as file:
        data = file.read()

    valeurs = re.search(r'<<DATA>>(.*?)<<END>>', data, re.DOTALL)
    liste_valeurs = list(map(float, valeurs.group(1).strip().split("\n")))

    match_canaux = re.search(r'MCA Channels: (\d+)', data)
    if match_canaux:
        canaux = int(match_canaux.group(1))
    else:
        canaux = None

    liste_canaux = range(canaux)

    return [liste_canaux, liste_valeurs]


## Fonction d'initialisation de courbe gaussienne ##

def gaussienne(x, a, sigma, mu):
    return a * np.exp(-((x - mu) / sigma) ** 2)


## Courbe d'ajustement ##

def gauss_fit(x_data, y_data, pos):
    popt, pcov = curve_fit(gaussienne, x_data, y_data, p0=[np.max(y_data), np.sqrt(np.std(y_data)), pos])

    fit = gaussienne(x_data, *popt)
    ## Calcul du R^2 du fit ##
    residuals = fit - y_data
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    plt.plot(x_data, fit, label=f"Courbe d'ajustement du pic d'énergie", alpha=0.8)

    return popt


## Mention des valeurs approximatives des pics d'énergie du générateur en canaux ##

pics = [796.05, 1011.89, 3336.32]


## Valeurs de référence de l'américium en keV ##

ref = [13.95, 17.74, 59.54]


## Valeurs brutes des pics du générateur ##

ref_brut = [10]


## Ajustement d'une relation linéaire ax + b ##

a, b = np.polyfit(pics, ref, 1)


## Initialisation des listes des valeurs d'énergie en keV ##

kev = []


## Transformation de canaux à énergie des valeurs pour chaque spectre ##

energie_list = []

def etalonnage(a, b, x):
    for ii in x:
        energie = ii * a + b
        energie_list.append(energie)
        kev.append(energie)

etalonnage(a, b, range(0, 4096))


## Calcul de l'incertitude d'un canal en énergie ##

inc_max = abs(a+b)


## Listes de marqueurs et de couleurs ##

marks = [".", "^", "P", "*", "d", "h"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


## Listes des paramètres ##

name = ["Énergie moyenne", "Énergie maximale", "Nombre de comptes"]
tension = [10, 13, 15, 17, 20]
courant = [15, 40, 50, 60, 70]
z = [42, 47, 29, 74, 13]
t = [10, 20, 40, 80]


## Affichage des valeurs pertinentes sous forme de graphique normalisé ##

def show_values(values, folder):
    plt.clf()

    num = 0
    for i in values:
        val = list(map(lambda x : x/max(i[0]), i[0]))
        inc = list(map(lambda x : x/max(i[0]), i[1]))

        if "Tension" in folder:
            plt.scatter(tension, val, label=f"{name[num]}", marker=marks[num], s=100)
            plt.errorbar(tension, val, inc, capsize=5)
            plt.xlabel('Tension [keV]')

        elif "Courant" in folder:
            plt.scatter(courant, val, label=f"{name[num]}", marker=marks[num], s=100)
            plt.errorbar(courant, val, inc, capsize=5)
            plt.xlabel('Courant [μA]')

            ## Calcul des paramètres la relation linéaire ##
            if num == 2:
                aa, bb = np.polyfit(courant, i[0], 1)
                fit = []
                for ii in range(5):
                    fit.append(aa*courant[ii]+bb)
                r_squared = r2_score(i[0], fit)
                print(f"La relation est : y = {aa}*x + {bb} avec R^2={r_squared}")

        elif "Filtres" in folder:
            plt.scatter(z, val, label=f"{name[num]}", marker=marks[num], s=100)
            plt.errorbar(z, val, inc, capsize=5, linestyle='')
            plt.xlabel('Numéro atomique Z [-]')

        elif "Épaisseur" in folder:
            plt.scatter(t, val, label=f"{name[num]}", marker=marks[num], s=100)
            plt.errorbar(t, val, inc, capsize=5)
            plt.xlabel('Épaisseur de filtre [mil]')

        else:
            pass

        num += 1

    plt.ylabel("Échelle normalisée d'énergie ou de nombre de comptes [-]")
    plt.tick_params("both", direction="in")
    plt.legend()
    plt.show()


## Affichage des spectres bruts après étalonnage ##

def spectres(spectres_folder):
    ## Obtention de la liste des fichiers .mca dans le dossier spectres_folder ##
    spectres = [file_name[:-4] for file_name in os.listdir(spectres_folder) if file_name.endswith(".mca")]

    ## Initialisation des valeurs ##
    num = 0 
    e_moy_list = []
    inc_moy_list = []
    e_max_list = []
    inc_max_list = []
    counts_list = []
    inc_counts_list = []


    for spectre in spectres:
        données = traitement(os.path.join(spectres_folder, f"{spectre}.mca"))
        x = données[0]
        y = données[1]

        plt.plot(energie_list, y, label=f"Spectre {spectre}", linewidth=2.0, zorder=-1)


        ## Affichage de la courbe normalisée et calcul de son incertitude ##

        inc_moy = 0
        for peak in ref_brut:
            popt = gauss_fit(energie_list, y, peak)
            inc_moy = abs(popt[1])


        ## Calcul des valeurs pertinentes pour le spectre sélectionné ##

        counts = 0
        e_tot = 0
        bine = 0
        for i in y:
            counts += i
            e_tot += i*x[bine]
            bine += 1
        e_moy = a*(e_tot/counts)+b
        non_nuls = [i for i, valeur in enumerate(y) if valeur > 3]
        e_max = a*non_nuls[-1]+b
        inc_count = math.sqrt(counts/len(y))


        ## Affichage de l'énergie moyenne ##

        # plt.axvline(x = e_moy, ymin = 0, label=f"Énergie moyenne à {spectre}", linestyle="--", color=colors[num])


        ## Affichage de l'énergie maximale ##

        # plt.scatter(e_max, [0], label=f"Énergie max {spectre}", zorder=1, s=100, marker=marks[num])
        num += 1


        ## Affichage et stockage des valeurs pertinentes ##

        # print("            "+
        #     f"""------ Spectre de {spectre} ------
        #     Énergie moyenne : {round(e_moy, 5)} ± {round(inc_moy, 2)}
        #     Énergie maximale : {round(e_max, 5)} ± {round(inc_max, 2)}
        #     Nombre de comptes : {counts} ± {round(inc_count, 2)}
        #     -------------------------------""")
        
        if "brut" in spectre:
            pass
        else:
            e_moy_list.append(round(e_moy, 5))
            inc_moy_list.append(round(inc_moy, 2))
            e_max_list.append(round(e_max, 5))
            inc_max_list.append(round(inc_max, 2))
            counts_list.append(counts)
            inc_counts_list.append(round(inc_count,2))

    show_values([[e_moy_list, inc_moy_list,],
                [e_max_list, inc_max_list,],
                [counts_list, inc_counts_list]],
                spectres_folder)

    plt.xlabel('Énergie [keV]')
    plt.ylabel('Nombre de comptes [-]')
    plt.tick_params("both", direction="in")
    plt.legend()
    plt.xlim(xmin=0, xmax=30)
    plt.show()


## Affichage des graphiques ##

spectres("Séance 1\\CdTe\\Tension")
spectres("Séance 1\\CdTe\\Courant")
spectres("Séance 1\\CdTe\\Filtres")
spectres("Séance 2\\Épaisseur")
spectres("Séance 2\\Combo à 25-15\\Al-Ag")
spectres("Séance 2\\Combo à 25-15\\Al-W")
