import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


## Traitement des données brutes sous format .mca ##

z = [42, 74, 47, 13, 29]
tau = [1.26E-20, 3.38E-20, 3.20E-20, 9.49E-22, 1.83E-20]
inc = [0.63E-20, 1.7E-20, 1.6E-20, 5.1E-22, 0.93E-20]
z_list = []
tau_list = []
inc_list = []

for i in z:
    z_list.append(math.log(i))

for i in tau:
    tau_list.append(math.log(i/10.E-24))

for i in inc:
    inc_list.append(math.log(i/10.E-24))

a, b = np.polyfit(z_list, tau_list, 1, cov=True)

x = []
y = []
for i in np.linspace(2.5, 4.5, num=100):
    x.append(i)
    y.append(i*a[0]+a[1])

print(a,b)

plt.plot(x, y, label=f"Régression linéaire avec pente de {a[0]:.1f}" + r"$\pm$" + f"{b[0][0]:.1f}")
plt.scatter(z_list, tau_list, label="Valeurs associées aux filtres")
plt.errorbar(z_list, tau_list, inc_list, capsize=5, linestyle='')
plt.xlabel('ln(Z)')
plt.ylabel('ln(tau)')
plt.tick_params("both", direction="in")
plt.legend()
plt.show()
# plt.xlim(xmin=0, xmax=30)
# plt.savefig('Graph_Regression.png', dpi = 600)
