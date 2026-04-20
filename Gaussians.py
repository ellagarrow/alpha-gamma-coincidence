#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size' : 10, "font.family" : "Times New Roman", "text.usetex" : True})

def fwhm_to_sigma(fwhm):
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def norm_gauss_peak1(x, sigma):
    return np.exp(-0.5 * (x / sigma)**2)

# Replace these with your actual fit results
results = {
    "FWHM (Peak)": {"fwhm": 5.421, "fwhm_err": 0.084},
    "FWHM (LE)":    {"fwhm": 6.370, "fwhm_err": 0.097},
    "FWHM (CF)":    {"fwhm": 4.424, "fwhm_err": 0.062},
}

x = np.linspace(-20, 20, 2000)

plt.figure(figsize=(9, 6))
colors = ["black", "r", "b"]

for (name, r), color in zip(results.items(), colors):

    fwhm = r["fwhm"]
    fwhm_err = r["fwhm_err"]

    sigma = fwhm_to_sigma(fwhm)
    sigma_lo = fwhm_to_sigma(max(fwhm - fwhm_err, 1e-6))
    sigma_hi = fwhm_to_sigma(fwhm + fwhm_err)

    y = norm_gauss_peak1(x, sigma)
    y_lo = norm_gauss_peak1(x, sigma_hi)   # broader curve
    y_hi = norm_gauss_peak1(x, sigma_lo)   # narrower curve

    line, = plt.plot(x, y, linewidth=2,
               label=f"{name}: {fwhm:.2f} ± {fwhm_err:.2f} ns", color = color)
    #plt.fill_between(x, y_lo, y_hi, alpha=0.9)

#plt.title("Timing Resolution Comparison", fontsize=22)
plt.xlabel(r"$\Delta t - \mu$ (ns)", fontsize=20)
plt.ylabel("Relative Amplitude", fontsize=20)
plt.xlim(-20, 20)
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('FWHM_Comparison.png')
plt.show()


# In[ ]:




