#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import modules
import numpy as np
import os, platform 
import string
import matplotlib.pyplot as plt
import requests
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
plt.rcParams.update({'font.size' : 10, "font.family" : "Times New Roman", "text.usetex" : True})


# In[2]:


print(os.listdir("/home/2660162g/Masters Project/BinFiles"))


# ## Loading in Alpha data

# ## Loading in gamma data

# In[3]:


RAW_DIR = "/home/2660162g/Masters Project/BinFiles"

GAMMA_RAW = os.path.join(RAW_DIR, "Gammalongfeb_u16.bin")
ALPHA_RAW = os.path.join(RAW_DIR, "Alphalongfeb_u16.bin")

record_length = 256
dtype = np.uint16


def load_waveforms_memmap(bin_path, record_length=256, dtype=np.uint16):
    
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"File not found: {bin_path}")

    print("Loading:", bin_path)

    data = np.memmap(bin_path, dtype=dtype, mode="r")

    n_waveforms = data.size // record_length
    n_use = n_waveforms * record_length

    pulses = data[:n_use].reshape(n_waveforms, record_length)

    return data, pulses, n_waveforms



gamma_data, gamma_u16, Ng = load_waveforms_memmap(GAMMA_RAW, record_length)
alpha_data, alpha_u16, Na = load_waveforms_memmap(ALPHA_RAW, record_length)

N = min(Ng, Na)  # safe if one file is slightly shorter
print("Gamma shape:", gamma_u16.shape)
print("Alpha shape:", alpha_u16.shape)
print("Using N waveforms:", N)


# In[4]:


# timing axis conversion
fs = 250e6 # sampling rate
dt = 1.0 / fs # timing interval
dt_ns = dt * 1e9 # nanosecond conversion
t_ns = np.arange(record_length) * dt * 1e9 # full axis nanosecond conversion

# ADC to Volts
Vpp = 2.0 # voltage range
ADC_levels = 4096  # 12-bit
V_per_count = Vpp / ADC_levels # voltage conversion


# In[5]:


# baseline correction
baseline_samples = 50

def baseline_correct_to_volts(wf_u16: np.ndarray,
                              baseline_samples: int = 50) -> np.ndarray:
   
    wf = wf_u16.astype(np.int32, copy=False)
    b = np.median(wf[:baseline_samples]).astype(np.float32)  # using only start 
    corr_counts = wf.astype(np.float32) - b # correcting counts
    return corr_counts * V_per_count  # volts


# In[6]:


# peak time function, parabolic interpolation of peak (tested at one point)
def peak_time_parabolic(wf: np.ndarray, pk: int, dt_ns: float):

    pk = int(pk) # identify peak
    if pk <= 0 or pk >= len(wf) - 1:
        return pk * dt_ns  

    # parabolic interpolation
    y_m1 = float(wf[pk - 1]) 
    y_0  = float(wf[pk])
    y_p1 = float(wf[pk + 1])

    denom = (y_m1 - 2.0*y_0 + y_p1)
    if denom == 0:
        return pk * dt_ns

    delta = 0.5 * (y_m1 - y_p1) / denom

    # peak should be within +/-1 sample
    if not np.isfinite(delta) or abs(delta) > 1.0:
        delta = 0.0

    return (pk + delta) * dt_ns


# In[7]:


# Thresholds
# fraction of maximum pulse height
thr_frac = 0.1 # 10%

# to avoid lots of noise
thr_floor = 0.004 #V

# negative pulses
POLARITY_ALPHA = -1 
POLARITY_GAMMA = -1

cut_V = -0.7 #V

min_dist = 10          # samples between peaks
prom_a_V = 0.01      
prom_g_V = 0.01

#min_dt_samp = 0
max_dt_samp = 75 # 300 ns coincidence window

# amplitude spectra things
bins = 300
a_range = (-2, 0)
g_range = (-2, 0)

a_hist = np.zeros(bins, dtype = np.int64)
g_hist = np.zeros(bins, dtype = np.int64)

# coincidence results using peak timing 
dt_peak_ns = []               # timing spectrum values
a_cond_V = []                 # alpha amps in coincidence sepctrum
g_cond_V = []                 # gamma amps in coincidence spectrum
coinc_examples = [] # store a few examples for plotting later 

a_all = []
g_all = []

EXAMPLE_KEEP = 30


processed = 0
skipped_a = 0    # seeing how many dont have coincidences
skipped_g = 0

for i in range(N):
    a_V = baseline_correct_to_volts(alpha_u16[i], baseline_samples)  # the baseline correction
    g_V = baseline_correct_to_volts(gamma_u16[i], baseline_samples)


    if POLARITY_ALPHA == -1:
        a_sig = -a_V  # positive-going for find_peaks

        # per-waveform dynamic threshold = 10% of max pulse height
        a_max = float(np.max(a_sig))                # same as -a_V.min()
        thr_a_this = max(thr_floor, thr_frac * a_max)

        if a_max < thr_floor:   # skips if pulse not at least floor 
            a_idx = np.array([], dtype=np.int32)
            skipped_a += 1
        else:
            a_idx, _ = find_peaks(a_sig, height=thr_a_this,
                              prominence=prom_a_V, distance=min_dist)


    if POLARITY_GAMMA == -1: # repeat for gamma
        g_sig = -g_V

        g_max = float(np.max(g_sig))
        thr_g_this = max(thr_floor, thr_frac * g_max)

        if g_max < thr_floor:
            g_idx = np.array([], dtype=np.int32)
            skipped_g += 1
        else:
            g_idx, _ = find_peaks(g_sig, height=thr_g_this,
                              prominence=prom_g_V, distance=min_dist)

    if a_idx.size:
        h, _ = np.histogram(a_V[a_idx], bins=bins, range=a_range)    # histograms all pulses 
        a_hist += h
        a_all.extend(a_V[a_idx])
        
    if g_idx.size:
        g_idx = g_idx[g_V[g_idx] >= cut_V]
        h, _ = np.histogram(g_V[g_idx], bins=bins, range=g_range)
        g_hist += h
        g_all.extend(g_V[a_idx])


    if a_idx.size and g_idx.size:
        a_amp = a_V[a_idx]
        g_amp = g_V[g_idx]

        # coincidence part:
        if a_idx.size and g_idx.size:
            for a_i, a_a in zip(a_idx, a_amp):
                lo = a_i - max_dt_samp
                hi = a_i + max_dt_samp
                cand = np.where((g_idx >= lo) & (g_idx <= hi))[0]
                if cand.size == 0:
                    continue

                # choosing gamma peak closest in time to alpha 
                j = cand[np.argmin(np.abs(g_idx[cand] - a_i))]
                g_i = int(g_idx[j])

                tA = peak_time_parabolic(a_V, int(a_i), dt_ns)
                tG = peak_time_parabolic(g_V, int(g_i), dt_ns)
                dt_ns_val = tG - tA
                #dt_ns_val = (g_i - int(a_i)) * dt_ns
                dt_peak_ns.append(dt_ns_val)
                a_cond_V.append(float(a_a))
                g_cond_V.append(float(g_amp[j]))

                if len(coinc_examples) < EXAMPLE_KEEP:
                    coinc_examples.append((i, int(a_i), int(g_i)))

    # progress checker
    processed += 1
    if processed % 100_000 == 0:
        print(f"{processed:,}/{N:,} | skipped_a={skipped_a:,} skipped_g={skipped_g:,} | dt entries={len(dt_peak_ns):,}")

dt_peak_ns = np.array(dt_peak_ns, dtype=np.float64)
a_cond_V = np.array(a_cond_V, dtype=np.float32)
g_cond_V = np.array(g_cond_V, dtype=np.float32)

print("Done.")
print("dt_peak_ns entries:", dt_peak_ns.size)
print("Stored examples:", len(coinc_examples))


# In[8]:


# amplitude spectra (all peaks) 
bins_a = a_hist.size
edges_a = np.linspace(a_range[0], a_range[1], bins_a + 1)
cent_a  = 0.5 * (edges_a[:-1] + edges_a[1:])
width_a = edges_a[1] - edges_a[0]
nz = np.where(a_hist > 0)[0]
idx0_a = nz[0] if nz.size else 0
x_start_a = edges_a[max(0, idx0_a - 5)]

bins_g = g_hist.size
edges_g = np.linspace(g_range[0], g_range[1], bins_g + 1)
cent_g  = 0.5 * (edges_g[:-1] + edges_g[1:])
width_g = edges_g[1] - edges_g[0]
nz = np.where(g_hist > 0)[0]
idx0_g = nz[0] if nz.size else 0
x_start_g = edges_g[max(0, idx0_g - 5)]

# all pulses spectra
# alpha 
plt.figure(figsize=(9,6))
plt.bar(cent_a, a_hist, width=width_a, align="center", color = 'black')
plt.yscale("log")
plt.xlim(x_start_a, a_range[1])
plt.xlabel("Alpha peak amplitude (V)", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.title("Alpha amplitude spectrum (all peaks)", fontsize = 20)
#plt.savefig('AlphaALL.png')
plt.show()

# gamma
plt.figure(figsize=(9,6))
plt.bar(cent_g, g_hist*2, width=width_g, align="center", color = 'black')
plt.yscale("log")
plt.xlim(x_start_g, g_range[1])
plt.xlabel("Gamma peak amplitude (V)", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.title("Gamma amplitude spectrum (all peaks)", fontsize = 20)
#plt.savefig('GammaALL.png')
plt.show()






# coincident events amplitude spectra 
# alpha
plt.figure(figsize=(9,6))
plt.hist(g_cond_V, bins=300, color = 'black')
plt.yscale("log")
plt.xlabel("Gamma amplitude in coincidences (V)", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.title("Gamma amplitude spectrum, used in coincidences", fontsize = 20)
#plt.savefig('GammaCoincidence.png')
plt.show()

# gamma
plt.figure(figsize=(9,6))
plt.hist(a_cond_V, bins=300, color = 'black')
plt.yscale("log")
plt.xlabel("Alpha amplitude in coincidences (V)", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.title("Alpha amplitude spectrum, used in coincidences", fontsize = 20)
#plt.savefig('AlphaCoincidence.png')
plt.show()



# In[9]:


# making timing spectrum

delta_t_ns = np.asarray(dt_peak_ns, dtype=float)
delta_t_ns = delta_t_ns[np.isfinite(delta_t_ns)]
#delta_t_ns = delta_t_ns[delta_t_ns >= 0]

t_max_plot = max_dt_samp * dt_ns
delta_t_ns = delta_t_ns[delta_t_ns <= t_max_plot] # coincidence window

# histogram set up
bins_t = 150
t_min_plot = 0

counts, edges = np.histogram(delta_t_ns, bins=bins_t, range=(-t_max_plot, t_max_plot))
centres = 0.5*(edges[:-1] + edges[1:])

lo_t_cut = 20  # avoiding prompt peak
hi_t_cut = 300

mask = (centres >= lo_t_cut) & (centres <= hi_t_cut) & (centres >= 0) & (counts > 0)

t_fit = centres[mask]
N_fit = counts[mask]

# defining exponential function
def exp_plus_bg(t, N0, tau, B):
    return N0*np.exp(-t/tau) + B
    
# Initial guesses
B_guess = float(np.median(N_fit[-10:])) if len(N_fit) >= 10 else 0.0 #␣background from end of exponential
N0_guess = max(N_fit[0] - B_guess, 1.0)
tau_guess = 67.0 # lifetime guess
p0 = [N0_guess, tau_guess, B_guess]

# error, poisson
sigma = np.sqrt(np.maximum(N_fit, 1)) # for fit
sigma1 = np.sqrt(counts) # for error bars 

popt, pcov = curve_fit(exp_plus_bg, t_fit, N_fit, p0=p0, sigma=sigma, absolute_sigma=True,maxfev=50_000)
perr = np.sqrt(np.diag(pcov))
N0, tau, B = popt
dN0, dtau, dB = perr

# fit
t_smooth = np.linspace(lo_t_cut, hi_t_cut, 800)
fit_smooth = exp_plus_bg(t_smooth, *popt)

# residuals and normalised residuals 
model_fit = exp_plus_bg(t_fit, *popt)   # predicted counts
residuals = N_fit - model_fit           # actual counts
norm_resid = residuals / np.sqrt(np.maximum(N_fit, 1))

nr_mean = float(np.mean(norm_resid))
nr_std  = float(np.std(norm_resid, ddof=1)) if norm_resid.size > 1 else float("nan")

# number of points
N = norm_resid.size

# mean
nr_mean = float(np.mean(norm_resid)) if N > 0 else float("nan")

# standard deviation
nr_std = float(np.std(norm_resid, ddof=1)) if N > 1 else float("nan")

# error on mean (standard error)
nr_mean_err = nr_std / np.sqrt(N) if N > 1 else float("nan")

# error on standard deviation
nr_std_err = nr_std / np.sqrt(2*(N - 1)) if N > 1 else float("nan")

# print nicely
print(f"Normalised residuals:")
print(f"  mean = {nr_mean:.4f} ± {nr_mean_err:.4f}")
print(f"  std  = {nr_std:.4f} ± {nr_std_err:.4f}")

print(f"Lifetime τ = {tau:.2f} ± {dtau:.2f} ns")
#print(f"Normalised residuals mean = {nr_mean:.3f}, std = {nr_std:.3f}")


fig = plt.figure(figsize=(9, 11))
gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[2.2, 1.2, 1.8], hspace=0.45)

#figures
# 1: timing spectrum + fit
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(centres, bins=edges, weights=counts, alpha=0.4, label="Data", color='black', zorder = 2)
#ax1.hist(centres, bins=edges, weights = counts_sub, range=(-t_max_plot, t_max_plot), alpha=0.8, label="Data", color = 'black')
ax1.plot(t_smooth, fit_smooth, linewidth=2,label=rf"Fit ($\tau={tau:.2f}\pm{dtau:.2f}$ ns)", color = 'r')
ax1.errorbar(centres, counts, yerr=sigma1, fmt='none',
             ecolor='black', elinewidth=1, capsize=2, zorder=3)
ax1.set_title("Timing spectrum with exponential fit", fontsize = 20)
ax1.set_ylabel("Counts", fontsize = 20)
ax1.set_xlabel(r"$t_\gamma - t_\alpha$ [ns]", fontsize = 20)
ax1.set_xlim(0, t_max_plot)
ax1.grid(alpha= 0.3, zorder = 0)
ax1.legend(fontsize = 15)

# # 2: residuals vs t 
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.errorbar(t_fit, residuals, yerr=sigma, fmt='o', capsize=2, color = 'black')
ax2.axhline(0, linestyle='--', linewidth=1, color = 'red')
#ax2.set_title("Peak Amplitude Timing, Residuals", fontsize = 20)
ax2.set_ylabel("Residuals", fontsize = 20)
ax2.set_xlabel(r"$t_\gamma - t_\alpha$ [ns]", fontsize = 20)
ax2.grid(alpha= 0.3, zorder = 0)
ax2.set_xlim(0, t_max_plot)

# 3: normalised residual distribution
ax3 = fig.add_subplot(gs[2, 0])
ax3.hist(norm_resid, bins=15, density=True, alpha=0.8, color = 'black', zorder = 2)

x = np.linspace(-5, 5, 400)
gauss01 = (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**2)
ax3.plot(x, gauss01, linestyle='--', linewidth=2, label="N(0,1)", color = 'r')
ax3.axvline(0, linestyle=':', linewidth=1, color= 'r')

#ax3.set_title("Normalised residuals distribution", fontsize = 20)
ax3.set_xlabel(r"Normalised residual", fontsize = 20)
ax3.set_ylabel("Probability density", fontsize = 20)
ax3.grid(alpha= 0.3, zorder = 0)
ax3.legend(fontsize = 15)

plt.savefig('3panel_PEAKK.png')
plt.show()


# In[10]:


# printing some examples for visual checks
if len(coinc_examples) == 0:
    print("No coincidence examples stored (coinc_examples is empty).")
else:
    def baseline_correct_to_volts_local(wf_u16):
        wf = wf_u16.astype(np.int32, copy=False)
        b = np.median(wf[:baseline_samples]).astype(np.float32)
        return (wf.astype(np.float32) - b) * V_per_count

    for (i, a_pk, g_pk) in coinc_examples[1:30]:
        a_V = baseline_correct_to_volts_local(alpha_u16[i])
        g_V = baseline_correct_to_volts_local(gamma_u16[i])

        plt.figure(figsize=(10,5))
        plt.plot(t_ns, a_V, label="Alpha", color = 'black')
        plt.plot(t_ns, g_V, label="Gamma", color = 'red')
        plt.axvline(a_pk * dt_ns, linestyle="--", label="Alpha peak time", color = 'black')
        plt.axvline(g_pk * dt_ns, linestyle="--", label="Gamma peak time", color = 'red')
        #plt.plot(a_pk*dt_ns, -0.203, 'x', markersize=10, label="Alpha Pulse", color = 'black')
        #plt.plot(g_pk*dt_ns, -0.101, 'x', markersize=10, label = 'Gamma Pulse', color = 'red')
        plt.xlabel("Time (ns)", fontsize = 20)
        plt.ylabel("Voltage (V)", fontsize = 20)
        plt.title('Peak Amplitude Timing Example Waveform' ,fontsize = 20)
        plt.legend(fontsize = 15)
        plt.grid(alpha = 0.3)
        plt.savefig('PeakTimingExample.png')
        plt.show()


# In[11]:


# defining range for 2D histogram

a_min = np.min(a_cond_V)
a_max = np.max(a_cond_V)

g_min = np.min(g_cond_V)
g_max = np.max(g_cond_V)

print(a_min, g_min)


# In[12]:


# 2d histogram
# coincidences
plt.figure(figsize=(8,7))

plt.hist2d(a_cond_V, g_cond_V, bins = 200, range = [[a_min, 0], [g_min, 0]], cmin = 1, cmax = 140)

plt.colorbar(label = 'Counts')
#plt.clim(1, np,max(h[0]))
plt.xlabel('Alpha peak amplitude (V)', fontsize = 20)
plt.ylabel('Gamma peak amplitude (V)', fontsize = 20)
plt.title('Alpha vs Gamma amplitudes (Coincidences)', fontsize = 20)
plt.savefig('2dhist_PEAK.png')
plt.show()


# In[13]:


# peak fitting, for the parabolic interpolation, needed in order to find this

bins = 200
tmin, tmax = -100, 300
counts, edges = np.histogram(delta_t_ns, bins=bins, range=(tmin, tmax))
centres = 0.5 * (edges[:-1] + edges[1:])


# prompt peak fit window
peak_lo, peak_hi = -10, 25

peak_mask = (centres >= peak_lo) & (centres <= peak_hi)
x = centres[peak_mask]
y = counts[peak_mask]

# define gaussian (+ constant maybe)
def gauss_bg(t, A, mu, sigma, C):
    return A * np.exp(-0.5 * ((t - mu) / sigma)**2) + C

# initial guesses from the window
C0 = np.median(y)  # baseline guess
A0 = np.max(y) - C0  # amplitude guess
mu0 = x[np.argmax(y)]  # mean guess
# sigma guess: a few ns (depends on your timing resolution); pick something reasonable
sigma0 = 4

p0 = [A0, mu0, sigma0, C0]

sigma_y = np.sqrt(np.maximum(y, 1))
popt, pcov = curve_fit(gauss_bg, x, y, p0=p0, sigma=sigma_y, absolute_sigma=True)

A, mu, sigma, C = popt
perr = np.sqrt(np.diag(pcov))
A_err, mu_err, sigma_err, C_err = perr

# FWHM 
k = 2.0 * np.sqrt(2.0 * np.log(2.0))      # 2.35482...
FWHM = k * sigma
FWHM_err = k * sigma_err

print(f"Prompt Gaussian fit:")
print(f"  mu    = {mu:.3f} ± {mu_err:.3f} ns")
print(f"  sigma = {sigma:.3f} ± {sigma_err:.3f} ns")
print(f"  FWHM  = {FWHM:.3f} ± {FWHM_err:.3f} ns")



# plotting
t_fit = np.linspace(peak_lo, peak_hi, 1000)
y_fit = gauss_bg(t_fit, *popt)

plt.figure(figsize=(8,4))
plt.step(centres, counts, where="mid", label="Data", color = 'black')
plt.plot(t_fit, y_fit, label=f"Gaussian fit (FWHM={FWHM:.2f} ns)", color = 'r')
plt.xlim(peak_lo - 5, peak_hi + 5)
plt.xlabel(r"$t_\gamma - t_\alpha$ [ns]", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.legend(fontsize = 12)
plt.tight_layout()
#plt.savefig('PeakWidthLE.png')
plt.show()


# In[14]:


# half life

halflife = tau * np.log(2)
halflife_err = dtau * np.log(2)

print(f"Half Life = {halflife:.2f} ± {halflife_err:.2f} ns")


# In[ ]:





# In[ ]:




