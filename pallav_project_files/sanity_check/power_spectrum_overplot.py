import numpy as n
import aipy as a
from matplotlib import pyplot as plt

#=================== Functions imported from calc_sense.py =====================
'''Some of use, some useless.'''

#Convert frequency (GHz) to redshift for 21cm line.
def f2z(fq):
    F21 = 1.42040575177
    return (F21 / fq - 1)

#Multiply by this to convert an angle on the sky to a transverse distance in Mpc/h at redshift z
def dL_dth(z):
    '''[h^-1 Mpc]/radian, from Furlanetto et al. (2006)'''
    return 1.9 * (1./a.const.arcmin) * ((1+z) / 10.)**.2

#Multiply by this to convert a bandwidth in GHz to a line of sight distance in Mpc/h at redshift z
def dL_df(z, omega_m=0.266):
    '''[h^-1 Mpc]/GHz, from Furlanetto et al. (2006)'''
    return (1.7 / 0.1) * ((1+z) / 10.)**.5 * (omega_m/0.15)**-0.5 * 1e3

#Multiply by this to convert a baseline length in wavelengths (at the frequency corresponding to redshift z) into a tranverse k mode in h/Mpc at redshift z
def dk_du(z):
    '''2pi * [h Mpc^-1] / [wavelengths], valid for u >> 1.'''
    return 2*n.pi / dL_dth(z) # from du = 1/dth, which derives from du = d(sin(th)) using the small-angle approx

#Multiply by this to convert eta (FT of freq.; in 1/GHz) to line of sight k mode in h/Mpc at redshift z
def dk_deta(z):
    '''2pi * [h Mpc^-1] / [GHz^-1]'''
    return 2*n.pi / dL_df(z)

#scalar conversion between observing and cosmological coordinates
def X2Y(z):
    '''[h^-3 Mpc^3] / [str * GHz]'''
    return dL_dth(z)**2 * dL_df(z)

#=================== Model PS21 file and other parameters ======================

z = f2z(0.135)
h = 0.7

modelfile = 'ps_no_halos_nf0.521457_z9.50_useTs0_zetaX-1.0e+00_200_400Mpc_v2'
model = n.loadtxt(modelfile)
mk, mpk = model[:,0]/h, model[:,1] #k, Delta^2(k)
#note that we're converting from Mpc to h/Mpc

sensefile = 'hera127.drift_mod_0.135.npz'
array = n.load(sensefile)
k_s = array['ks']
sense = array['errs']
Tsense = array['T_errs']

B = 0.008
deltat = 60.
dish_size_in_lambda = 7.
dish_size_in_lambda = dish_size_in_lambda*(0.135/.150)
bm = 1.13*(2.35*(0.45/dish_size_in_lambda))**2
bm2 = bm/2.
bm_eff = bm**2 / bm2
Tsys = 180e3 * (135./180.)**-2.5

#=============================== MAIN CODE =====================================

nn = 200 #Change in case you wish to increase the number of points at which GRF Noise (and correspondingly k, etc.) are generated
kmag = n.linspace(min(mk),n.max(mk), nn) #k values

sigma = Tsys / n.sqrt(B*1e9 * deltat)

s = n.random.normal(0, sigma, nn)
noise = s*s #only thermal noise.

PSN, PS, PS21 = [], [], []

for i, k in enumerate(kmag):
    psn = X2Y(z) * (k**3/2*n.pi**2) * bm_eff * B / 1e9 * noise[i] #Formula is correct as per my understanding. Please refer Parsons 2012 Pg.8 expl. above eq.16.
    #since mean=0 for GRF, the GRF only fluctuates according to sigma and thus the noise (from GRF) can be attributed as the error in how well Tsys is measured.
    PSN.append(psn)
    ps21 = n.interp(k, mk, mpk)
    PS21.append(ps21)
    ps = ps21 + psn #no need for inverse quadrature addition as we are not combining any k-mode measurements.
    PS.append(ps)

#Write to csv file
l = len(PS)
with open('PS.csv','w+') as f:
    f.write('k, PS21, Noise PS, Obsvd PS (PS21 + Noise)\n')
    for i in xrange(l):
        f.write('%f, %f, %f, %f\n' % (kmag[i], PS21[i], PSN[i], PS[i]))

#Plotting
plt.figure(figsize = (12,4))
plt.plot(kmag, PSN, 'c--', label = 'Noise PS')
plt.plot(kmag, PS, 'g-', label = 'Observed PS (PS21 + Noise PS)')
plt.plot(kmag, PS21, 'r-.', label = 'PS21')
plt.plot(k_s, sense, 'b-', drawstyle = 'steps', label = 'Full Sensitivity')
plt.plot(k_s, Tsense, 'y:', drawstyle = 'steps', label = 'Sensitivity [Thermal Noise Only]')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k\ [h\ Mpc^{-1}]$')
plt.ylabel(r'$\Delta^{2}(k)\ [mK^{2}]$')
plt.legend()
plt.savefig('PS_overplot.png', dpi = 600)
plt.show()






