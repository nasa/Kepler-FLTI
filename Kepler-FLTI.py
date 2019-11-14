# -*- coding: utf-8 -*-
"""
Kepler-FLTI.py - Illustrate using the Flux-Level Transit Injection (FLTI) Tests
    of TPS for Data Release 25.  The FLTI test output is in the FITs file
    format.  This code generates the figures in the documentation of FLTI
    Burke, C.J. & Catanzarite, J. 2017, "Planet Detection Metrics: 
       Per-Target Flux-Level Transit Injection Tests of TPS
       for Data Release 25", KSCI-19109-001
    Assumes python packages astropy, numpy, and matplotlib are available
      and file kplr007702838_dr25_5008_flti.fits is available in the 
      same directory as Kepler-FLTI.py
    Invocation: python Kepler-FLTI.py
    Output: Displays a series of figures and generates hardcopy

Notices:

Copyright © 2017 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

NASA acknowledges the SETI Institute’s primary role in authoring and producing the Plotting Program for Kepler Planet Detection Efficiency Products under Cooperative Agreement Number NNX13AD01A.


Disclaimers

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
 
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

def show_basic_fits_data(hdulist):
    """ Print to terminal basic data about fits file along with
        primary header and data table header"""
    #Fits file info
    hdulist.info()
    #Primary Header
    print(repr(hdulist[0].header))
    #Data Table Header
    print(repr(hdulist[1].header))
    return
    
def show_empirical_detection_contour(hdulist):
    """Calculate empirical detection contour based upon
        FLTI output.  Also
        produce figure of output"""
    fltidata = hdulist[1].data
    # Get injected data and whether injection was recovered
    injPeriod = np.log10(fltidata.field('Period'))
    injRp = np.log10(fltidata.field('Rp'))
    recvrFlag = fltidata.field('Recovered')
    
    # Get grid dimensions
    logRpMax = np.log10(hdulist[0].header['RPMAX'])
    logRpMin = np.log10(hdulist[0].header['RPMIN'])
    logPerMax = np.log10(hdulist[0].header['PERMAX'])
    logPerMin = np.log10(hdulist[0].header['PERMIN'])
    nInjection = hdulist[1].header['NAXIS2']
    kicWant = hdulist[0].header['KEPLERID']
    print("KIC: {0:09d} Num Inj: {1:d}".format(kicWant, nInjection))
    # Set bin edge spacing to roughly achieve nWantPerBin
    # injections per bin.  Always have a minimum minNBin bins
    # each dimension
    nWantPerBin = 100
    minNBin = 7
    twoDNBin = nInjection / nWantPerBin
    oneDNBin = np.sqrt(twoDNBin)
    nXBin = np.uint32(np.floor(oneDNBin))
    if nXBin < minNBin:
        nXBin = minNBin
    # Add 1 to Y direction number of bins to make differentiating between
    #  X and Y dimensions trivial for the 2D array
    nYBin = nXBin + 1
    # Orbital period is assigned x dimension
    # Planet Radius is assigned y dimension
    print("X dimen Porb: {0:d} Bins Y dimen Rp: {1:d} Bins".format(nXBin, nYBin))
    # Use numpy histogram2d to return counts of injected signals in 2d grid
    nAll = np.histogram2d(injPeriod, injRp, \
                bins=(nXBin,nYBin), \
                range=[[logPerMin, logPerMax], [logRpMin, logRpMax]], \
                normed=False)[0]
    # Identify injected signals that are recovered
    # Return counts of recovered signals in 2d grid
    idxRecvr = np.where(recvrFlag == 1)[0]
    nRecvr, xedges, yedges = np.histogram2d(injPeriod[idxRecvr], \
                injRp[idxRecvr], \
                bins=(nXBin,nYBin), \
                range=[[logPerMin, logPerMax], [logRpMin, logRpMax]], \
                normed=False)
    # Detection contour is number recovered / number injected for each bin
    probdet = np.double(nRecvr) / np.double(nAll)

    # Begin showing detection probability contour
    # Get the bin centers from edges and make a 2d version of bin centers
    midx = xedges[:-1] + np.diff(xedges)/2.0
    midy = yedges[:-1] + np.diff(yedges)/2.0
    X2, Y2 = np.meshgrid(midx, midy)
    # Setup figure filename and figures
    wantFigure = 'emp_det_cont_{0:09d}'.format(kicWant)
    fig, ax, fsd = setup_figure()
    # Define contour levels to show
    uselevels = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        
    CS2 = plt.contour(X2, Y2, probdet.T, levels=uselevels, 
                          linewidth=fsd['datalinewidth'], 
                          colors=(fsd['myblue'],) * len(uselevels))
    plt.clabel(CS2, inline=1, fontsize=fsd['labelfontsize'], fmt='%1.2f', 
                   inline_spacing=10.0, fontweight='ultrabold')
    CS1 = plt.contourf(X2, Y2, probdet.T, levels=uselevels, cmap=plt.cm.bone)    
    plt.xlabel('Log10(Period) [day]', fontsize=fsd['labelfontsize'], 
                   fontweight='heavy')
    plt.ylabel('Log10(R$_{p}$) [R$_{\oplus}$]', fontsize=fsd['labelfontsize'], 
                   fontweight='heavy')
    ax.set_title('KIC: {0:d}'.format(kicWant))
    for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(fsd['plotboxlinewidth'])
                ax.spines[axis].set_color(fsd['mynearblack'])
    ax.tick_params('both', labelsize=fsd['tickfontsize'], 
                       width=fsd['plotboxlinewidth'], 
                       color=fsd['mynearblack'], length=fsd['plotboxlinewidth']*3)
    # Make eps and png hard copy of figure
    plt.savefig(wantFigure+'.png',bbox_inches='tight')
    plt.savefig(wantFigure+'.eps',bbox_inches='tight')
    plt.show()

    return

def show_window_function(hdulist):
    """Calculate empirical window function based upon
        FLTI output.  Also
        produce figure of output"""
    fltidata = hdulist[1].data
    # Get injected data and output
    injPeriods = fltidata.field('Period')
    injDurations = fltidata.field('t_dur')
    nTransits = fltidata.field('ralt_ntran')
    transitWghtChk = fltidata.field('ralt_threeTransitFail')
    injRps = fltidata.field('Rp')
    injImps = fltidata.field('b')
    # Get grid dimensions
    rpMax = hdulist[0].header['RPMAX']
    rpMin = hdulist[0].header['RPMIN']
    perMax = hdulist[0].header['PERMAX']
    perMin = hdulist[0].header['PERMIN']
    nInjection = hdulist[1].header['NAXIS2']
    kicWant = hdulist[0].header['KEPLERID']
    # Get stellar parameters
    rStar = hdulist[0].header['RADIUS']
    # Get target noise
    cdppHeadStr = ['RCDP01P5','RCDP02P0','RCDP02P5','RCDP03P0','RCDP03P5',\
                    'RCDP04P5','RCDP05P0','RCDP06P0','RCDP07P5','RCDP09P0', \
                    'RCDP10P5','RCDP12P0','RCDP12P5','RCDP15P0']
    pulsedurs=np.array([1.5,2.0,2.5,3.0,3.5,4.5,5.0,6.0,7.5,9.0,10.5,12.0,12.5,15.0])                
    cdpps = np.zeros_like(pulsedurs)
    for idx, hdrstring in enumerate(cdppHeadStr):
        cdpps[idx] = hdulist[0].header[hdrstring]
        
    print("KIC: {0:09d} Num Inj: {1:d}".format(kicWant, nInjection))
    # We need to find injections that are expected to be in the high
    # SNR regime.  We need to do the selection of targets by injected Rp
    # rather than by MES alone because MES depends upon # of transits
    #  and we have to keep injections in the sample that have MES=0
    # due to having no transits in order to get the statistics correct
    # for the window function calculation
    # We want to find an injected Rp that has a characteristic high MES
    # Define the high MES we want to roughly achieve with an Rp cut.
    minMES = 13.0
    # Define the maximum impact parameter so the SNR isn't too supressed
    maxImp = 0.7
    #  Most targets have nearly 100% detection efficiency by MES=13
    # First need to get the transit duration expected for the longest
    # orbital periods represented in the injections in order to get cdpp
    # noise on that duration
    minNTran = np.max([np.min(nTransits), 3])
    # Find injections that have minNTran events
    idx = np.where( (nTransits == minNTran) & (np.isfinite(injDurations)) )[0]
    expDuration = np.median(injDurations[idx])
    useCdpp = np.interp([expDuration], pulsedurs, cdpps)
    # Find the orbital period range that is relevant to the data
    expMaxPeriod = np.median(injPeriods[idx])*1.55
    maxNTran = np.min([np.max(nTransits), 10])
    idx = np.where( (nTransits == maxNTran) & (np.isfinite(injDurations)))[0]
    expMinPeriod = np.median(injPeriods[idx])
    
    # Find simple geometric depth [ppm] for 1 Rp [Rearth] planet around
    # a star with this targets size and noise.
    rearthDRsun = 6378137.0/696000000.0
    k = 1.0 / rStar * rearthDRsun
    depth = k * k * 1.0e6
    # With depth for 1Rp [Rearth] planets determine MES of this planet
    oneEarthMes = depth / useCdpp * np.sqrt(3.0)
    mesRatio = minMES / oneEarthMes
    useRpMin = np.sqrt(mesRatio)
    print("Use Rp Min [Rearth]: {0:f}".format(useRpMin[0]))

    # Trim the data outside period and rp range wanted
    idx = np.where((injRps > useRpMin) & (injPeriods > expMinPeriod) & \
                (injPeriods < expMaxPeriod) & (injImps < maxImp))[0]
    usePeriods = injPeriods[idx]
    useNTransits = nTransits[idx]
    useTransitWghtChk = transitWghtChk[idx]
    useN = usePeriods.size
    # Set bin edge spacing to roughly achieve nWantPerBin
    # injections per bin.  Always have a minimum minNBin bins
    nWantPerBin = 300
    minNBin = 30
    oneDNBin =  useN / nWantPerBin
    nXBin = np.uint32(np.floor(oneDNBin))
    if nXBin < minNBin:
        nXBin = minNBin
    
    # Use numpy histogram to return counts of injected signals in period
    nAll = np.histogram(usePeriods, bins=nXBin, \
                range=(expMinPeriod,expMaxPeriod), normed=False)[0]
    # Identify injected signals that pass window function
    idxPass = np.where((useNTransits > 3) | ((useNTransits == 3) & (useTransitWghtChk == 0)))[0]
    nPass, xedges = np.histogram(usePeriods[idxPass], bins=nXBin, \
                range=(expMinPeriod,expMaxPeriod), normed=False)
    # Window function is number recovered / number injected for each bin
    winFunction = np.double(nPass) / np.double(nAll)
    midx = xedges[:-1] + np.diff(xedges)/2.0
    print("Kic: {0:d} useN: {1:d} nBin: {2:d}".format(kicWant, useN, len(midx))) 

    # Setup figure filename and figures
    wantFigure = 'emp_win_func_{0:09d}'.format(kicWant)
    fig, ax, fsd = setup_figure()
    plt.plot(midx, winFunction, linewidth=fsd['datalinewidth'])
    plt.xlabel('Period [day]', fontsize=fsd['labelfontsize'], fontweight='heavy')
    plt.ylabel('Window Function', fontsize=fsd['labelfontsize'], 
                    fontweight='heavy')
    ax.set_title('KIC: {0:d}'.format(kicWant))
    for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(fsd['plotboxlinewidth'])
                ax.spines[axis].set_color(fsd['mynearblack'])
    ax.tick_params('both', labelsize=fsd['tickfontsize'], width=fsd['plotboxlinewidth'], 
                       color=fsd['mynearblack'], length=fsd['plotboxlinewidth']*3)
    # Make eps and png hard copy of figure
    plt.savefig(wantFigure+'.png',bbox_inches='tight')
    plt.savefig(wantFigure+'.eps',bbox_inches='tight')
    plt.show()

def show_detection_efficiency(hdulist):
    """Calculate empirical detection efficiency based upon
        FLTI output.  Also
        produce figure of output"""
    fltidata = hdulist[1].data
    # Get injected data and output
    injPeriods = fltidata.field('Period')
    injDurations = fltidata.field('t_dur')
    nTransits = fltidata.field('ralt_ntran')
    transitWghtChk = fltidata.field('ralt_threeTransitFail')
    expMes = fltidata.field('exp_mes')
    recvrFlag = fltidata.field('Recovered')
    # Get grid dimensions
    rpMax = hdulist[0].header['RPMAX']
    rpMin = hdulist[0].header['RPMIN']
    perMax = hdulist[0].header['PERMAX']
    perMin = hdulist[0].header['PERMIN']
    nInjection = hdulist[1].header['NAXIS2']
    kicWant = hdulist[0].header['KEPLERID']
    print("KIC: {0:09d} Num Inj: {1:d}".format(kicWant, nInjection))
    # We need to find injections that pass the window function so
    # the detection efficiency is not contaminated by injections
    # due to window function effects.  Also avoid injections
    # with durations longer than 15 hours since the pipeline
    # artifically suppresses their MES due to not searching
    # toward longer durations
    maxDur = 15.0
    # Define the range of mes and spacing to calculate detection efficiency for
    minMes = 3.5
    maxMes = 30.0
    delMes = 0.5
    # Define the range of periods to calculate detection eff.
    minPer = 1.0
    maxPer = 700.0
    # Passes Window Function Tests
    passWinFunction = ((nTransits > 3) |  \
                    ((nTransits == 3) & \
                     (transitWghtChk == 0)))

    # Only keep injections that are within the requested period range
    #  and were not rejected due to window function effects
    idxKeep = np.where((injPeriods >= minPer) & (injPeriods <= maxPer) & \
                    (passWinFunction) & \
                     (injDurations < maxDur))[0]
    recvrFlag = recvrFlag[idxKeep]
    expMes = expMes[idxKeep]
    injPeriods = injPeriods[idxKeep]
    useN = injPeriods.size

    # Start the binning
    nBins = np.round((maxMes - minMes) / delMes)
    xedges = np.linspace(minMes, maxMes, nBins+1)
    midx = xedges[:-1] + np.diff(xedges)/2.0
    print("Kic: {0:d} useN: {1:d} nBin: {2:d}".format(kicWant, useN, len(midx))) 
    detectionEff = np.zeros_like(midx)
    detectionEffN = np.zeros_like(midx)
    if idxKeep.size > 5:
        binidx = np.digitize(expMes, xedges)
        for i in np.arange(1,len(xedges)):
            idxInBin = np.where(binidx == i)[0]
            if idxInBin.size > 0: 
                curPc = recvrFlag[idxInBin]
                idxPcInBin = np.where(curPc == 1)[0]
                detectionEff[i-1] = np.double(idxPcInBin.size) / np.double(idxInBin.size)
                n = np.double(idxInBin.size)
                detectionEffN[i-1] = n
            else:
                detectionEff[i-1] = 1.0
                detectionEffN[i-1] = 0
            print("Kic: {0:d} Mes: {1:f} DetEff: {2:f} NinBin: {3:f}".format( \
                        kicWant, midx[i-1], detectionEff[i-1], detectionEffN[i-1]))
    # Setup figure filename and figures
    wantFigure = 'emp_det_eff_{0:09d}'.format(kicWant)
    fig, ax, fsd = setup_figure()
    plt.plot(midx, detectionEff, linewidth=fsd['datalinewidth'])
    plt.xlabel('Expected MES', fontsize=fsd['labelfontsize'], fontweight='heavy')
    plt.ylabel('Detection Efficiency', fontsize=fsd['labelfontsize'], 
                    fontweight='heavy')
    ax.set_title('KIC: {0:d}'.format(kicWant))
    for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(fsd['plotboxlinewidth'])
                ax.spines[axis].set_color(fsd['mynearblack'])
    ax.tick_params('both', labelsize=fsd['tickfontsize'], width=fsd['plotboxlinewidth'], 
                       color=fsd['mynearblack'], length=fsd['plotboxlinewidth']*3)
    # Make hard copy of figures
    plt.savefig(wantFigure+'.png',bbox_inches='tight')
    plt.savefig(wantFigure+'.eps',bbox_inches='tight')
    plt.show()



def setup_figure():
    '''Setup up figure and make a dictionary of preferred plotting styles'''
    # Define colors, font sizes, line widths, and marker sizes
    myblack = tuple(np.array([0.0, 0.0, 0.0]) / 255.0)
    mynearblack = tuple(np.array([75.0, 75.0, 75.0]) / 255.0)
    myblue = tuple(np.array([0.0, 109.0, 219.0]) / 255.0)
    myred = tuple(np.array([146.0, 0.0, 0.0]) / 255.0)
    myorange = tuple(np.array([219.0, 209.0, 0.0]) / 255.0)
    myskyblue = tuple(np.array([182.0, 219.0, 255.0]) / 255.0)
    myyellow = tuple(np.array([255.0, 255.0, 109.0]) / 255.0)
    mypink = tuple(np.array([255.0, 182.0, 119.0]) / 255.0)
    labelfontsize = 19.0
    tickfontsize = 14.0
    datalinewidth = 3.0
    plotboxlinewidth = 3.0
    markersize = 1.0
    bkgcolor = 'white'
    axiscolor = myblack
    labelcolor = myblack
    fig = plt.figure(figsize=(8,8), facecolor=bkgcolor)
    ax = plt.gca()
    
    figstydict={'labelfontsize':labelfontsize, 'tickfontsize':tickfontsize, \
                'datalinewidth':datalinewidth, 'plotboxlinewidth':plotboxlinewidth, \
                'markersize':markersize, 'bkgcolor':bkgcolor, \
                'axiscolor':axiscolor, 'labelcolor':labelcolor, \
                'myblack':myblack, 'mynearblack':mynearblack, \
                'myblue':myblue, 'myred':myred, 'myorange':myorange, \
                'myskyblue':myskyblue, 'myyellow':myyellow, 'mypink':mypink}
    return fig, ax, figstydict
          
    
if __name__ == "__main__":
    # To run program 'python Kepler-FLTI.py' at command line
    # fitsfile kplr007702838_dr25_5008_flti.fits also needs to be available
    fitsfile='kplr007702838_dr25_5008_flti.fits'
    # Open fits file
    hdulist = fits.open(fitsfile,mode='readonly')

    show_basic_fits_data(hdulist)
    
    show_empirical_detection_contour(hdulist)
    
    show_window_function(hdulist)
    
    show_detection_efficiency(hdulist)    
    
    
    hdulist.close()
