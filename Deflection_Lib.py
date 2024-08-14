### Functions for the Deflection Simulation ###
###############################################
import csv

import numpy as np
from math import pi, sqrt

from numpy import linalg, array, zeros, tanh
from scipy.special import erf, erfinv
from scipy.interpolate import interp1d
from math import exp,sqrt, log, sinh, cosh, floor
from pathlib import Path
from scipy.optimize import curve_fit
#from scipy.signal import fftconvolve

#import matplotlib.pyplot as plt
mass_Helium = 6.6464764e-27  #Mass of helium atom in kg
energy_to_boil_1_He = 9.94066934e-23 # energy in J (CONFIRM?)
kB = 1.38064852e-23
He_susceptibility = 5.7e-2 #Susceptibility of liquid helium
eps0 = 8.854e-12 #permittivity of free space
eps = 1.057*eps0 #permittivity of liquid helium

l1 = 0.15       #length of deflection plates
l2_E = 1.65       #length of free flight region for electric deflection
l2_M = 1.25       #length of free flight region for magnetic deflection

def Boiloff(N, mass_of_dopant, temp_of_dopant, droplet_velocity): #TESTED AND WORKS
    # Boil off from KE deposition 
    mDop = mass_of_dopant
    TDop = temp_of_dopant
    vz = droplet_velocity

    eDop = 0.0
    vProb = sqrt(2.0*kB*TDop/mDop)
    xJL = vz/vProb
    mReduced = (mDop*N*mass_Helium)/(mDop+N*mass_Helium)
    eDop = kB*TDop*(mReduced/mDop)*(xJL*(5.0/2.0 + xJL*xJL)*exp(-xJL*xJL) + sqrt(np.pi)*(3.0/4.0 + 3.0*xJL*xJL + xJL**4.0)*erf(xJL))/(xJL*exp(-xJL*xJL)+sqrt(np.pi)*(0.5 + xJL*xJL)*erf(xJL))
    # print (int(floor(eDop/energy_to_boil_1_He))) Around 350 for Cl2Fe 
    # Boil off from rotational contribution??
    eDop += 2.0/2.0*kB*TDop # Average rotational degrees of freedom from equipartition (Only two because we have a linear rotor)
    # Boil off from vibrational contribution

    dN = int(floor(eDop/energy_to_boil_1_He))

    # Make sure there are enough He left, set to 0 if not
    if (dN<N):
        N -= dN
    else:
        N = 0
    return N

def ElectricDeflection(number_of_He_atoms_in_droplet, effective_dipole_moment, plate_voltage, mass_of_dopant, He_cavity_radius, droplet_velocity): # TESTED, SEEMS TO WORK
    'Implement deflection of the droplet between deflection plates an thru free flight region'
    # Where do these numbers come from? Conversion to SI units I'm sure.
    pEff = effective_dipole_moment
    N = number_of_He_atoms_in_droplet
    volts = plate_voltage
    mDop = mass_of_dopant
    molR = He_cavity_radius
    vz = droplet_velocity
    E0 = volts*(411413.0)       # Electric field at center position
    E1 = volts*(-1.68888e8)     # First derivative of electric field
    E2 = volts*(6.18785e10)     # Second derivative of electric field

    p = pEff*3.336e-30		        #Effective dipole moment 
    p += pInd(N, p, E0, molR)     #Add polarization of the helium droplet
    
    mDrop = N*mass_Helium + mDop	#Total mass of the doped helium droplets
    
    om = sqrt(abs(p*E2/mDrop))	#Time constant in dynamics
            
    a0 = p*E1/mDrop		#Initial acceleration
    
    #Position at exit of deflection plates
    x1 = ( a0/(om*om) )*(cosh(l1*om/vz) - 1.0)
    
    #Transverse velocity at exit of deflection plates
    v1 = ( a0/om )*sinh(l1*om/vz)
    
    #Position at end of free flight path
    x2 = -1000*(x1 + v1*l2_E/vz)

    return x2


def MagneticDeflection(number_of_He_atoms_in_droplet, effective_dipole_moment, mass_of_dopant, droplet_velocity): # TESTED, SEEMS TO WORK
    'Implement deflection of the droplet between deflection plates an thru free flight region'
    # Where do these numbers come from? Conversion to SI units I'm sure.
    pEff = float(effective_dipole_moment)
    N = number_of_He_atoms_in_droplet
    mDop = mass_of_dopant
    vz = droplet_velocity
    p = float(pEff)*9.274e-24	         #Effective magnetic moment 
    mDrop = N*mass_Helium + mDop	#Total mass of the doped helium droplets
    
    #Position at end of free flight path
    x2 = (.5*(l1)*(l1)+(l1)*(l2_M))/((mDrop)*(vz)*(vz))*p*334*1000 #Everything in SI Standard, but times 1000 to convert to mm

    return x2


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# Asymmetric PV functions
def PV_func(x,A,f,x0,a,gamma):
    # Defines the Assymmetric V function; Stancik and Brauns, 2008
    return f*(2*A/(np.pi*(2*gamma/(1+np.exp(a*(x-x0))))))/(1+4*((x-x0)/(2*gamma/(1+np.exp(a*(x-x0)))))**2) + (1-f)*(A/(2*gamma/(1+np.exp(a*(x-x0)))))*((4*np.log(2)/np.pi)**(1/2))*np.exp(-4*np.log(2)*((x-x0)/(2*gamma/(1+np.exp(a*(x-x0)))))**2)

def get_PV_params(x_data,y_data):
    # requires PV function defined; initial guess and bounds given
    popt = curve_fit(PV_func,x_data,y_data,maxfev=100000,p0=[np.max(y_data),0.5,np.mean(x_data),1,1],bounds=([0,0,-10,-np.inf,-np.inf],[np.inf,1,10,np.inf,np.inf]))
    return popt[0]

def get_PV_points(x_range,PV_params):
    # requires PV function defined
    y_range_fit = PV_func(x_range,PV_params[0],PV_params[1],PV_params[2],PV_params[3],PV_params[4])
    return y_range_fit

def PV_Fit(x_input,y_input,x_output):
    # x_input, y_input are the data points, x_output is the domain to use in the simulation
    x_data = x_input.copy()
    y_data = y_input.copy()
    pv_params = get_PV_params(x_data,y_data)
    y_output = get_PV_points(x_output,pv_params)
    return [x_output, y_output]

# Symmetric

def PV_func_sym(x,A,f,x0,gamma0):
    # Defines the Symmetric V function; Stancik and Brauns, 2008
    return f*(2*A/(np.pi*gamma0))/(1+4*((x-x0)/(gamma0))**2) + (1-f)*(A/(gamma0))*((4*np.log(2)/np.pi)**(1/2))*np.exp(-4*np.log(2)*((x-x0)/(gamma0))**2)

def get_PV_params_sym(x_data,y_data):
    # requires Symmetric PV function defined; initial guess and bounds given
    popt = curve_fit(PV_func_sym,x_data,y_data,maxfev=100000,p0=[np.max(y_data),0.5,np.mean(x_data),1],bounds=([0,0,-10,-np.inf],[np.inf,1,10,np.inf]))
    return popt[0]

def get_PV_points_sym(x_range,PV_params):
    # requires Symmetric PV function defined
    y_range_fit = PV_func_sym(x_range,PV_params[0],PV_params[1],PV_params[2],PV_params[3])
    return y_range_fit

def PV_Fit_sym(x_input,y_input,x_output):
    # x_input, y_input are the data points, x_output is the domain to use in the simulation
    # Symmetric Version
    x_data = x_input.copy()
    y_data = y_input.copy()
    pv_params = get_PV_params_sym(x_data,y_data)
    y_output = get_PV_points_sym(x_output,pv_params)
    return [x_output, y_output]

### Create the Hamiltonian of the Rigid Rotor in the Electric Field ###
def StarkH(m, jmax, A, B, C, pa, pb, pc, E): # apparently works since StarkRotor works... Check math
    "Build the Hamiltonian block for quantum number m for a particular applied electric field (E)"
    
    #Initialize block
    blockSize = (jmax - m + 1)*(jmax + m + 1)
    H = zeros((blockSize,blockSize),dtype=complex)
    
    for j in range(abs(m),jmax+1):
            
        for k in range(-j,j+1): #k runs from -j to j
            
            #Kinetic Energy Terms
            H[ind(j,k,m)][ind(j,k,m)] = ((B+C)/2) * (j*(j+1)-k*k) + A*k*k
                
            if k+2<=j:
                H[ind(j,k+2,m)][ind(j,k,m)] = H[ind(j,k,m)][ind(j,k+2,m)] = ((B-C)/4) * sqrt(j*(j+1) - k*(k+1)) * sqrt(j*(j+1) - (k+1)*(k+2))
                
            #Potential Energy Terms
            #Axis a projection
            if j>0:
                H[ind(j,k,m)][ind(j,k,m)] += -E*pa * m*k/(j*(j+1))
                
            if j+1<=jmax:
                H[ind(j+1,k,m)][ind(j,k,m)] = H[ind(j,k,m)][ind(j+1,k,m)] = -E*pa * sqrt((j+1)*(j+1)-k*k)*sqrt((j+1)*(j+1)-m*m) / ( (j+1)*sqrt( (2*j+1)*(2*j+3) ) )
                
            #Axis b projection
            if (k+1<=j)&(j>0):
                H[ind(j,k+1,m)][ind(j,k,m)] = H[ind(j,k,m)][ind(j,k+1,m)] = -E*pb * m*sqrt( (j-k)*(j+k+1) ) / (2*j*(j+1))
                
            if (j+1<=jmax)&(k+1<=j):
                H[ind(j+1,k+1,m)][ind(j,k,m)] = H[ind(j,k,m)][ind(j+1,k+1,m)] = E*pb * sqrt( (j+k+1)*(j+k+2) )*sqrt( (j+1)*(j+1) - m*m )/ ( 2*(j+1)*sqrt( (2*j+1)*(2*j+3) ) )

                H[ind(j+1,k,m)][ind(j,k+1,m)] = H[ind(j,k+1,m)][ind(j+1,k,m)] = -E*pb * sqrt( (j-k)*(j-k+1) )*sqrt( (j+1)*(j+1) - m*m )/ ( 2*(j+1)*sqrt( (2*j+1)*(2*j+3) ) )
                
            #Axis c projection
            if (k+1<=j)&(j>0):
                H[ind(j,k+1,m)][ind(j,k,m)] += E*pc * 1j*m*sqrt( (j-k)*(j+k+1) ) / (2*j*(j+1))
                
                H[ind(j,k,m)][ind(j,k+1,m)] += -E*pc * 1j*m*sqrt( (j-k)*(j+k+1) ) / (2*j*(j+1))
            
            if (j+1<=jmax)&(k+1<=j):
                H[ind(j+1,k+1,m)][ind(j,k,m)] += -E*pc * 1j*sqrt( (j+k+1)*(j+k+2) )*sqrt( (j+1)*(j+1) - m*m )/ ( 2*(j+1)*sqrt( (2*j+1)*(2*j+3) ) )
                
                H[ind(j,k,m)][ind(j+1,k+1,m)] += E*pc * 1j*sqrt( (j+k+1)*(j+k+2) )*sqrt( (j+1)*(j+1) - m*m )/ ( 2*(j+1)*sqrt( (2*j+1)*(2*j+3) ) )
                
                H[ind(j+1,k,m)][ind(j,k+1,m)] += -E*pc * 1j*sqrt( (j-k)*(j-k+1) )*sqrt( (j+1)*(j+1) - m*m )/ ( 2*(j+1)*sqrt( (2*j+1)*(2*j+3) ) )
                
                H[ind(j,k+1,m)][ind(j+1,k,m)] += E*pc * 1j*sqrt( (j-k)*(j-k+1) )*sqrt( (j+1)*(j+1) - m*m )/ ( 2*(j+1)*sqrt( (2*j+1)*(2*j+3) ) )
                
    return H

# Code below still needs to be run through to check the math etc
# Currently saves in the same file as the support functions, can be changed
# Line with the directory is 224 (or ~90 lines from start of func)
def StarkRotor(molecule_name, plate_voltage, rotational_constants_array, dipole_moment_array):# TESTED, WORKS.
    'Generate File of Stark Curves, for electric fields from 0 to 2*E0'
    molName = molecule_name
    volts = plate_voltage
    Bs = rotational_constants_array
    p = dipole_moment_array
    E0 = volts*(411413.0)       # Electric field at center position
    #Projections of dipole moment onto principal axes
    pa = p[0]*3.336e-30
    pb = p[1]*3.336e-30
    pc = p[2]*3.336e-30

    p0 = sqrt( pa*pa + pb*pb + pc*pc ) #Total dipole moment

    #Field paramters
    esteps = 20 #Number of steps to take from 0 to E0

    #Planck constant
    h = 6.62607004e-34

    #Rotational constants
    A = h*Bs[0]*1.0e6
    B = h*Bs[1]*1.0e6
    C = h*Bs[2]*1.0e6

    ##Zero-valued rotational constants are assumed to be very large (like in a linear rotor)
    if A==0:
        A = 1e3*(B+C)

    if B==0:
        B = 1e3*(A+C)

    if C==0:
        C = 1e3*(A+B)

    #Size of basis set
    jmax = 15

    #Maximum number of eigenvalues to return for each m
    nmax = 30
    
    if nmax > 2*jmax:
        print ("nmax > 2*jmax, changing to nmax = 2*jmax")
        nmax = 2*jmax

    #Write table of eigenvalues
    print ('Generating New Stark Curves...')
    print ('')

    eigsSmall = np.full((2,2*nmax),(jmax+1)*1.0) #Array that will contain the lowest eigenvalues at zero field
    
    #First determine the lowest zero-field rotational states
    for m in range(-jmax,jmax+1):
    
        eigsm = np.array([np.full((1 + jmax - m)*(1 + jmax + m),m)]) #First row of array of eigenvalues is filled with the m values
        
        H = StarkH(m, jmax, A, B, C, pa, pb, pc, 0.0) #Make the Hamiltonian block
        
        eigs = linalg.eigvalsh(H) #Diagonalize the block; eigenvalues for the current value of m and electric field
        
        eigsm = np.append( eigsm , array([eigs]) , axis=0) #Put in the eigenvalues for this field strength and m
    
        eigslength = min( nmax , (jmax - m + 1)*(jmax + m + 1) ) #Number of eigenvalues to take from eigsm for each field strength

        eigsSmall[:,nmax:(nmax+eigslength)] = eigsm[:,:eigslength] #Append the lowest eigenvalues for m to the list of lowest eigenvalues
    
        eigsSmall = sortEigs(eigsSmall) #Sort the eigenvalues  
             
    eigsSmall = eigsSmall[:,np.argsort(eigsSmall[0,:nmax])] #Sort in order of increasing m

    lowestms, mindx, mcounts = np.unique(eigsSmall[0,:nmax],return_counts=True,return_index=True) #The values and counts of m with the lowest energies at zero field

    for index, count in zip(mindx,mcounts): #For each m sort the energies in increasing order
        eigsSmall[1,index:index+count] = np.sort(eigsSmall[1,index:index+count])
      
    print( eigsSmall )
    #Follow eigenvalues as a function of field strength
    for step in range(1,esteps+1):
    
        frac = 0.9 + 0.2*step / float(esteps-1) #Fraction of electric field at this step
    
        eigsstep = np.array([])
    
        for m, count in zip(lowestms,mcounts):
        
            H = StarkH(int(m), jmax, A, B, C, pa, pb, pc, frac*E0) #Make the Hamiltonian block
        
            lowesteig = linalg.eigvalsh(H)[:count] #Diagonalize the block and collect lowest mcounts eigenvalues for the current value of m and electric field
        
            eigsstep = np.append(eigsstep, lowesteig) #Put the eigenvalues in the list
    
        eigsSmall = np.append(eigsSmall,np.array([eigsstep]), axis=0)

    eigsFile = open(str(Path(__file__).resolve().parent)+'\\'+molName+'_eigen.csv','w',newline='',)

    eigsWriter = csv.writer(eigsFile)

    eigsWriter.writerow( np.array([molName,p0,esteps,jmax,nmax])) #Write some info about molecule
    eigsWriter.writerow( np.concatenate( [ [''], eigsSmall[0] ]) ) #Write the values of m

    for step in range(esteps+1):
        frac = 0.9 + 0.2*step / float(esteps-1)
        eigsWriter.writerow( np.concatenate( [ [frac*E0], eigsSmall[step+1] ]) ) #Write the eigenvalues as a function of electric field

    eigsFile.close()
    
def PEffC(effective_dipole, plate_voltage, He_temp): #UNTESTED
    p0 = effective_dipole
    volts = plate_voltage
    T = He_temp

    'Calculate thermally averaged dipole moment, assuming classical mechanics'
    E0 = volts*(411413.0)       # Electric field at center position
    kT = 1.38e-23*T #Boltzmann constant times the temperature
    
    x = p0*E0/kT
    
    return p0*(1/tanh(x) - 1/x)
    

def PEffQ(molName,DipoleMoment,volts,dopantTemp):#TESTED, seems to work
    'Calculate the thermally averaged dipole moment, based on the Stark curve data'
    p0 = DipoleMoment*3.336e-30
    T = dopantTemp
    E0 = volts*(411413.0)
    kT = 1.38e-23*T #Boltzmann constant times the temperature

    #directory = 'G:/.shortcut-targets-by-id/0B9Ekf3SxBCyDamhIV3l1a1VXbnM/Notes/Computational Results/Simulated Deflection Profile III/Molecules/'
    
    eigsFile = open(str(Path(__file__).resolve().parent)+'\\'+molName+'_eigen.csv','r')
    eigsDat = csv.reader(eigsFile)
    
    # Get some info about the data file
    rowCount = sum(1 for row in eigsDat) #Number of rows in the data file, including header info and M values
    eigsFile.seek(0) #The above line goes to end of file; Go to the beginning of file
    eigsInfo = next(eigsDat) #Basic information about the file
    ms = map(float,next(eigsDat)[1:]) #Values of the quantum number M
    colCount = len(next(eigsDat)) #Number of columns in the data file, inlcuding the field value
    
    #Go to start of Stark curve data
    eigsFile.seek(0) 
    next(eigsDat)
    next(eigsDat)
    
    field = np.zeros(rowCount - 2)              #Array of electric field values
    e = np.zeros((colCount-1,rowCount-2)) #Table of Stark energies
    
    #Store data to variables 'field' and 'energies'
    for i in range(len(field)):
        
        data = next(eigsDat)

            
        field[i] = float(data[0])
        e[0:,i] = [float(x) for x in data[1:]]
    
    eigsFile.close()
    
    eFunc = interp1d(field, e)
        
    e = eFunc(E0) #Stark energies at the given electric field
    
    #Average dipole moments at the given electric field
    p = -( eFunc(E0 + 0.001*E0) - eFunc(E0 - 0.001*E0) ) / (0.002*E0)
    
    return (sum( p*np.exp(-e/kT ) ) / sum( np.exp(-e/kT) ))/p0

### Convert (j,k,m) into a 1D array index. Used in the StarkH function. ###
def ind(j,k,m):
    'Index for the Hamiltonian block, for quantum numbers j, k and m'
    return int( j*j-m*m+j+k )


### Sort array of eigenvalues ###
def sortEigs(eigs):
    'Sorts columns of eigenvalues in ascending order according to the minimum value of each column'
    #Each column starts with the associated quantum number m, which is ignored in the sorting process
    
    minEigs = array([]) #Will contain the minimum eigenvalue in each column
    
    for col in range(len(eigs[0])):
        minEigs = np.append(minEigs, np.min(eigs[1:,col]) ) #Exclude the first row, which contains the values of m
    
    colOrder = np.argsort(minEigs) #Returns sorted order for columns
    
    eigsSorted = eigs[:,colOrder] #Sort the columns
    
    return eigsSorted

def pInd(NDrop, pEnc, Eext, molR):
    
    a = molR    #Inner radius of droplet
    b = pow( (2.2e-9)**3*float(NDrop) / 1.0e3 + a**3 , 1.0/3) 
    a3b3 = a*a*a/(b*b*b)
    pz = 2*(1.0 - a3b3)*( 2*pi*eps0*He_susceptibility*(2*He_susceptibility+3)*pow(b,3.0)*Eext - He_susceptibility*He_susceptibility*pEnc )/( (He_susceptibility + 3)*(2*He_susceptibility+3) - 2*a3b3*He_susceptibility*He_susceptibility )
	
    return pz
