import numpy as np
import os
from os import chdir as cd

#
# RIXS cross-section within an orbital approximation
# from orca calculations
# 
# Berlin, 24th of May of 2019

def read_orca_params(fnam):
    params=[None]*6
    lines=open(fnam).readlines()
    for i in range(len(lines)):
        if('Number of atoms' in lines[i] and params[1] == None):
            params[0] = int(lines[i].strip().split()[-1])
        elif('Total Charge' in lines[i] and params[2] == None):
            params[1] = int(lines[i].strip().split()[-1])
        elif('Multiplicity' in lines[i] and params[3] == None):
            params[2] = int(lines[i].strip().split()[-1])
        elif('Number of Electrons' in lines[i] and params[4] == None):
            params[3] = int(lines[i].strip().split()[-1])
        elif('Total Energy       : ' in lines[i]):
            params[4] = float(lines[i].strip().split()[3])
        elif('# of contracted basis functions' in lines[i]):
            params[5] = int(lines[i].strip().split()[6])
    return params

# check if file exists
def chk_file(fnam,msg):
    if(os.path.isfile(fnam)):
        print(msg+' file found')
    else:
        print(msg+' file not found')
        exit()
    return

# generate molden file from an orca calculation
def gen_molden_orca(fnam):
    #orca_2mkl='/home/vinicius/software/Orca_4.1.2/orca_4_1_2_linux_x86-64_shared_openmpi215/orca_2mkl'
    orca_2mkl='/home/vinicius/software/Orca_4.0.1/orca_4_0_1_2_linux_x86-64_openmpi202/orca_2mkl'
    os.system(orca_2mkl+" "+fnam+" -molden")
    return

# generates input for multiwfn
def gen_dip_input():
    f=open('dip.inp','w')
    print("200\n"       # other options
          "10\n"        # output various kinds of integral between orbitals
          "1\n"         # Electric dipole moment
          "3\n"         # Between all orbitals
          "0\n"
          "-10\n"
          ,file=f)
    return

# use multiwfn to compute the transition dipole moments
# within a one-electron approximation
# The integrals have been outputted to orbint.txt in current folder
# The first and the second columns correspond to orbital indices,
# the next three columns correspond to the integral in X/Y/Z (a.u.), the final column is the norm
def compute_dipoles(fnam):
    multiwfn="/home/vinicius/software/multiwfn/Multiwfn_3.6_dev_bin_Linux/Multiwfn"
    os.system(multiwfn+" "+fnam+".molden.input"+" < dip.inp > dip.out 2>&1")
    os.system("mv orbint.txt "+fnam+"_orbint.txt")    
    return

# read the transition dipole moments (x,y,z)
# returns a matrix with dimensions dip[3,norb,norb] the first dimension being the xyz components
def read_dipoles(norb,fnam,debug=False):
    dip=np.genfromtxt(fnam+'_orbint.txt',usecols=(2,3,4),dtype=float,unpack=True)
    print(dip.shape)
    dip=dip.reshape(3,norb,norb)
    print(dip.shape)
    print(dip[0,0,0],dip[0,1,0],dip[0,0,1])
    print(dip[1,0,0],dip[1,1,0],dip[1,0,1])

    if(debug):
        f=open('check_dipoles.dat','w')
        for i in range(norb):
            for j in range(norb):
                print(i,j,' '.join(str(x) for x in dip[:,i,j]),file=f)
    return dip

# reads the orbital energies
# returns the orbital energies e[norb] and the transition energy matrix tr_e[norb,norb]
def read_orca_orb_en(fnam,norb,debug=False):
    e=np.zeros(norb,dtype=float)
    tr_e=np.zeros((norb,norb),dtype=float)
    lines=open(fnam).readlines()
    for i in range(len(lines)):
        if('ORBITAL ENERGIES' in lines[i]):
            i=i+4
            for k in range(norb):
                e[k]=float(lines[i].strip().split()[3])
                i=i+1
                
    for i in range(norb):
        for j in range(norb):
            tr_e[i,j] = e[i] - e[j]

    if(debug):
        f=open('check_tr_energies.dat','w')
        for i in range(norb):
            for j in range(norb):
                print('%6.3f'%tr_e[i,j],end=' ',file=f)
            print('\n',file=f)
            
    return e,tr_e

def lorentz(x,y):
    return  y/(np.pi*(x**2 + y**2))

def sig_tensor(om,eloss,orb0,f,alpha,beta,gamma,delta,nel,n_excite,dip,tr_e,gamma_c,gamma_f):
    norb_occ=int(nel/2)
    y=np.zeros_like(eloss)
    for i in range(norb_occ,norb_occ+n_excite):
        y_int = dip[alpha,orb0,f] * dip[beta,f,orb0] * dip[gamma,orb0,i] * dip[delta,i,orb0]
        y_int = y_int/((om - tr_e[i,orb0])**2 + gamma_c**2)
        #print(y,tr_e[i,f])
        if(y_int > 5e-7):
            print('f =',f,'i=',i,'ef0=',tr_e[i,f],'intensity = ',y_int)
        y[:] = y[:] +  y_int*lorentz(tr_e[i,f] - eloss,gamma_f)
    return y 

def rixs_cross_section(om,eloss,theta,dip,tr_e,norb,nel,orb0,n_excite,n_decay,gamma_c,gamma_f):
    sig=np.zeros_like(eloss)
    norb_occ=int(nel/2)
    coeff1=2.0e0 * (2.0e0 - 0.5e0*(1.0e0 - np.cos(np.radians(theta))**2))
    coeff2=3.0e0*0.5e0*(1.0e0 - np.cos(np.radians(theta))**2) -1.0e0 


    for f in range(norb_occ - n_decay,norb_occ):
        for alpha in range(3):
            for beta in range(3):
                sig[:] =\
                sig[:] + coeff1 * sig_tensor(om,eloss,orb0,f,alpha,alpha,beta,beta,nel,n_excite,dip,tr_e,gamma_c,gamma_f)\
                + coeff2 * sig_tensor(om,eloss,orb0,f,alpha,beta,alpha,beta,nel,n_excite,dip,tr_e,gamma_c,gamma_f)\
                + coeff2 * sig_tensor(om,eloss,orb0,f,alpha,beta,beta,alpha,nel,n_excite,dip,tr_e,gamma_c,gamma_f)
    return sig

def dump_data(fnam,x,y):
    f=open(fnam,'w')
    for i in range(x.size):
        print('% 5.3f'%x[i],'% 9.10e'%y[i],file=f)
    f.close()
    return

# compute the electronic RIXS cross-section
# from the transition dipole moment matrix
# the matrix is given as <i| mu_\alpha |j > over all orbitals
def compute_rixs(fnam,om,eloss_range,gamma_c,gamma_f=0.1,orb0=0,n_excite=1,n_decay=1,theta=90.0e0,neloss=1024):
    print('single-electron RIXS')
    print('by Vinicius Vaz da Cruz')
    print('\n-------------\n\n')


    print('using orca calculation')
    print('base file name: ',fnam,'\n\n')
    
    params=read_orca_params(fnam+'.out')
    nel=params[3];norb=params[5]
    print('Calculation parameters')
    print('Number of atoms:',params[0])
    print('Total Charge:',params[1])
    print('Multiplicity:',params[2])
    print('Number of Electrons:',nel)
    print('Total Energy       : ',params[4])
    print('Number of contracted basis functions    : ',norb)

    print('\n-------------\n')
    
    print('RIXS parameters')
    print('excitation energy = ',om,' eV')
    print('desired energy loss range = ',eloss_range[0],eloss_range[1],' eV')
    print('idex of orbital to excite from= ',orb0)
    print('number of unnocupioed orbitals to excite to = ',n_excite)
    print('number of occupied orbitals to decay from   = ',n_decay)
         
    print('\n-------------\n')

    print('Reading orbital energies and generating transition energies')
    # e is a 1D arrauy with n=params[5] entries
    e,tr_e=read_orca_orb_en(fnam+'.out',norb,debug=True)
    print('done!')

    print('\n-------------\n')

    if(not os.path.isfile(fnam+"_orbint.txt")):
        print('Computing transition dipole moments with MultiWfn')
        gen_molden_orca(fnam)
        gen_dip_input()
        compute_dipoles(fnam)
    else:
        print('Previously computed transition dipole moments found!')
        print('Reading data from '+fnam+'_orbint.txt')
    dip=read_dipoles(norb,fnam)
        

    print('computing cross-section')
    eloss=np.linspace(eloss_range[0],eloss_range[1],neloss)
    sig=rixs_cross_section(om,eloss,theta,dip,tr_e,norb,nel,orb0,n_excite,n_decay,gamma_c,gamma_f)
    dump_data('rixs_test'+str(om)+'_'+str(theta)+'.dat',eloss,sig)
    print('done!\n')
    return




