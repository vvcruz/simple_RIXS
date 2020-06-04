import numpy as np
import os
from os import chdir as cd
from io import StringIO

#
# RIXS cross-section based on a RHF/CIS or RKS/TDDFT
# from orca calculations
# 
# Berlin, 27th of February of 2020

def read_orca_params(fnam):
    params=[None]*7
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
        elif('Number of roots to be determined' in lines[i]):
            params[6] = int(lines[i].strip().split()[7])
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
    #orca_2mkl='/home/vinicius/software/Orca_4.0.1/orca_4_0_1_2_linux_x86-64_openmpi202/orca_2mkl'
    orca_2mkl='/home/vinicius/software/Orca_4.1.2/orca_4_1_2_linux_x86-64_shared_openmpi215/orca_2mkl'
    os.system(orca_2mkl+" "+fnam+" -molden")
    return

# generates input for multiwfn
def gen_dip_input_tddft(fnam):
    f=open('dip.inp','w')
    print("18\n"        # Electron excitation analysis
          "5\n"+        # Calculate transition electric dipole moments between all states and electric dipole moment of each state 
          fnam+".out"+"\n"         # Please input path of Gaussian/ORCA output file or plain text file, electron excitation information will be loaded from this file
          "2\n"         # Output transition dipole moments to transdipmom.txt in current folder
          "0\n"
          "-10\n"
          ,file=f)
    return

# use multiwfn to compute the transition dipole moments
# within a one-electron approximation
# The integrals have been outputted to orbint.txt in current folder
# The first and the second columns correspond to orbital indices,
# the next three columns correspond to the integral in X/Y/Z (a.u.), the final column is the norm
def compute_dipoles_tddft(fnam):
    multiwfn="/home/vinicius/software/multiwfn/Multiwfn_3.6_dev_bin_Linux/Multiwfn"
    os.system(multiwfn+" "+fnam+".molden.input"+" < dip.inp > dip.out 2>&1")
    os.system("mv transdipmom.txt "+fnam+"_transdipmom.txt")    
    return

# # read the transition dipole moments (x,y,z)
# # returns a matrix with dimensions dip[3,norb,norb] the first dimension being the xyz components
# def read_dipoles_energy_tddft(n_states,fnam,debug=False):
#     f=open(fnam+"_transdipmom.txt",'r')
#     content = f.readlines()
#     dip=np.zeros((3,n_states,n_states))
#     dip[:,0,0]=np.genfromtxt(StringIO(content[0]),usecols=(6,7,8),dtype=float)
#     dip[:,0,1:]=np.transpose(np.genfromtxt(StringIO(''.join(content[4:n_states+4-1])),dtype=float,usecols=(2,3,4)))
#     #print(dip[:,0,0])
#     #print(dip[:,0,:])
#     #
#     n_l=(n_states+4-1) + 5
#     for i in range(1):
# #        for j in range(i,n_states):
#         j=n_states - i -1
#         print(i,j)
#         dip[:,i,i:j]=np.transpose(np.genfromtxt(StringIO(''.join(content[n_l:n_l + j])),dtype=float,usecols=(2,3,4)))
#         print(dip[:,i,i:j])

#         n_l+=j
#     # dip[:,0,n_states]=np.genfromtxt(fnam+"_transdipmom.txt",usecols=(2,3,4),dtype=float,skip_header=4,skip_footer)
#     # print(dip.shape)
#     # dip=dip.reshape(3,norb,norb)
#     # print(dip.shape)
#     # print(dip[0,0,0],dip[0,1,0],dip[0,0,1])
#     # print(dip[1,0,0],dip[1,1,0],dip[1,0,1])

#     # if(debug):
#     #     f=open('check_dipoles.dat','w')
#     #     for i in range(norb):
#     #         for j in range(norb):
#     #             print(i,j,' '.join(str(x) for x in dip[:,i,j]),file=f)
#     return dip

# read the transition dipole moments (x,y,z)
# returns a matrix with dimensions dip[3,norb,norb] the first dimension being the xyz components
def read_dipoles_energy_tddft(n_states,fnam,debug=False):
    #initialize dictionaries for transition dipoles and energies
    dip={}
    tr_en={}
    for i in range(n_states):
        dip[i]={}
        tr_en[i]={}
        
    # read file into variable
    f=open(fnam+"_transdipmom.txt",'r')
    content = f.readlines()

    #start reading data: <0|r|0>
    #----------------------------------------
    l=0 # line counter
    i=0;j=0
    dip[i][j]=np.genfromtxt(StringIO(content[0]),usecols=(6,7,8),dtype=float)
    tr_en[i][j]=0.0e0
    #-----------------------------------------

    # read <0|r|j>
    #----------------------------------------
    l=3
    i=0
    for j in range(1,n_states):
        l+=1
        if(debug):
            print(content[l].strip())
        st_i=int(content[l].strip().split()[0])
        st_j=int(content[l].strip().split()[1])
        if(st_i==i and st_j==j):
            dip[i][j]=np.transpose(np.genfromtxt(StringIO(''.join(content[l])),dtype=float,usecols=(2,3,4)))
            tr_en[i][j]=np.transpose(np.genfromtxt(StringIO(''.join(content[l])),dtype=float,usecols=(5)))
        else:
            print('error reading file')
            return None
    #----------------------------------------
    # print(" <<>>")
    # read <i|r|j>
    #----------------------------------------
    l=l+4
    for i in range(1,n_states):
        for j in range(i,n_states):
            l+=1
            st_i=int(content[l].strip().split()[0])
            st_j=int(content[l].strip().split()[1])
            if(debug):
                print("(",i,j,")","(",st_i,st_j,")",content[l].strip())
            if(st_i==i and st_j==j):
                dip[i][j]=np.transpose(np.genfromtxt(StringIO(''.join(content[l])),dtype=float,usecols=(2,3,4)))
                tr_en[i][j]=np.transpose(np.genfromtxt(StringIO(''.join(content[l])),dtype=float,usecols=(5)))
            else:
                print('error reading file')
                return None
    return dip,tr_en

def lorentz(x,y):
    return  y/(np.pi*(x**2 + y**2))

def gauss(x,y):
    z = np.exp(-(4.0 * np.log(2) * x**2)/y**2)
    return z

def line_shape(x,y,tp):
    if(tp=='lorentz'):
        z = lorentz(x,y)
    elif(tp=='gauss'):
        z = gauss(x,y)
    return z

def xas_cross_section(om,dip,tr_e,psi_0,psi_i,gamma_f,tp='gauss'):
    sig=np.zeros_like(om)
    for zero in psi_0:
        for i in psi_i:
            y_int = (dip[zero][i][0])**2 +  (dip[zero][i][1])**2 +  (dip[zero][i][2])**2
            #sig[:] = sig[:] +  y_int * lorentz(om - tr_e[zero][i],gamma_f)
            sig[:] = sig[:] +  y_int * line_shape(om - tr_e[zero][i],gamma_f,tp)
            if(y_int > 5e-7):
                print('0 = ',zero,'i=',i,'ei0=',tr_e[zero][i],'intensity = ',y_int)
    return sig


def sig_tensor(om,eloss,zero,f,alpha,beta,gamma,delta,psi_i,dip,tr_e,gamma_c,gamma_f,tp='gauss'):
    y=np.zeros_like(eloss)
    for i in psi_i:
        y_int = dip[f][i][alpha] * dip[f][i][beta] * dip[zero][i][gamma] * dip[zero][i][delta]
        y_int = y_int/((om - tr_e[zero][i])**2 + gamma_c**2)
        #print(y,tr_e[i,f])
        if(y_int > 5e-5):
            print('0 = ',zero,'f =',f,'i=',i,'ef0=',tr_e[zero][f],'intensity = ','%5.3e'%y_int)
        #y[:] = y[:] +  y_int*lorentz(tr_e[zero][f] - eloss,gamma_f)
        y[:] = y[:] +  y_int*line_shape(tr_e[zero][f] - eloss,gamma_f,tp)
    return y 

def rixs_cross_section(om,eloss,theta,dip,tr_e,zero,psi_i,psi_f,gamma_c,gamma_f):#rixs_cross_section(om,eloss,theta,dip,tr_e,norb,nel,orb0,n_excite,n_decay,gamma_c,gamma_f):
    sig=np.zeros_like(eloss)
    coeff1=2.0e0 * (2.0e0 - 0.5e0*(1.0e0 - np.cos(np.radians(theta))**2))
    coeff2=3.0e0*0.5e0*(1.0e0 - np.cos(np.radians(theta))**2) -1.0e0 

    for f in psi_f:
        for alpha in range(3):
            for beta in range(3):
                sig[:] =\
                         sig[:] + coeff1 * sig_tensor(om,eloss,zero,f,alpha,alpha,beta,beta,psi_i,dip,tr_e,gamma_c,gamma_f)\
                         + coeff2 * sig_tensor(om,eloss,zero,f,alpha,beta,alpha,beta,psi_i,dip,tr_e,gamma_c,gamma_f)\
                         + coeff2 * sig_tensor(om,eloss,zero,f,alpha,beta,beta,alpha,psi_i,dip,tr_e,gamma_c,gamma_f)
    return sig

def dump_data(fnam,x,y):
    f=open(fnam,'w')
    for i in range(x.size):
        print('% 5.3f'%x[i],'% 9.10e'%y[i],file=f)
    f.close()
    return

def dump_map(fnam,x,y,z):
    f=open(fnam,'w')
    for i in range(x.size):
        for j in range(y.size):
            print('% 5.3f'%x[i],'% 5.3f'%y[j],'% 9.10e'%z[i,j],file=f)
        print('',file=f)
    f.close()
    return

# compute the electronic RIXS cross-section
# from the transition dipole moment matrix
# the matrix is given as <i| mu_\alpha |j > over all orbitals
def compute_rixs(fnam,om,eloss_range,gamma_c,psi_0,psi_i,psi_f,gamma_f=0.1,theta=90.0e0,neloss=1024,do_xas=True):
    print('single-electron RIXS')
    print('by Vinicius Vaz da Cruz')
    print('\n-------------\n\n')


    print('using orca calculation')
    print('base file name: ',fnam,'\n\n')
    
    params=read_orca_params(fnam+'.out')
    nel=params[3];norb=params[5];n_states=params[6]+1
    print('Calculation parameters')
    print('Number of atoms:',params[0])
    print('Total Charge:',params[1])
    print('Multiplicity:',params[2])
    print('Number of Electrons:',nel)
    print('Total Energy       : ',params[4])
    #print('Number of contracted basis functions    : ',norb)

    print('\n-------------\n')
    
    print('RIXS parameters')
    print('excitation energies = ',om,' eV')
    print('desired energy loss range = ',eloss_range[0],eloss_range[1],' eV')
    print('Total number of CIS/TDDFT states   = ',n_states)
    print('initial state index = ',psi_0)
    print('intermediate states indexes =',psi_i)
    print('final state indexes  = ',psi_f)
    
         
    print('\n-------------\n')

    # print('Reading orbital energies and generating transition energies')
    # # e is a 1D arrauy with n=params[5] entries
    # e,tr_e=read_orca_orb_en(fnam+'.out',norb,debug=True)
    # print('done!')

    print('\n-------------\n')

    if(not os.path.isfile(fnam+"_transdipmom.txt")):
        print('Computing transition dipole moments with MultiWfn')
        gen_molden_orca(fnam)
        gen_dip_input_tddft(fnam)
        compute_dipoles_tddft(fnam)
    else:
        print('Previously computed transition dipole moments found!')
        print('Reading data from '+fnam+'_transdipmom.txt')
    dip,tr_e=read_dipoles_energy_tddft(n_states,fnam) 


    if(do_xas):
        print('Computing XAS')
        om_xas=np.linspace(tr_e[psi_0[0]][psi_i[0]]-2.0,tr_e[psi_0[0]][psi_i[-1]] + 2.0,neloss)
        sig_xas=xas_cross_section(om_xas,dip,tr_e,psi_0,psi_i,gamma_f)
        dump_data(fnam+'_xas.dat',om_xas,sig_xas)
        print('saved to '+fnam+'_xas.dat \n')

    print('computing cross-section')
    eloss=np.linspace(eloss_range[0],eloss_range[1],neloss)
    #sig=np.zeros_like(eloss)
    sig=np.zeros((om.size,eloss.size),dtype=float)
    for i in range(om.size):
        print('computing excitation energy ',om[i],'eV')
        for zero in psi_0:
            #sig_orb=rixs_cross_section(om[i],eloss,theta,dip,tr_e,norb,nel,orb,n_excite,n_decay,gamma_c,gamma_f)
            sig_zero=rixs_cross_section(om[i],eloss,theta,dip,tr_e,zero,psi_i,psi_f,gamma_c,gamma_f)
            sig[i,:] = sig[i,:] + sig_zero
            dump_data(fnam+'_rixs_'+str(om[i])+'_'+str(theta)+'.dat',eloss,sig[i,:])
        print('\n ---------------------------\n')
    dump_map(fnam+'_rixs_map_'+str(theta)+'.dat',om,eloss,sig)
    print('done!\n')
    return




