import numpy as np
import pandas as pd
from numba import njit

@njit
def update_astrocyte_dynamics(dt, identifier, gamma, parents, DiffNa, DiffK, DiffCa, g_l, distance_from_soma, radius, area, area_cross, volume, Cm, stim_start_nt, stim_end_nt, stim_dur, glut_stim, pot_stim, stim_comp_glut, stim_comp_pot, n_stim, n_stimulated, N, last, iskip, branch_max_dict, branch_to_comps):
    t_in = 20.0

    # # Overwrite for testing/identifying only the base, and the peak if needed
    # last = 300000000 
    # iskip = 1000

    time_array = np.zeros((last-int(t_in/dt))//iskip)

    I_pump = np.zeros(N) #JNaKk
    I_pumps = np.zeros((N, (last-int(t_in/dt))//iskip))     
    I_NBC = np.zeros(N)
    I_NBCs = np.zeros((N, (last-int(t_in/dt))//iskip)) 
    I_NHE = np.zeros(N)
    I_NHEs = np.zeros((N, (last-int(t_in/dt))//iskip)) 
    I_NKCC1 = np.zeros(N)
    I_NKCC1s = np.zeros((N, (last-int(t_in/dt))//iskip)) 
    I_NCX = np.zeros(N)
    I_NCXs = np.zeros((N, (last-int(t_in/dt))//iskip)) 

    I_pump_alpha2beta1 = np.zeros(N)
    I_pump_alpha2beta2 = np.zeros(N)
    I_pumps_alpha2beta1 = np.zeros((N, (last-int(t_in/dt))//iskip)) 
    I_pumps_alpha2beta2 = np.zeros((N, (last-int(t_in/dt))//iskip)) 

    JGluT = np.zeros(N)
    JGluTs = np.zeros((N, (last-int(t_in/dt))//iskip))

    I_K_bath_to_Ko = np.zeros(N)
    v = np.ones(N) * -83.79    # Membrane potential of the astrocyt(Ac) mV s92
    vs = np.zeros((N, (last-int(t_in/dt))//iskip))    
    Ki = np.ones(N) * 146000.0    # K+ concentration in the astrocyte uM s84
    Kis = np.zeros((N, (last-int(t_in/dt))//iskip))
    K_stim = np.ones(N)
    K_bath_BL = 2900.0
    K_bath = np.ones(N) * K_bath_BL
    Nai = np.ones(N) * 12000.0   # Na+ concentration in the astrocytes(uM) s85
    # Nak_BL_threshold = 10 #um
    # Nak[distance_from_soma > Nak_BL_threshold] = 15000
    Nais = np.zeros((N, (last-int(t_in/dt))//iskip))
    HCO3i = np.ones(N) * 25000.0  # HCO3-concetration in the astrocytes(uM): s86
    Cai = np.ones(N) * 0.082    # The astrocytic cytosolic Ca2+ concentation(uM) s88
    Cais = np.zeros((N, (last-int(t_in/dt))//iskip))
    Cli = np.ones(N) * 8215.0     # Cl- concentration in the astrocyts via electroneutrality(uM) s87
    Clis = np.zeros((N, (last-int(t_in/dt))//iskip))
    Ca_ER = np.ones(N) * 480.80     # The ca2+ concentation in the astrocytic endoplasmic reciculum(ER)
    h_ip3 = np.ones(N) * 0.4086     # The inactivation variable of the astrocytic IP3R channel(-) s94
    Ip3_i = np.ones(N) * 0.0483    # The astrocytic inositol triphophate(IP3) concentration(uM):s89 0.0483
    Ip3_is = np.zeros((N, (last-int(t_in/dt))//iskip))
    I_IP3 = np.ones(N)
    I_Na_leak = np.ones(N)
    I_Na_leaks = np.zeros((N, (last-int(t_in/dt))//iskip)) 
    I_TRPV = np.ones(N)
    I_TRPVs = np.zeros((N, (last-int(t_in/dt))//iskip))
    m = np.ones(N) * 0.5635     # The open probability of the transient receptor potential(-)
    
    Ko = np.ones(N) * 2900.0    # K+ concentration in the synaptic cleft(sc)(uM):s81
    Kos = np.zeros((N, (last-int(t_in/dt))//iskip))
    Nao = np.ones(N) * 152000.0   # Na+ concentration in the SC(uM):s82
    Naos = np.zeros((N, (last-int(t_in/dt))//iskip))
    HCO3o = np.ones(N) * 26000.0  # HCO3-concentration in the SC(UM):s83  
    pho = np.ones(N) * 7.35
    phos = np.zeros((N, (last-int(t_in/dt))//iskip))
    phi = np.ones(N) * 7.33
    phis = np.zeros((N, (last-int(t_in/dt))//iskip))
    Ho = np.zeros(N)
    Hi = np.zeros(N)
    bi = np.ones(N)
    gsyn = np.ones(N)
    gsynmax = np.zeros(N)
    Isyn = np.zeros(N)
    IdiffNas = np.zeros((N, (last-int(t_in/dt))//iskip))  
    IdiffKs = np.zeros((N, (last-int(t_in/dt))//iskip)) 
    IdiffCas = np.zeros((N, (last-int(t_in/dt))//iskip)) 
    Icoups = np.zeros((N, (last-int(t_in/dt))//iskip)) 
    Glu = np.ones(N) * 1.5
    # Glus = np.zeros((N, (last-int(t_in/dt))//iskip)) 
    j_10 = 0
    Istim = np.zeros(N)
    time_t = 0.0

    
    # fac_isoform = list(np.arange(1,0.7, -0.05))   
    pka = 6.1
    P_co2 = 5332.9 #Pa
    s = 2.25e-4 * 1000 #uM Pa^-1
    CO2_aq = s * P_co2   #uM 
    k_h = 800 * 0.001 #nmol L^-1 * 0.001 = uM 

    Cap = 1846.40  # Perivascular Calcium concentration uM external     

    # Flux conversion from pA/um^2 or A/m^2 to uM/s
    F = 96485.0  # c/mol
      
    # For constant volume
    pi = 3.14159265358979     
    # vol_a = 168 * 1e-18 #50.0e-18  # m^3    # 168 um^3 is the mode volume
    # Aa = 4.0 * pi * (3.0 * vol_a / (4.0 * pi))**(2.0 / 3.0)  # m^2    
    # convg = Aa / (F * vol_a) * 1.0e3  # 1.0e3 convert pA/um^2 to uM/sec 

    R = 22.92e-6  # Vessel radius m s213
    cao = 2000.0  # Extracellular calcium concentration
    R0passivek = 20.0e-6  # Vessel radius when passive and no stress is applied m

    VRsa = 3.0

    phii = 26.69950  # RgT/F mV
    conv = 1970.0  # 7.350197546d0 !1970.0d0 ! Change in membrane potential by scaling factor mVuM^-1
    # Ionic valence for K+,Na+,cl-,Ca2+
    zK = 1.0
    zNa = 1.0
    zCl = -1.0
    zCa = 2.0
    zNBC = -1.0  # Effective valence for NBC cotransporter complex

    # max_peak_value_perc = list(np.arange(0.8, 1.5, 0.1)) 
    I_pump_max = 2.3667e4  # Maximum flux through the Na+/K+ ATP-ase pump uMs^-1 2.3667e4 
    # Conductances- now in uMmV^-1s^-1 so consistent with other fluxes
    # original conductances in mhom^-2 are commented
    # converted by dividing by R_K=6d-8 and F=9.65d4

    g_k = 6907.770
    g_Na = 226.940 #113.470 #226.940 (Org) #(11 mM normal ,15mM (* 1.4) for alpha2beta1) (22 mM normal, 15mM (* 0.8) for alpha2beta2) *0.6

    g_NBC = 130.740
    g_KCC1 = 1.7280
    g_NKCC1 = 9.5680
    g_Cl = 151.930
    g_TRPV = 3.17e-4

    I_IP3_max = 2880.0  # Maximum rate of Ca2+ through the IP3 mediated channel
    KI =  0.030  # Disassociation constant for IP3 binding to an IP3R
    Kact = 0.170  # Disassociation constant for Ca2+ binding to an activation site on IP3R
    # original constants are in m^-2, converted to uM^-1
    PL = 0.08040  # ER leak channel steady state balance constant uMs^-1
    Vmax = 20.0  # Maximum rate of Ca2+ uptake pump on the ER uMs^-1
    kpump = 0.240  # Ca2+ uptake pump dissociation constant uM

    rhomin = 0.10  # Minimum ratio of bound to unbound IP3 receptors
    rhomax = 0.70  # Maximum ratio of bound to unbound IP3 receptors

    Glumax = 1846.0  # Maximum glutamate concentration(one vesicle) uM

    Bkend = 40.0  # Ratio of endogenous buffer concentration to dissociation constant
    Kex = 0.260  # Dissociation constant of exogenous buffer uM
    Bex = 11.350  # Concentration of exogenous buffer uM
    delta = 1.2350e-2  # Ratio of the activities of the unbound and bound receptors
    KG = 8.820  # G-protein dissociation constant

    gamCaek = 200.0  # Ca2+ concentration constant uM
    gamCaik = 0.010  # Ca2+ concentration constant uM
    epshalfk = 0.10  # strain required for half activation of the TRPV4 channel
    Kappa = 0.10  # TRPVS channel constant
    v1TRPVk = 120.0  # TRPV4 channel voltage gating constant
    v2TRPVk = 13.0  # TRPV4 channel voltage gating constant
    rbuff = 0.050  # Rate of Ca2+ buffering astrocyte endfoot compared to the astrocytes body
    VRERcyt = 0.1850  # Volume ratio between ER and astrocytic cytosol
    kon = 2.0  # Rate of Ca2+ binding to the inhibitory site on the IP3R uMs^-1
    kinh = 0.1  # Dissociation constant of IP3R uM
    rh = 8.6 # Maximum rate of IP3 production in astrocytes due to glutamate receptors uMs^-1  4.80
    kdeg = 1.250  # Rate constant for IP3 degradation in astrocyte s^-1
    trpvswitch = 1.0
    tTRPVk = 0.90  # Characteristic time constant for mk


    # stim_comp_glut = np.arange(0,N) # to have global, decaying stimulation at each compartment. THis replaces the one that comes from the UI
    # glut_stim = np.ones(N) * 1000.00 # to have global, decaying stimulation at each compartment. THis replaces the one that comes from the UI

    stim_comp_pot = np.arange(0,N) # to have global, decaying stimulation at each compartment. THis replaces the one that comes from the UI
    pot_stim = np.ones(N) * 7000.00   # -2300 4100.00# to have global, decaying stimulation at each compartment. THis replaces the one that comes from the UI 4100
    # pot_stim_amp = 7000  # 7000 600    
    alpha =  0.8 # 0.04 for test 50 sec, o/w 0.08 
    beta =  1 # 0.8 for test 50 sec, o/w 0.01

    # # Overwrite for testing
    # stim_start_nt = 220000000   #220000000 #22nd second 1000000000
    # stim_end_nt = 230000000  #230000000 #22.1st second       100ms of stimulation  2200000000
    # alpha =  0.04 # 0.04 for test 50 sec, o/w 0.08 -0.8(low)0.8(low) 0.8(high)1(high) for 30 sec from 22 to 23
    # beta =  0.8 # 0.8 for test 50 sec, o/w 0.01


    # print(f"\n\n\nInitial DiffNa\n: {DiffNa}")
    ####################### To vary Na diff coeff and EAAT density along the distance ################################################################
    max_distance_array = np.zeros(N)
    IGluTmax = np.ones(N) * 0.10  # pA/um^2
    IGluTmax_highest = 0.30
    for k in range(N):
        for bname_idx, comps in branch_to_comps.items():
            if k in comps:
                max_distance_array[k] = branch_max_dict.get(bname_idx)
                break
    
    for k in range(N):
        if max_distance_array[k] > 0:
            dis_ratio = distance_from_soma[k] / max_distance_array[k]        
            # Diffusion scales with area (smaller area = slower diffusion)
            dist_ratio = np.exp(-k * dis_ratio) * (1.0 - dis_ratio)
            DiffNa[k] *= dist_ratio
            IGluTmax[k] = IGluTmax[0] + (IGluTmax_highest  - IGluTmax[0]) * dis_ratio  ### change EAAT transporter density as a function of distance 
            # print(f"dis ratio :{dist_ratio}")
    DiffCa = DiffCa * 0.0
    # print(f"\n\nIgutmax: {IGluTmax}")
    # print(f"Max Distance\n: {max_distance_array}")
    # print(f"Final DiffNa\n: {DiffNa}") 
    # print(branch_max_dict)   
    # sys.exit() 
    ###########################################################################################################################
    
    for ii in range(1, last+1):

        time_t = time_t + dt 

        Icoup = np.zeros(N)
        IdiffK = np.zeros(N)
        IdiffNa = np.zeros(N)
        IdiffCa = np.zeros(N)
        Na_stim = np.zeros(N) 
        if pot_stim is None and glut_stim is None:       
            for i2 in range(1, N):
                j2 = parents[i2]            
                Icoup[i2] = gamma[i2] * (v[j2-1] - v[i2])  ## uA/cm^2            
                Icoup[j2-1] += gamma[j2-1] * (v[i2] - v[j2-1])            
                IdiffK[i2] = DiffK[i2] * (Ko[j2-1] - Ko[i2])   # uM/s 
                IdiffK[j2-1] += DiffK[j2-1] * (Ko[i2] - Ko[j2-1]) 
                IdiffNa[i2] = DiffNa[i2] * (Nai[j2-1] - Nai[i2]) 
                IdiffNa[j2-1] += DiffNa[j2-1] * (Nai[i2] - Nai[j2-1]) 
                IdiffCa[i2] = DiffCa[i2] * (Cai[j2-1] - Cai[i2]) 
                IdiffCa[j2-1] += DiffCa[j2-1] * (Cai[i2] - Cai[j2-1])
        
        for k in range(0, N):
            if glut_stim is not None:
                if ii*dt < stim_start_nt*dt:
                    if N != 1 and k != 0:
                        j2 = parents[k]            
                        Icoup[k] = gamma[k] * (v[j2-1] - v[k])  ## uA/cm^2            
                        Icoup[j2-1] += gamma[j2-1] * (v[k] - v[j2-1])            
                        IdiffK[k] = DiffK[k] * (Ko[j2-1] - Ko[k])   # uM/s 
                        IdiffK[j2-1] += DiffK[j2-1] * (Ko[k] - Ko[j2-1]) 
                        IdiffNa[k] = DiffNa[k] * (Nai[j2-1] - Nai[k]) 
                        IdiffNa[j2-1] += DiffNa[j2-1] * (Nai[k] - Nai[j2-1]) 
                        IdiffCa[k] = DiffCa[k] * (Cai[j2-1] - Cai[k]) 
                        IdiffCa[j2-1] += DiffCa[j2-1] * (Cai[k] - Cai[j2-1]) 
                    else:
                        pass

                elif stim_start_nt*dt <= ii*dt <= stim_end_nt*dt:
                    # corresponding stimulation values
                    for i, comp in enumerate(stim_comp_glut):
                        if k == comp:
                            stim_value = glut_stim[i]
                            Glu[k] = 1.5 + stim_value *(1 - np.exp(-alpha * (ii * dt - stim_start_nt * dt))) 
                        if N != 1 and k != 0:                                                                       
                            j2 = parents[k]            
                            Icoup[k] = gamma[k] * (v[j2-1] - v[k])  ## uA/cm^2            
                            Icoup[j2-1] += gamma[j2-1] * (v[k] - v[j2-1])            
                            IdiffK[k] = DiffK[k] * (Ko[j2-1] - Ko[k])   # uM/s 
                            IdiffK[j2-1] += DiffK[j2-1] * (Ko[k] - Ko[j2-1]) 
                            IdiffNa[k] = DiffNa[k] * (Nai[j2-1] - Nai[k]) 
                            IdiffNa[j2-1] += DiffNa[j2-1] * (Nai[k] - Nai[j2-1]) 
                            IdiffCa[k] = DiffCa[k] * (Cai[j2-1] - Cai[k]) 
                            IdiffCa[j2-1] += DiffCa[j2-1] * (Cai[k] - Cai[j2-1])
                        else:
                            continue

                elif ii*dt > stim_end_nt*dt:
                    # with exponential decay after stimulation ends
                    for i, comp in enumerate(stim_comp_glut):
                        if k == comp:
                            stim_value = glut_stim[i]                            
                            Glu[k] = stim_value * np.exp(-beta * (ii * dt - stim_end_nt  * dt)) + 1.5 * (1 - np.exp(-beta * (ii * dt - stim_end_nt * dt))) #stim_value * np.exp(-(ii*dt - stim_end_nt*dt) / 0.5)
                        if N != 1 and k != 0:                                                                       
                            j2 = parents[k]            
                            Icoup[k] = gamma[k] * (v[j2-1] - v[k])  ## uA/cm^2            
                            Icoup[j2-1] += gamma[j2-1] * (v[k] - v[j2-1])            
                            IdiffK[k] = DiffK[k] * (Ko[j2-1] - Ko[k])   # uM/s 
                            IdiffK[j2-1] += DiffK[j2-1] * (Ko[k] - Ko[j2-1]) 
                            IdiffNa[k] = DiffNa[k] * (Nai[j2-1] - Nai[k]) 
                            IdiffNa[j2-1] += DiffNa[j2-1] * (Nai[k] - Nai[j2-1]) 
                            IdiffCa[k] = DiffCa[k] * (Cai[j2-1] - Cai[k]) 
                            IdiffCa[j2-1] += DiffCa[j2-1] * (Cai[k] - Cai[j2-1])
                        else:
                            continue

            if pot_stim is not None:
                if ii*dt < stim_start_nt*dt:
                    if N != 1 and k != 0:
                        j2 = parents[k]            
                        Icoup[k] = gamma[k] * (v[j2-1] - v[k])  ## uA/cm^2            
                        Icoup[j2-1] += gamma[j2-1] * (v[k] - v[j2-1])            
                        IdiffK[k] = DiffK[k] * (Ko[j2-1] - Ko[k])   # uM/s 
                        IdiffK[j2-1] += DiffK[j2-1] * (Ko[k] - Ko[j2-1]) 
                        IdiffNa[k] = DiffNa[k] * (Nai[j2-1] - Nai[k]) 
                        IdiffNa[j2-1] += DiffNa[j2-1] * (Nai[k] - Nai[j2-1]) 
                        IdiffCa[k] = DiffCa[k] * (Cai[j2-1] - Cai[k]) 
                        IdiffCa[j2-1] += DiffCa[j2-1] * (Cai[k] - Cai[j2-1]) 
                    else:
                        pass
                elif stim_start_nt*dt <= ii*dt <= stim_end_nt*dt:
                    # corresponding stimulation values
                    for i, comp in enumerate(stim_comp_pot):                        
                        if k == comp:
                            stim_value = pot_stim[i]
                            # Ks[k] = 2900 +  stim_value *(1 - np.exp(-alpha * (ii * dt - stim_start_nt * dt)))                            
                            K_stim[k] = stim_value *(1 - np.exp(-alpha * (ii * dt - stim_start_nt * dt)))  
                            K_bath[k] = K_bath_BL + K_stim[k] 

                        if N != 1 and k != 0:                                                                       
                            j2 = parents[k]            
                            Icoup[k] = gamma[k] * (v[j2-1] - v[k])  ## uA/cm^2            
                            Icoup[j2-1] += gamma[j2-1] * (v[k] - v[j2-1])            
                            IdiffK[k] = DiffK[k] * (Ko[j2-1] - Ko[k])   # uM/s 
                            IdiffK[j2-1] += DiffK[j2-1] * (Ko[k] - Ko[j2-1]) 
                            IdiffNa[k] = DiffNa[k] * (Nai[j2-1] - Nai[k]) 
                            IdiffNa[j2-1] += DiffNa[j2-1] * (Nai[k] - Nai[j2-1]) 
                            IdiffCa[k] = DiffCa[k] * (Cai[j2-1] - Cai[k]) 
                            IdiffCa[j2-1] += DiffCa[j2-1] * (Cai[k] - Cai[j2-1])
                        else:
                            continue

                elif ii*dt > stim_end_nt*dt:
                    # with exponential decay after stimulation ends
                    for i, comp in enumerate(stim_comp_pot):
                        if k == comp:
                            stim_value = pot_stim[i]
                            # Ks[k] = pot_stim_amp * np.exp(-beta * (ii * dt - stim_end_nt  * dt)) + 2900 * (1 - np.exp(-beta * (ii * dt - stim_end_nt * dt))) 
                            K_stim[k] = stim_value * np.exp(-beta * (ii * dt - stim_end_nt  * dt)) #+ 2900 * (1 - np.exp(-beta * (ii * dt - stim_end_nt * dt)))  #pot_stim_amp 
                            K_bath[k] = K_bath_BL * (1 - np.exp(-beta * (ii * dt - stim_end_nt * dt))) + K_stim[k]
                                     
                        if N != 1 and k != 0:                                                                       
                            j2 = parents[k]            
                            Icoup[k] = gamma[k] * (v[j2-1] - v[k])  ## uA/cm^2            
                            Icoup[j2-1] += gamma[j2-1] * (v[k] - v[j2-1])            
                            IdiffK[k] = DiffK[k] * (Ko[j2-1] - Ko[k])   # uM/s 
                            IdiffK[j2-1] += DiffK[j2-1] * (Ko[k] - Ko[j2-1]) 
                            IdiffNa[k] = DiffNa[k] * (Nai[j2-1] - Nai[k]) 
                            IdiffNa[j2-1] += DiffNa[j2-1] * (Nai[k] - Nai[j2-1]) 
                            IdiffCa[k] = DiffCa[k] * (Cai[j2-1] - Cai[k]) 
                            IdiffCa[j2-1] += DiffCa[j2-1] * (Cai[k] - Cai[j2-1])
                        else:
                            continue
            vol_a = volume[k] * 1e-18 # 50.0e-18  # m^3    
            Aa = area[k] * 1e-12 # 4.0 * pi * (3.0 * vol_a / (4.0 * pi))**(2.0 / 3.0)  # m^2    
            convg = Aa / (F * vol_a) * 1.0e3  # 1.0e3 convert pA/um^2 to uM/sec 

            if 15000 <= ii < 15000 + 200:
                gsyn[k] = gsynmax[k]
            else:
                gsyn[k] = 0.0
            
            Isyn[k] = -1.0 * gsyn[k]

            I_K_bath_to_Ko[k] = 50.0 * (K_bath[k] - Ko[k])
            phofit = 7.34943 #7.34716767              
            
            HCO3o[k] = 10**(pho[k] - pka) * CO2_aq  #uM
            HCO3i[k] = 10**(phi[k] - pho[k]) * HCO3o[k]
            
            Ho[k] = ((s*k_h*P_co2)/HCO3o[k])  # uM
            Hi[k] = ((s*k_h*P_co2)/HCO3i[k])
            
            betai = 25000  + (2.3 * (HCO3i[k])) #uM     
            betao = 10000 + (2.3 * (HCO3o[k])) #uM    

            vkdas=26.7 * np.log((Ko[k]+Nao[k]*0.01)/(Ki[k]+0.01*Nai[k]))
            
            # Synaptic Cleft:
            # Cl_concentration in the SC via electroneutrality(uM):s97
            Clo = Nao[k] + Ko[k] - HCO3o[k]          
         
            rho = rhomin + (rhomax - rhomin) / Glumax * Glu[k] # *0.3d-3  # Uncomment if needed
            
            EKk = phii / zK * np.log(Ko[k] / Ki[k])  # Nernst potentials (in mV)
            ENak = phii / zNa * np.log(Nao[k] / Nai[k])
            EClk = phii / zCl * np.log(Clo / Cli[k])    
            ENBCk = -phii * np.log((Nao[k] * HCO3o[k]**2.0) / (Nai[k] * HCO3i[k]**2.0)) 

            ETRPVk = phii / zCa * np.log(Cap / Cai[k])
            EHCO3 = -26.7 * np.log(HCO3o[k]/HCO3i[k])
            EH=26.7 * np.log(Ho[k]/Hi[k])
            
            ################################################# START NKA PUMP CODE ##################################################################################################
            ##################################################################################################################################################################
            ##################################################################################################################################################################

            # alphas=0.08
            # betas=0.01

            # NKA Pump Na+ affinity: 11.09 * 1000 uM for alpha1beta1 (H.C 2.63), 10.6 * 1000 uM for alpha2beta1 (H.C 2.39), 6.2 * 1000 uM for alpha1beta2 (H.C 1.79), 6.8 * 1000 uM for alpha2beta2 (H.C 1.55)
            # NKA Pump K+ affinity: 0.25 * 1000 uM for alpha1beta1, 0.91 * 1000 uM for alpha2beta1, 0.67 * 1000 uM for alpha1beta2, 3.6 * 1000 uM for alpha2beta2
            KNak_alpha1beta1 = 11.09 * 1000
            HC_alpha1beta1 = 2.63

            KNak_alpha2beta1 = 10.6 * 1000 
            HC_alpha2beta1 = 2.39

            KNak_alpha1beta2 = 7.4 * 1000  #6.2 (Org mean) 7.0 (VLowK+)
            HC_alpha1beta2 = 1.79

            KNak_alpha2beta2 = 6.8 * 1000
            HC_alpha2beta2 = 1.55

            KKs_alpha1beta1 = 0.25 * 1000

            KKs_alpha2beta1 = 0.91 * 1000 # 1.80488e-08 * vk[k]**4 + 1.69091e-06 * vk[k]**3 + 1.49894e-04 * vk[k]**2 + 9.45077e-04 * vk[k] + 1.63552e-01
            
            KKs_alpha1beta2 = 0.67 * 1000
            
            KKs_alpha2beta2 = 3.6 * 1000  # 2.37718e-07 * vk[k]**4 + 3.51700e-05 * vk[k]**3 + 1.97647e-03 * vk[k]**2 + 2.42484e-02 * vk[k] + 1.07469e+00 
            
            # JNaKmax = -1.8081727882526715e-05 * 1000
            # KNak = 27.236605493975365 * 1000
            # KKss = -1.9999981797479687 * 1000
            # if time_t >= 90.00:
            #     JNaKk[k] = JNaKk[k]    
            # else:
            max_peak_value_perc = 1.0
            I_pump_alpha2beta1[k] =  I_pump_max * Nai[k]**HC_alpha2beta1 / (Nai[k]**HC_alpha2beta1 + KNak_alpha2beta1**HC_alpha2beta1) * Ko[k] / (Ko[k] + KKs_alpha2beta1)
            I_pump_alpha2beta1[k] = 1.5  * max_peak_value_perc * I_pump_alpha2beta1[k]        # 1.5 for 14mM Na+ baseline    1.1 for 22/23mM baseline     #max_peak_value_perc[0] to max_peak_value_perc[4] in [0.8, 0.9, 1, 1.1, 1.2]

            I_pump_alpha2beta2[k] =  I_pump_max * Nai[k]**HC_alpha2beta2 / (Nai[k]**HC_alpha2beta2 + KNak_alpha2beta2**HC_alpha2beta2) * Ko[k] / (Ko[k] + KKs_alpha2beta2)
            I_pump_alpha2beta2[k] = 1.5 * max_peak_value_perc * I_pump_alpha2beta2[k]        # 1.5 for 14mM Na+ baseline    1.1 for 22/23mM baseline   
            
            max_d  = 0.0
            # if identifier[k] != 1: #skip somatic compartments
            for bname, comps in branch_to_comps.items():
                if k in comps:              # <- k is a row index (compartment index)
                    max_d =  branch_max_dict.get(bname)
                    break         
     
            # distance_from_soma[k] in [0, max_d]
            if max_d > 0.0:
                r = distance_from_soma[k] / max_d
                # clamp r to [0,1] just in case
                if r < 0.0: r = 0.0
                elif r > 1.0: r = 1.0                
            else:
                r = 0.0       

            w_alpha2beta2 = 0.7 #r
            w_alpha2beta1 = 0.3 #1.0 - r

            I_pump[k] =  w_alpha2beta1 * I_pump_alpha2beta1[k] + w_alpha2beta2 * I_pump_alpha2beta2[k]

            # # JNaKk[k] =  fac_isoform[0] * JNaKk_alpha2beta1[k] + (1-fac_isoform[0]) * JNaKk_alpha2beta2[k] # alpha2beta1 dominated current # fac_isoform[0] to fac_isoform[6] in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
            # # JNaKk[k] =  (1-fac_isoform[6]) * JNaKk_alpha2beta1[k] + fac_isoform[6] * JNaKk_alpha2beta2[k] # alpha2beta2 dominated current # fac_isoform[0] to fac_isoform[6] in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
            # max_distance_from_soma = max(distance_from_soma)
            # if distance_from_soma[k] < max_distance_from_soma/2:
            #     I_pump[k] =  (1-0.7) * I_pump_alpha2beta1[k] + 0.7 * I_pump_alpha2beta2[k]
            # else:
            #     I_pump[k] =  (1-0.9) * I_pump_alpha2beta1[k] + 0.9 * I_pump_alpha2beta2[k]
            # # if 220000000 <= ii <= 240000000: #and k == 10: # 22 sec to 24 sec                           
            # #     JNaKk[k] = JNaKk[k] - 0.25 * JNaKk[k]           
     
            ################################################# END NKA PUMP CODE ##################################################################################################
            ##################################################################################################################################################################
            ##################################################################################################################################################################

            I_k_leak = 0.18 * g_k * (v[k] - EKk) # K+ flux through the K+ channel(uMs^-1)s99 #0.18
            
            # if time_t >= 90.00:
            #     JNak[k] = JNak[k]  
            # else:

            # g_Na_strength = g_Na * (0.5 + 0.8 * r)  # 0.4 → 2.0

            I_Na_leak[k] = 1.3 * g_Na * (v[k] - ENak) #Na+ flux through the Na+ channel(uMs^-1) s100 #2.6
            
            I_NBC[k] =  g_NBC * (v[k] - ENBCk) #Na+ and HCO3- flux through the NBC channel (uMs^-1) s101    
       
            I_KCC1 =  g_KCC1 * phii * np.log((Ko[k] * Clo) / (Ki[k] * Cli[k])) * 30.0  #Cl- and K+ flux through the KCC1 channel(uMs^-1) s102
                   
            I_NKCC1[k] =  g_NKCC1 * phii * np.log((Nao[k] * Ko[k] * Clo**2.0) / (Nai[k] * Ki[k] * Cli[k]**2.0)) 
            
            I_Cl_leak = g_Cl * (v[k] - EClk) #Cl-flux through the cl- channel(uMs^-1) s98          

            Bcyt = 1.0 / (1.0 + Bkend + Kex * Bex / (Kex + Cai[k])**2.0) #Fast Ca2+ buffering is described with in the steady stat approximatin(-)s117
            
            G = (rho + delta) / (KG + rho + delta)   #The ratio of active to total G-protein due to metabotropic glutamate receptor s116


            HCak  = Cai[k]/gamCaik + Cap/gamCaek 
            eta   = (R - R0passivek)/(R0passivek)    
            minfk = (1.0 / (1.0 + np.exp(-(eta - epshalfk) / Kappa))) * ((1.0 / (1.0 + HCak)) * (HCak + np.tanh((v[k] - v1TRPVk) / v2TRPVk)))  # The equilibrium state of the TRPV4 channel (-) s127
            I_TRPV[k] = (g_TRPV / 2.0) * m[k] * (v[k] - ETRPVk)  # TRPV4 channel current calculation     
            
            # Na+/Ca2+(NCX) exchanger equations Osman
            INCXmax = 0.001  # pA/um^2
            KNCXmN = 87500.0  # uM
            KNCXmc = 1380.0  # uM
            etancx = 0.350
            ksat = 0.40
            I_NCX[k] = INCXmax * Nao[k]**3.0 / (KNCXmN**3.0 + Nao[k]**3.0) * cao / (KNCXmc + cao) * ((Nai[k]**3.0 / Nao[k]**3.0 * np.exp(etancx * v[k] / 26.6)) - (Cai[k] / cao * np.exp((etancx - 1.0) * v[k] / 26.6))) / (1.0 + ksat * np.exp((etancx - 1.0) * v[k] / 26.60))
            I_NCX[k] = I_NCX[k] * 0.5e4

            # IGluTmax = 0.10  # pA/um^2
            KGluTmN = 15000.0  # uM
            KGluTmK = 5000.0  # uM
            KGluTmg = 34.0  # uM
            # KGlutHo = 0.1 # uM
            JGluT[k] = IGluTmax[k] * Ki[k] / (Ki[k] + KGluTmK) * Nao[k]**3.0 / (Nao[k]**3.0 + KGluTmN**3.0) * Glu[k] / (Glu[k] + KGluTmg) # test:* Ho[k] / (Ho[k] + KGlutHo)
            # if distance_from_soma[k] < 10.0:    # ER located only within the radius of 10 um from soma        
                # if Glu[k] > 0:
            I_IP3[k] = I_IP3_max * (Ip3_i[k] / (Ip3_i[k] + KI) * Cai[k] / (Cai[k] + Kact) * h_ip3[k])**3.0 * (1.0 - Cai[k] / Ca_ER[k])
            # else:
            #     JIP3[k] = 0.0 
            JERleak = PL * (1.0 - Cai[k] / Ca_ER[k])
            Jpump = Vmax * Cai[k]**2.0 / (Cai[k]**2.0 + kpump**2.0)
            # else:
            #     I_IP3[k] = 0.0
            #     JERleak = 0.0
            #     Jpump = 0.0 
            
            ENHE=26.7 * (np.log((Nai[k]*Ho[k])/(Nao[k]*Hi[k])))   #7.3  membrane potential can be increased if constant is changed from 1.70d0 to 3.8d0            
            I_NHE[k] = g_NBC * (v[k]-ENHE) # * 4.0
            
            EClHNO3=0.0 #26.7 * np.log(Ho/Hi*10.0)
            JHCO3 = 0.0 #GNBCK*0.6*(vk-EHCO3)
            Jh=0.0  #0.32*GNBCK*(vk-Eh)

        # # ensuring there's no diffusion happening for one compartment simulation. Remove these for multi        
            # Icoup[k] = 0.0
        #     Istim[k] = 0.0
        #     IdiffK[k] = 0.0
            # IdiffNa[k] = 0.0
            IdiffCa[k] = 0.0
        #     Na_stim[k] = 0.0

            # uncomment for multicompartmental model only iff this logic needs to be implemented
            # if distance_from_soma <= 50: # TRPV4 to be included at the endfeet
            #     I_TRPV[k] = 0.0

            # JGluT = 0.0        
            # if time_t >= 90.00:
            #     vk[k] = vk[k]
            # else:
            v[k] = v[k] + dt * (conv * (-I_k_leak-I_Cl_leak-I_Na_leak[k]-I_pump[k]+I_NBC[k] + Na_stim[k]- I_NHE[k] -2.0 * I_TRPV[k] + Isyn[k] * 0.01 * convg + Istim[k] * 0.01 * convg + Icoup[k] * 0.01 * convg - I_NCX[k]*convg + 2.0 * JGluT[k] * convg)) #(-JBKk - JKk - JClk - JNBCk - JNak - JNaKk - 2.0 * JTRPVk - (-2.0 * JGluT + JNCX - JNMDA - Jampa) * convg)  #Istim[k] * 0.01 * convg ---- uA/cm^2 to pA/um^2  overall mV/s
                
            if (np.isnan(v[k])): 
                print("Nan values encountered")                 
                # print(g_Na_strength)
                # print(w_alpha2beta1, w_alpha2beta2)                
                break
            Ki[k] = Ki[k] + dt * (-I_k_leak + 2.0 * I_pump[k] + I_NKCC1[k] + I_KCC1 - JGluT[k] * convg) #-JKk + 2.0 * JNaKk + JNKCC1k + JKCC1k - JBKk - JKIRk - (JGluT + JNMDAKk + JampaKk) * convg  # JNMDA * convg
            Nai[k] = Nai[k] + dt * (-I_Na_leak[k] - 3.0 * I_pump[k] + I_NBC[k] - I_NHE[k] + I_NKCC1[k] +  IdiffNa[k] + Na_stim[k] * 1000.0 - 3.0 * I_NCX[k] * convg  + 3.0 * JGluT[k] * convg) #-JNak - 3.0 * JNaKk + JNKCC1k + JNBCk - 3.0 * JNaCak - (3.0 * JNCX - JNMDANak - JampaNak - 3.0 * JGluT) * convg
            
            Cai[k] = Cai[k] + dt * (Bcyt * (I_NCX[k] * convg + I_IP3[k] - Jpump + JERleak + IdiffCa[k] + I_TRPV[k]/rbuff)) #Bcyt * (JIP3 - Jpump + JERleak + JNaCak - JVOCCk + JTRPVk / rbuff + (JNCX + JNMDACak) * convg)
            Cli[k] = Cli[k] + dt * ((-I_k_leak + 2.0 * I_pump[k] + I_NKCC1[k] + I_KCC1 - JGluT[k] * convg) + (-I_Na_leak[k] - 3.0 * I_pump[k] + I_NBC[k] - I_NHE[k] + I_NKCC1[k] + IdiffNa[k] + Na_stim[k] * 1000.0 - 3 * I_NCX[k] * convg  + 3 * JGluT[k] * convg) -(2 * I_NBC[k] + I_NHE[k]) + 2.0  * (Bcyt * (I_NCX[k] * convg + I_IP3[k] - Jpump + JERleak + IdiffCa[k] + I_TRPV[k]/rbuff))) #fun[2] + fun[1] - fun[3] + 2.0 * fun[4]
            
            Ko[k] = Ko[k] + dt * (1.0/VRsa * (I_k_leak - 2.0 * I_pump[k] - I_NKCC1[k] - I_KCC1 + JGluT[k] * convg)  + IdiffK[k] + I_K_bath_to_Ko[k]) #  1.0 / VRsa * (JKk - 2.0 * JNaKk - JNKCC1k - JKCC1k + JKIRk + (JNMDAKk + JampaKk + JGluT) * convg) + JKNEtoSCk  # (JNMDAKk + JampaKk + JEAATKk) * convg
            Nao[k] = Nao[k] + dt * (1.0/VRsa * (I_Na_leak[k] + 3.0 * I_pump[k] - I_NBC[k] + I_NHE[k] - I_NKCC1[k] + 3.0 * I_NCX[k] * convg - 3.0 * JGluT[k] * convg)) #  1.0 / VRsa * (JNak + 3.0 * JNaKk - JNKCC1k - JNBCk - (-3.0 * JNCX + JNMDANak + JampaNak + 3.0 * JGluT) * convg) + JNaNEtoSCk  # (JNMDANak + JampaNak + JEAATNak) * convg

            Ca_ER[k] = Ca_ER[k] + dt * (-Bcyt * (I_IP3[k] - Jpump + JERleak) / VRERcyt)
            h_ip3[k] = h_ip3[k] + dt * (kon * (kinh - (Cai[k] + kinh) * h_ip3[k]))
            if Glu[k] > 0:
                Ip3_i[k] = Ip3_i[k] + dt * (rh * G - kdeg * Ip3_i[k])
            m[k] = m[k] + dt * (trpvswitch * ((minfk - m[k]) / tTRPVk)) 

            phi[k] = phi[k] + dt * ((I_NHE[k] + 2 * I_NBC[k] + Jh - JHCO3)/(betai)) #+0.45d0*(phint-7.33d0)!-JGluT*convg/betai
            pho[k] = pho[k] + dt * (1.0/VRsa * ((-I_NHE[k] -2 * I_NBC[k] -Jh + JHCO3)/betao) + 0.5 * (phofit-pho[k]))   #*ATPNHE/HCO3s!*0.1d0*JNak/HCO3s!-0.0142d0*(pHo-6.20d0)!1000 !.7

         #te = np.array([JBKk, JKk, JClk, JNBCk, JNak, JNaKk, JTRPVk, JGluT, JNCX, JNMDA, Jampa])
                    
        if (np.isnan(v[k])):   
            # print(I_pump[k]) 
            # print(g_Na_strength)             
            break
        if time_t > t_in and ii % iskip == 0:
            #print(f"\033[KCurrent_time: {time_t}", end='\r')           
            time_array[j_10] = time_t  
            I_pumps[:, j_10] = I_pump
            I_NBCs[:, j_10] = I_NBC
            I_NHEs[:, j_10] = I_NHE
            I_NKCC1s[:, j_10] = I_NKCC1
            I_NCXs[:, j_10] = I_NCX * convg
            Icoups[:, j_10] = Icoup * convg
            IdiffKs[:, j_10] = K_bath  
            IdiffNas[:, j_10] = IdiffNa
            IdiffCas[:, j_10] = IdiffCa  
            I_pumps_alpha2beta1[:, j_10] = I_pump_alpha2beta1
            I_pumps_alpha2beta2[:, j_10] = I_pump_alpha2beta2
            JGluTs[:, j_10] = JGluT * convg

            # print(ENak)
            vs[:, j_10] = v            
            Kos[:, j_10] = Ko
            Nais[:, j_10] = Nai
            Clis[:, j_10] = Cli
            Kis[:, j_10] = Ki
            Naos[:, j_10] = Nao
            phis[:, j_10] = phi
            phos[:, j_10] = pho
            Cais[:, j_10] = Cai
            I_Na_leaks[:, j_10] = I_Na_leak
            I_TRPVs[:, j_10] = I_TRPV
            Ip3_is[:, j_10] = Ip3_i
            # Glus[:, j_10] = Glu
            j_10 += 1  
    return vs,  Kos, Nais, Clis, Kis, Naos, time_array, I_pumps, phis, phos, Icoups, IdiffKs, IdiffNas, IdiffCas, Cais, I_Na_leaks, I_TRPVs, Ip3_is, I_NBCs, I_NHEs, I_NKCC1s, I_NCXs, I_pumps_alpha2beta1, I_pumps_alpha2beta2, JGluTs

def concat_ast_param(vs, Nak, Kos, Kks, Cak, pho, phi, IdiffNas, JNaks, JNaKks, Iks, time, JNBCks, JNHEs, JNKCC1ks, JNCXs, JNaKks_alpha2beta1, JNaKks_alpha2beta2, JGluTs, N1, N2):
    vs = pd.DataFrame(vs.T)
    nak = pd.DataFrame(Nak.T)
    kos = pd.DataFrame(Kos.T)
    kks = pd.DataFrame(Kks.T)
    cak = pd.DataFrame(Cak.T)
    pho = pd.DataFrame(pho.T)
    phi = pd.DataFrame(phi.T)
    IdiffNa = pd.DataFrame(IdiffNas.T)
    JNak = pd.DataFrame(JNaks.T)
    JNKA = pd.DataFrame(JNaKks.T)
    Ik = pd.DataFrame(Iks.T)    
    time = pd.DataFrame(time)
    JNBCk = pd.DataFrame(JNBCks.T)
    JNHE = pd.DataFrame(JNHEs.T)
    JNKCC1k = pd.DataFrame(JNKCC1ks.T)
    JNCX = pd.DataFrame(JNCXs.T)
    JGluT = pd.DataFrame(JGluTs.T)
    JNKA_alpha2beta1 = pd.DataFrame(JNaKks_alpha2beta1.T)
    JNKA_alpha2beta2 = pd.DataFrame(JNaKks_alpha2beta2.T)

    nak = nak/1000 #uM to mM
    kos = kos/1000 #uM to mM
    kks = kks/1000 #uM to mM
    cak = cak * 1000 #uM to nM
    JNak = JNak/1000 #uM/s to mM/s
    JNKA = JNKA/1000 #uM to mM
    JNKA_alpha2beta1 = JNKA_alpha2beta1/1000 #uM to mM
    JNKA_alpha2beta2 = JNKA_alpha2beta2/1000 #uM to mM
    IdiffNa = IdiffNa/1000 #uM/s to mM/s
    # JIP3 = JIP3 * 1000 #uM/s to nM/s 
    # JTRPVk = JTRPVk * 1000 #uM/s to nM/s
    JNBCk =JNBCk/1000 #uM/s to mM/s
    JNHE = JNHE/1000 #uM/s to mM/s
    JNKCC1k = JNKCC1k/1000  #uM/s to mM/s
    JNCX = JNCX/1000 #uM/s to mM/s
    JGluT = JGluT/1000 #uM/s to mM/s

    df = pd.concat([time, vs, nak, cak, kos, kks, pho, phi, IdiffNa, JNak, JNKA, JNKA_alpha2beta1, JNKA_alpha2beta2, Ik, JNBCk, JNHE, JNKCC1k, JNCX, JGluT], axis=1)

    columns = ['Time'] + \
          [f'V_{i}' for i in range(N1,N2)] + \
          [f'Nak_{i}' for i in range(N1,N2)] + \
          [f'Cak_{i}' for i in range(N1,N2)] + \
          [f'Kos_{i}' for i in range(N1,N2)] + \
          [f'Kks_{i}' for i in range(N1,N2)] + \
          [f'Pho_{i}' for i in range(N1,N2)] + \
          [f'Phi_{i}' for i in range(N1,N2)] + \
          [f'IdiffNa_{i}' for i in range(N1,N2)] + \
          [f'NaLeak_{i}' for i in range(N1,N2)] + \
          [f'NKA_pump_{i}' for i in range(N1,N2)] + \
          [f'NKA_pump_alpha2beta1_{i}' for i in range(N1,N2)] + \
          [f'NKA_pump_alpha2beta2_{i}' for i in range(N1,N2)] + \
          [f'Ik_{i}' for i in range(N1,N2)]  + \
          [f'JNBCk_{i}' for i in range(N1,N2)] + \
          [f'JNHE_{i}' for i in range(N1,N2)] + \
          [f'JNKCC1k_{i}' for i in range(N1,N2)] + \
          [f'JNCX_{i}' for i in range(N1,N2)] + \
          [f'JGluT_{i}' for i in range(N1,N2)]

    df.columns = columns    
    return df
