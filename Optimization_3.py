import pandapower.networks as nw
import pandapower as pp
import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.sparse import csr_matrix
import mosek
import plotly.express as px
import pandapower.plotting as plot

# Select IEEE test case
def load_network(case):
    if case == 5:
        return nw.case5()
    if case == 14:
        return nw.case14()
    elif case == 30:
        return nw.case30()
    elif case == 57:
        return nw.case57()
    elif case == 118:
        return nw.case118()
    elif case == 300:
        return nw.case300()
    elif case == 'GB':
        return nw.GBnetwork()
    else:
        raise ValueError("Unsupported test case")
    
# Load the chosen IEEE test case
case_number = 5 # Change to 30, 57, or 118 for larger cases
net = load_network(case_number)

pp.runpp(net)  # Run power flow to initialize values
plot.simple_plot(net, show_plot=True, plot_loads=True, plot_gens=True, gen_size =4, load_size=3, line_width=2, bus_size=2)


#make sure all units are consistent (convert everything to per unit)
Sbase = net.sn_mva # unit value for power MVA
Vbase = min(net.bus.vn_kv) # unit value for voltage kV
Zbase = Vbase**2 / Sbase # base impedance in Ohms
Ybase = 1 / Zbase # base admittance in S

#Now, we extract network parameters
Ybus = net._ppc["internal"]["Ybus"].todense() # Bus admittance matrix (Already in per unit. We need todense since the matrices are stored as sparcse matrices, but dense are easier to work with)
n = len(net.bus)  # Number of buses

#we need line information to add line flow constraints
# Extract line connections as (from_bus, to_bus) tuples
lines = list(zip(net.line['from_bus'].astype(int), net.line['to_bus'].astype(int)))

#################### series admittance ##########################

y_lines = {}
for idx, row in net.line.iterrows():
    l = int(row['from_bus'])
    m = int(row['to_bus'])
    y_lines[(l,m)]= -Ybus[l, m]


#y_lines = np.zeros((n, n), dtype=complex) ## Initialize series admittance dictionary (n x n) with all zeros

## just use a list of lines that actually exist
#for l in range(n):
    #for m in range(n):
        #if l != m:
            #y_lines[l, m] = -Ybus[l, m]

###################### shunt admittance ######################

# Note c_nf_per_km (float) - line capacitance in nano Farad per km
y_shunt = {}
f = 50  # System frequency (Hz); change to 60 if using US grid
omega = 2 * np.pi * f  # Angular frequency (rad/s)

for idx, row in net.line.iterrows():
    l = int(row['from_bus'])
    m = int(row['to_bus'])
    C_per_km = row['c_nf_per_km']  # Capacitance in nF/km
    length = row['length_km']      # Line length in km
    C_total = C_per_km * 1e-9 * length  # Total capacitance (F)
    B_total = omega * C_total      # Total susceptance (S)
    y_shunt_lm = 1j * B_total / 2  # Half at each end (π-model)
    y_shunt_pu = y_shunt_lm/Ybase
    y_shunt[(l,m)]=y_shunt_pu
    
#y_shunt = np.zeros((n, n), dtype=complex)
#for k in range(n):
        #y_shunt[k, k] = 1/2 * (Ybus[k, k] + sum(Ybus[k, m] for m in range(n) if m != k))

##################### Double check s_max ####################################

# Define s_max for each line using max_i_ka and Vbase
s_max = {}
p_max = {}
max_v_diff_lm = {}
for idx, row in net.line.iterrows():
    l = int(row['from_bus'])
    m = int(row['to_bus'])
    Imax = row['max_i_ka']  # in kA   # IS This phase current or line current? #################################
    # Smax in MVA
    s_max[(l, m)] = np.sqrt(3) * Vbase * Imax  
    # Since we don't have info on P^max_lm, we will assume it is 95% of S^max_lm.
    p_max[(l, m)] = 0.95 * s_max[(l, m)]
    # Since we don't have the maximum voltage difference across a line, we will estimate it
    v_max_l = net.bus.loc[l, "max_vm_pu"]
    v_min_m = net.bus.loc[m, "min_vm_pu"]
    v_max_m = net.bus.loc[m, "max_vm_pu"]
    v_min_l = net.bus.loc[l, "min_vm_pu"]
    max_v_diff_lm[(l,m)] = max(abs(v_max_l - v_min_m), abs(v_max_m - v_min_l))

######################################################################

# Extract generator information for generation constraints
all_gen = pd.concat([net.gen, net.sgen]) #combine generators and static generators.
gen_buses = all_gen["bus"].values
all_gen = all_gen.groupby("bus").sum(numeric_only=True).reset_index() #if multiple generators are present at a bus, combine them.

# Extract load buses for generation constraints
load_buses = net.load["bus"].values #index of the load buses.
#load active and reactive power for load
p_load = np.zeros(n)
q_load = np.zeros(n)
all_load = net.load.groupby("bus").sum(numeric_only=True).reset_index() #if multiple loads are present at a bus, combine them.
for i,row in all_load.iterrows():
    p_load[int(row['bus'])] = -row['p_mw'] #negative sign because pandapower defines load as positive, but we need it as negative for consumption
    q_load[int(row['bus'])] = -row['q_mvar']

# Voltage limits
min_v_pu = net.bus["min_vm_pu"].values ** 2  # Squared for SDP
max_v_pu = net.bus["max_vm_pu"].values ** 2

# Slack bus voltage
vm_slack = net.ext_grid['vm_pu'][0] #initial voltage of the slack

# Slack bus
slack_bus = int(net.ext_grid["bus"].iloc[0])

#######################################################################

# Extract generator cost coefficients from net.poly_cost
cost = net.poly_cost[net.poly_cost["et"] != 'ext_grid']
c2 = cost["cp2_eur_per_mw2"].values   # Quadratic cost coefficients
c1 = cost["cp1_eur_per_mw"].values   # Linear cost coefficients
c0 = cost["cp0_eur"].values         # Constant cost coefficients

ext = net.poly_cost[net.poly_cost["et"] == 'ext_grid']
c2_ext = ext["cp2_eur_per_mw2"].values   # Quadratic cost coefficients
c1_ext = ext["cp1_eur_per_mw"].values   # Linear cost coefficients
c0_ext = ext["cp0_eur"].values         # Constant cost coefficients

# Ensure the cost coefficients have the correct size
if len(c2) != len(gen_buses):
    print("The number of generators does not match the number of cost coefficients")
elif len(c1) != len(gen_buses):
    print("The number of generators does not match the number of cost coefficients")
elif len(c0) != len(gen_buses):
    print("The number of generators does not match the number of cost coefficients")
else:
    print("The number of generators matches the number of cost coefficients")



############################## Initialize ###################################
# Define our problem matrix, W. It is 2n x 2n, where the first n rows and columns correspond to the real part of the voltage, and the second n rows and columns correspond to the imaginary part of the voltage.
W = cp.Variable((2*n, 2*n), PSD=True)

# Initialize the problem matrices
Y_set = {} # Hermitian part of Y_j
Y_bar_set = {} # Skewed Hermitian part of Y_j
M_set = {}

for k in range(n):
    e_k = np.zeros((n, 1))
    e_k[k] = 1
    Yk = e_k @ e_k.T @ Ybus #Yk is the admittance matrix with only the (k,k) entry, everything else is zero
    #print('Size of Yk', Yk.shape)
    
    #Y_k is the Hermitian part of Y_k
    Y_k = 1/2 * np.block([
        [np.real(Yk + Yk.T),     np.imag(Yk.T - Yk)],
        [np.imag(Yk - Yk.T),     np.real(Yk + Yk.T)]
    ])
    #print('Size of Y_k', Y_k.shape)
    #Y_kbar is the skewed Hermitian part of Y_k
    Y_kbar = -1/2 * np.block([
        [np.imag(Yk + Yk.T),    np.real(Yk - Yk.T)],
        [np.real(Yk.T - Yk),    np.imag(Yk + Yk.T)]
    ])
    #print('Size of Y_kbar', Y_kbar.shape)
    #M_k is the matrix that will be used to enforce the voltage constraints
    M_k = np.block([
        [e_k @ e_k.T,       np.zeros((n, n))],
        [np.zeros((n, n)), e_k @ e_k.T]
    ])
    #print('Size of M_k', M_k.shape)
    Y_set[k] = Y_k
    Y_bar_set[k] = Y_kbar
    M_set[k] = M_k

# Create the matrices for the line constraints
M_line_set = {}
Y_line_set = {}
Y_bar_line_set = {}

for (l, m) in lines:  # lines is a list of (l, m) tuples

    e_l = np.zeros((n, 1))
    e_m = np.zeros((n, 1))
    e_l[l] = 1
    e_m[m] = 1
    delta = e_l - e_m
    M_lm = np.block([
        [delta @ delta.T,          np.zeros((n, n))],
        [np.zeros((n, n)), delta @ delta.T]
    ])
    M_line_set[(l, m)] = M_lm

    # y_lm and y_shunt_lm from precious
    y_lm = y_lines[(l, m)]         
    y_shunt_lm = y_shunt[(l, m)]

    Ylm = (y_shunt_lm + y_lm) * (e_l @ e_l.T) - y_lm * (e_l @ e_m.T)
    #print('Size of Y_lm', Ylm.shape)

    # Real-valued lifting
    Re_Y = np.real(Ylm + Ylm.T)
    Im_Y = np.imag(Ylm.T - Ylm)
    Re_Y_skew = np.real(Ylm - Ylm.T)
    Im_Y_sum = np.imag(Ylm + Ylm.T)

    Y_lm = 1/2 * np.block([
        [Re_Y,      Im_Y],
        [-Im_Y,     Re_Y]
    ])
    #print('Size of Y_lm', Y_lm.shape)
    Y_lmbar = -1/2 * np.block([
        [Im_Y_sum,      Re_Y_skew],
        [-Re_Y_skew,    Im_Y_sum]
    ])
    #print('Size of Y_lmbar', Y_lmbar.shape)
    Y_line_set[(l, m)] = Y_lm
    Y_bar_line_set[(l,m)] = Y_lmbar

############################## Objective Function #############################


alpha = cp.Variable(len(gen_buses))
objective = cp.Minimize(cp.sum(alpha))

    
############################## Constraints ####################################
constraints = []

for k in range(n):

    # For the slack bus, we fix the voltage magnitude since it is not adjustable
    if k == slack_bus:
        constraints.append(cp.trace(M_set[k] @ W) == vm_slack**2)

    else:

        # Voltage constraints (except slack)
        constraints.append(min_v_pu[k] <= cp.trace(M_set[k] @ W)) #constraint 4c, note that min_v_pu is already squared
        constraints.append(cp.trace(M_set[k] @ W) <= max_v_pu[k]) #constraint 4c

for idx, row in all_gen.iterrows():
    
    k = int(row['bus']) # k is the index of the actual bus where the generator idx is at.
    Pgen_k = cp.trace(Y_set[k] @ W) * Sbase # Initialize the power balance variables
    Qin_k = cp.trace(Y_bar_set[k] @ W) * Sbase

    Pin_k = cp.trace(Y_set[k] @ W) * Sbase + p_load[k] #we use positive since p_load is negative
    Qin_k = cp.trace(Y_bar_set[k] @ W) * Sbase + q_load[k]
    idx = np.where(gen_buses == k)[0][0]

    constraints.append(row['min_p_mw'] <= Pin_k) #constraint 4a
    constraints.append(Pin_k <= row['max_p_mw']) #constraint 4a
    constraints.append(row['min_q_mvar'] <= Qin_k) #constraint 4b
    constraints.append(Qin_k <= row['max_q_mvar']) #constraint 4b

for idx, row in cost.iterrows():
    
    k = int(row['element']) # k is the index of the actual bus where the generator idx is at.
    Pgen_k = cp.trace(Y_set[k] @ W) * Sbase # Initialize the power balance variables

    c2_k = row["cp2_eur_per_mw2"]   # Quadratic cost coefficients
    c1_k = row["cp1_eur_per_mw"]   # Linear cost coefficients
    c0_k = row["cp0_eur"]         # Constant cost coefficients
    Pd_k = p_load[k]
    
    a_k = c0_k - c1_k * Pd_k #negative because p_load is negative
    b_k = float(np.sqrt(c2_k)) * -Pd_k #negative because p_load is negative

    top_left = c1_k * Pgen_k - alpha[idx] + a_k
    top_right = float(np.sqrt(c2_k)) * Pgen_k + b_k

    Schur = cp.bmat([
        [top_left, top_right],
        [top_right, -1]
    ])
    constraints.append(Schur << 0) # constraint 6

# Line Flow Constraints
P_lm = {}
Q_lm = {}
for (l, m) in lines:
    Plm = cp.trace(Y_line_set[(l, m)] @ W) * Sbase
    Qlm = cp.trace(Y_bar_line_set[(l, m)] @ W) * Sbase
    Smax = s_max[(l, m)]

    P_lm[(l, m)] = Plm
    Q_lm[(l, m)] = Qlm

    S_constr = cp.bmat([
        [-Smax**2, Plm, Qlm],
        [Plm, -1, 0],
        [Qlm, 0, -1]
    ])

    constraints.append(Plm <= p_max[(l,m)])  # constraint 4e
    constraints.append(cp.trace(M_line_set[(l,m)]@W) <= max_v_diff_lm[l,m]**2)  # constraint 4f
    constraints.append(S_constr << 0) # constraint 5

# Positive semidefinite constraint
constraints.append(W >> 0) ##constraint 4g

# Solve the SDP
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK)

# Print results
print("Problem Status:", prob.status)
print("Optimal cost EUR:", prob.value)