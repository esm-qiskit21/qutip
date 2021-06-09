import numpy as np
import matplotlib.pyplot as plt
import datetime
from qutip import Qobj, identity, sigmax, sigmay, sigmaz, sigmam, tensor
from qutip.superoperator import liouvillian, sprepost
from qutip.qip.operations import cnot, cr
import qutip.logging_utils as logging
logger = logging.get_logger()
#Set this to None or logging.WARN for 'quiet' execution
log_level = logging.INFO
#QuTiP control modules
import qutip.control.pulseoptim as cpo

#example_name = 'Lindblad'

#The Hamiltoninan model
Sx = sigmax()
Sy = sigmay()
Sz = sigmaz()
#Sm = sigmam()
Si = identity(2)
sm1 = tensor(sigmam(),Si)
sm2 = tensor(Si,sigmam())
sz1 = tensor(Sz,Si)
sz2 = tensor(Si,Sz)
#cnot gate
cnot_gate = cnot()
U_targ = cr(phi=-np.pi/2)
U0 = identity(4)

#Hamiltonian parameters
#IBM Bogota
w0 = 5. #GHz
w1 = 4.844 #GHz
del12 = w0-w1 #GHz
J = (0.0083)/(2*np.pi) #GHz
coup = (J*0.25/del12)
t01 = (85.86/1e-3) #ns
t02 = (108.53/1e-3) #ns
t11 = (113.13/1e-3) #ns
t12 = (72.74/1e-3) #ns
# Time allowed for the evolution
#evo_times = [140,150,160,175,195]
evo_times = [1312]
#evo_times = [384]
#evo_times = [350]

# Number of time slots
#n_ts = int(float(evo_time/0.222))

#H0 = -(2.22e-1/2.)*tensor(Sz,Si)*0.#toronto
#Hc = [(-(2.22e-1/2.)*tensor(Sz,Si)),(0.5*tensor(Si,Sx)),-(0.5*0.01*tensor(Sz,Sx))]

#H0 = 0.5*tensor(Sz,Sx)+ 0.5*tensor(Sx,Si)
#Hc = [tensor(Si,Sx),tensor(Si,Sy),tensor(Sx,Si),tensor(Sy,Si)]

#H0 = w0*tensor(Sz,Si)+w1*tensor(Si,Sz)
H0 = -0.5*del12*tensor(Sz,Si)
#(-J*0.25/del12)*
#Hc = [tensor(Sz,Sx),tensor(Si,Sx),tensor(Si,Sy),tensor(Sy,Si)]
#Hc = [tensor(Sz,Sx),tensor(Si,Sx),tensor(Si,Sy),tensor(Si,Sz),tensor(Sz,Si),tensor(Sz,Sy),tensor(Sz,Sz)]

#ctrl_term = (tensor(Sz,Si)+tensor(Si,Sx)-tensor(Sz,Sx))#tensor(Sz,Sx)+tensor(Si,Sx)+tensor(Sz,Si)

#Hc = [ctrl_term]#,tensor(Sz,Sz),tensor(Si,Sx),tensor(Si,Sy)]
Hc = [tensor(Si,Sx),
         tensor(Si,Sy),
         tensor(Si,Sz),
         #tensor(identity(2), sigmax()),
         #tensor(identity(2), sigmay()),
         #tensor(identity(2), sigmaz()),
         tensor(Sz, Sx) +
         tensor(Sz, Sy) +
         tensor(Sz, Sz)]

#Hc = [tensor(Si,Sx),tensor(Sx,Si),tensor(Sx,Sx),tensor(Sy,Sy)]
gamma01 = 1/t01 #ns-1  t1 qubit 1
gamma02 = 1/t02 #ns-1  t1 qubit 2
gamma11 = 1/t11
gamma12 = 1/t12

L0 = liouvillian(H0,[np.sqrt(gamma01)*sm1,np.sqrt(gamma11)*sm2])
#,np.sqrt(gamma02)*sz1,np.sqrt(gamma12)*sz2])
LC_zx = liouvillian(Hc[3])
#LC_zz = liouvillian(Hc[1])

LC_ix = liouvillian(Hc[0])
#LC_xi = liouvillian(Hc[1])
#LC_xx = liouvillian(Hc[2])
#LC_yy = liouvillian(Hc[3])

LC_iy = liouvillian(Hc[1])
LC_iz = liouvillian(Hc[2])
#LC_zi = liouvillian(Hc[4])
#LC_zy = liouvillian(Hc[5])
#LC_zz = liouvillian(Hc[6])



drift = L0
#ctrls = [LC_ix,LC_xi,LC_xx,LC_yy]#,LC_yi]
ctrls = [LC_ix,LC_iy,LC_iz,LC_zx]#,LC_ix,LC_iy,LC_iz,LC_zi,LC_zy,LC_zz]
#ctrls = [LC_zx,LC_ix,LC_iy,LC_yi]
#ctrls = [LC_zx,LC_ix,LC_iy,LC_yi]
nctrl = len(ctrls)

E0 = sprepost(U0,U0)
#E_targ = sprepost(cnot_gate,cnot_gate)
E_targ = sprepost(U_targ,U_targ)

# Fidelity error target
fid_err_targ = 1e-3
# Maximum iterations for the optisation algorithm
max_iter = 3500
# Maximum (elapsed) time allowed in seconds
max_wall_time = 1600
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20
p_type = 'CUSTOM'
for evo_time in evo_times:
	n_ts = int(float(evo_time/0.222))
	result = cpo.optimize_pulse(drift, ctrls, E0, E_targ, n_ts, evo_time,
                             amp_lbound=0,amp_ubound=1,
                fid_err_targ=fid_err_targ, min_grad=min_grad, 
                max_iter=max_iter, max_wall_time=max_wall_time, 
                out_file_ext=None, init_pulse_type=p_type, 
                log_level=log_level, gen_stats=True)
	result.stats.report()
	print("Final evolution\n{}\n".format(result.evo_full_final))
	print("********* Summary *****************")
	print("Initial fidelity error {}".format(result.initial_fid_err))
	print("Final fidelity error {}".format(result.fid_err))
	print("Final gradient normal {}".format(result.grad_norm_final))
	print("Terminated due to {}".format(result.termination_reason))
	print("Number of iterations {}".format(result.num_iter))
	print("Completed in {} HH:MM:SS.US".format(datetime.timedelta(seconds=result.wall_time)))
	print("results for evolution time{}".format(evo_time))



    
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial control amps for CNOT gate")
ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
for j in range(nctrl):
    ax1.step(result.time, 
             np.hstack((result.initial_amps[:, j], result.initial_amps[-1, j])), 
             where='post')

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences for CNOT gate")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(nctrl):
    ax2.step(result.time, 
             np.hstack((result.final_amps[:, j], result.final_amps[-1, j])), 
             where='post')
fig1.tight_layout()
plt.show()



#with open('test_cnot_bogota_lind1.npy','wb') as f:
#    np.save(f,result.final_amps.T[0])
#    np.save(f,result.final_amps.T[1])
#    np.save(f,result.final_amps.T[2])
#    np.save(f,result.final_amps.T[3])
    #np.save(f,result.final_amps.T[4])
   







