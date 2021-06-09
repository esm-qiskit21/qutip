import numpy as np
import krotov
import matplotlib.pyplot as plt
import datetime
from qutip import Qobj, identity, sigmax, sigmay, sigmaz, sigmam, tensor
from qutip.superoperator import liouvillian, sprepost, spre,spost
from qutip.qip.operations import cnot
from qutip.qip.operations import cr, rz, rx
import qutip.logging_utils as logging
logger = logging.get_logger()
#Set this to None or logging.WARN for 'quiet' execution
log_level = logging.INFO
#QuTiP control modules
import qutip.control.pulseoptim as cpo
from qutip.tomography import qpt, qpt_plot_combined

#example_name = 'Lindblad'

#The Hamiltoninan model
Sx = sigmax()
Sy = sigmay()
Sz = sigmaz()
#Sm = sigmam()
Si = identity(2)
sm1 = tensor(sigmam(),Si)
sm2 = tensor(Si,sigmam())
#Rome
#w0 = 4.969 #GHz
#w1 = 4.77 #GHz
#w2 = 5.015 
#w3 = 5.259
#Manhattan
w0 = 4.838
w1 = 4.681
del12 = w1-w0 #GHz
#del23 = w2-w3
J = (0.0104)/(2*np.pi)  #(0.0084019)/(2*np.pi) #GHz
coup = (J*0.25/del12)

#CR gate
def targ_unit():
    mat = (np.sqrt(1/2))*np.array([[1,0,-1.j,0],[0,1,0,1.j],[-1.j,0,1,0],[0,1.j,0,1]])
    return Qobj(mat,dims=[[2,2],[2,2]])
#def had_gate():
#	mat = (np.sqrt(1/2))*np.array([[1,1],[1,-1]])
#	return Qobj(mat,dims=[[2],[2]])
#cnot gate
cnot_gate = cnot()
print(tensor(rz(phi=np.pi/2),Si)*tensor(Si,rx(phi=np.pi/2))*cr(phi=-np.pi/2.))
#u_targ = tensor(rz(phi=np.pi/2),Si)*tensor(Si,rx(phi=np.pi/2))*cr(phi=-np.pi/2.)
u_targ = cr(phi=-np.pi/2)
#u_targ = had_gate()
U0 = identity(4)
print('U0=',U0)
print('targ=',u_targ)
# Time allowed for the evolution
#evo_times = [140,150,160,175,195]
evo_times = [384]
#evo_times = [1312]
# Number of time slots
#n_ts = int(float(evo_time/0.222))
#H0 = (4.969/2.)*Sz
#Hc = [Sz,Sx,Sy]

H0 = -0.5*del12*tensor(Sz,Si)
#H0 = w0*tensor(Sz,Si)+w1*tensor(Si,Sz)
#Hc = [0.99*tensor(Si,Sx),0.008*tensor(Sz,Sx)]
#H0 = 0*0.5*tensor(Sz,Sx)+ 0*0.5*tensor(Sx,Si)
#ctrl_term = (0.5*tensor(Si,Sx)-coup*tensor(Sz,Sx))

#ctrl_term = tensor(Sz,Sx)+tensor(Sy,Sx)+tensor(Sz,Sz)
#Hc = [ctrl_term]#,tensor(Sz,Sz),tensor(Si,Sx)]
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
#Hc = [tensor(Sz,Sx),tensor(Si,Sx),tensor(Si,Sy),tensor(Si,Sz),tensor(Sz,Si),tensor(Sz,Sy),tensor(Sz,Sz)]
nctrl = len(Hc)
print('***********nctrl***********',nctrl)
# Fidelity error target
fid_err_targ = 1e-5
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
	result = cpo.optimize_pulse_unitary(H0, Hc, U0, u_targ, n_ts, evo_time,
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
ax1.set_title("Initial control amps for CR gate")
ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
for j in range(nctrl):
    ax1.step(result.time, 
             np.hstack((result.initial_amps[:, j], result.initial_amps[-1, j])), 
             where='post')

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences for CR gate")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(nctrl):
    ax2.step(result.time, 
             np.hstack((result.final_amps[:, j], result.final_amps[-1, j])), 
             where='post')
fig1.tight_layout()
plt.show()

#from qutip.operators import qeye
#op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()]] * 2
#op_label = [["i", "x", "y", "z"]] * 2
#cnot1 = tensor(rz(phi=np.pi/2),Si)*tensor(Si,rx(phi=np.pi/2))*result.evo_full_final
#fig = plt.figure()
#SU = spre(result.evo_full_final)*spost(result.evo_full_final.dag())
#SU = spre(cnot1)*spost(cnot1.dag())
#chi = qpt(SU,op_basis)
#fig = qpt_plot_combined(chi, op_label, fig=fig)
#plt.show()

with open('test_cr_gate_manhattan_w0w1.npy','wb') as f:
    np.save(f,result.final_amps.T[0])
    np.save(f,result.final_amps.T[1])
    np.save(f,result.final_amps.T[2])
    np.save(f,result.final_amps.T[3])






