from quadrocopter_dynamics_simulation import *
from quadrocopter_lyapunov_control import *

"""инициализация временных параметров"""
N = 3000
h = 0.01
"""инициация возмущения"""
Mext = np.zeros((3,N))
"""создание объекта"""        
QQ1 = quadrocopter_control()     
"""ввод возмущения"""      
QQ1.perturbation(Mext)
"""инициализация начальных условий"""
w_init = np.array([0,0,0])
q_init = np.array([1,0,0,0])
p_init = np.array([0,0,0])
r_init = np.array([0,0,0])
u_int = np.array([0,0,0,0])
"""определения желаемых импульсов """
p_like_cont = np.zeros((3,N))
# p_like_cont[2,:] = 1

t = np.linspace(0, h*N,N)
p_like_cont[1,:] =  7e-2*np.cos(t[:]/2) 
"""вычисления функций управления"""
QQ1.control_calc(h, N, w_init, q_init, p_init, u_int, 0.2, 0.3, 7, p_like_cont)
"""создание объекта"""        
QQ2 = quadrocopter()    
"""ввод возмущения"""      
QQ2.perturbation(Mext)
"""ввод управления""" 
QQ2.control(QQ1.ww_h)
"""вызов решения задачи Коши"""
QQ2.motion_simulation(h, N, w_init, q_init, p_init, r_init)
"""вывод анимации"""
QQ2.simulation_animate()
"""вывод графиков расчитанных функций движения""" 
QQ2.motion_pattern_graph()


# plt.plot(QQ1.ww_h[0,:-1])
# plt.plot(QQ1.ww_h[1,:-1])
# plt.plot(QQ1.ww_h[2,:-1])
# plt.plot(QQ1.ww_h[3,:-1])