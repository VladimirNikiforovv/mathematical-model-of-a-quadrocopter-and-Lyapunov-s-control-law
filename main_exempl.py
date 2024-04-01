from quadrocopter_dynamics_simulation import *
import random as rn

"""пример симуляции"""
"""инициация начальных условий задачи Коши"""
w_init = np.array([0,0,0])
q_init = np.array([1,0,0,0])
p_init = np.array([0,0,0])
r_init = np.array([0,0,0])
N = 1000
h = 0.01
"""инициация возмущения"""
Mext = np.zeros((3,N))
Mext[0,:] = 0.00001*np.array([rn.random() for i in range(N)])
Mext[1,:] = 0.00001*np.array([rn.random() for i in range(N)])
Mext[2,:] = 0.00001*np.array([rn.random() for i in range(N)])
"""инициация функций управления"""
w_helix = np.zeros((4,N))
t = np.linspace(0,h*N, N)
ww_hel = np.zeros((4,N))
"""ввод управления"""
ww_hel[0,:] = 2.214*np.cos(t/33)
ww_hel[1,:] = -2.214*np.sin(t/33)
ww_hel[2,:] = 2.214*np.cos(t/33)
ww_hel[3,:] = -2.214*np.sin(t/33)
"""создание объекта"""        
QQ1 = quadrocopter()    
"""ввод возмущения"""      
QQ1.perturbation(Mext)
"""ввод управления""" 
QQ1.control(ww_hel)
"""вызов решения задачи Коши"""
QQ1.motion_simulation(h, N, w_init, q_init, p_init, r_init)
"""вывод анимации"""
QQ1.simulation_animate()
"""вывод графиков расчитанных функций движения""" 
QQ1.motion_pattern_graph()