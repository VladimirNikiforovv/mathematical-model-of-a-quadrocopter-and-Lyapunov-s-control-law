import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import random as rn
from scipy.optimize import fsolve

class quadrocopter_control:
    """Класс определения динамики квадракоптера"""
    def __init__(self, m_qq = 1, m_hl = 0.01, k_L = 1, L = 1, L_hh = 0.1):
        """коэфициент подъемной силы винта"""
        self.k_L = k_L 
        """ускорение силы тяжести"""
        self.g = -9.8
        """полная масса коптера"""
        self.m_qq = m_qq
        """масса винта"""
        self.m_hl = m_hl
        """длина плеча"""
        self.L = L
        """длинна лопости винта"""
        self.L_hh = L_hh
        
        """тензор инерции коптера по умолчанияю"""
        self.J_qq = np.array([[(self.L/4)**2*self.m_qq/2, 0, 0],
                              [0, (self.L/4)**2*self.m_qq/2, 0],
                              [0, 0, (self.L/2)**2*self.m_qq/2]])
        
        """тензор инерции винтов по умолчанияю, для всех одинаковый"""
        J_hh = np.array([[(self.L_hh/4)**2*self.m_qq/2, 0, 0],
                         [0, (self.L_hh/4)**2*self.m_qq/2, 0],
                         [0, 0, (self.L_hh/2)**2*self.m_qq/2]])
        self.J_hh_0 = J_hh
        self.J_hh_1 = J_hh
        self.J_hh_2 = J_hh
        self.J_hh_3 = J_hh
        
    def inertia_tensor(self, J_qq, J_hh_0, J_hh_1, J_hh_2, J_hh_3):
        """определение моментов инерции коптера и винтов, по умолчанию это 
        тензоры инерции плоского диска"""
        self.J_qq = J_qq
        self.J_hh_0 = J_hh_0
        self.J_hh_1 = J_hh_1
        self.J_hh_2 = J_hh_2
        self.J_hh_3 = J_hh_3
        
    def perturbation(self, Mext):
        """Определение возмущающих моментов"""
        self.Mextr = Mext
        
    def control_calc(self, h, N, w_init, q_init, p_init, u_init, gamma, beta, k_q, p_like):
        """Определение шага итегрирования, колличества итераций и 
        начальных условий моделирования"""
        
        """Определение векторов угловой скорости, кватернионной части,
        импульса и радикс-вектора"""         
        w = np.zeros((3,N))
        q = np.zeros((4,N))
        p = np.zeros((3,N))
        
        u = np.zeros((4,N))
        w_h = np.zeros((4,N))
        
        Mext = self.Mextr
        
        """Определение начальных условий"""
        w[:,0] = w_init[:]
        q[:,0] = q_init[:]
        p[:,0] = p_init[:]
        
        u[:,0] = u_init[:]
                
        self.h = h
        self.N = N  
        self.beta = beta
        self.k_q = k_q
        self.gamma = gamma
        
        def q_optimize(u2, q0, p0, p1, p2, p_like, q1n1, q2n1, q3n1, wqq0, wqq1, wqq2, Mext, u0n1, u1n1, u3n1):
            """решение системы нелинейных уравнений для разрешения задачи управления через кватернионы, выпуклый оптимизатор"""
            def equation_q(p):
                u0, u1, u3, q1, q2, q3 = p
            
                return ((-2*self.beta*(q0*q3 + q1*q2)*(p1- p_like[1]) - self.beta*(q0**2 + q1**2 - q2**2 - q3**2)*(p0- p_like[0]) + 2*(self.beta*(p2- p_like[2]) + self.g*self.m_qq)*(q0*q2 - q1*q3),
                         2*self.beta*(q0*q3 - q1*q2)*(p0- p_like[0]) - self.beta*(q0**2 - q1**2 + q2**2 - q3**2)*(p1- p_like[1]) - 2*(self.beta*(p2- p_like[2]) + self.g*self.m_qq)*(q0*q1 + q2*q3),
                         (2*self.beta*(q0*q1 - q2*q3)*(p1- p_like[1]) - 2*self.beta*(q0*q2 + q1*q3)*(p0- p_like[0]) - (self.beta*(p2- p_like[2]) + self.g*self.m_qq)*(q0**2 - q1**2 - q2**2 + q3**2)) - u3,
                         q0**2 + q1**2 + q2**2 + q3**2 - 1,
                         (self.J_qq[1,1]*wqq1*wqq2 + Mext[0] + self.gamma*wqq0 + 1/2*(self.k_q*q1) -(self.J_qq[2,2]*wqq2 + u2)*wqq1) + u0,
                         (-self.J_qq[0,0]*wqq0*wqq2 - Mext[1] + self.gamma*wqq1 + 1/2*(self.k_q*q2) +(self.J_qq[2,2]*wqq2 + u2)*wqq0) + u1))
            
            u0, u1, u3, q1, q2, q3 = fsolve(equation_q, (u0n1, u1n1, u3n1, q1n1, q2n1, q3n1))
            
            return np.array([u0, u1, u3, q1, q2, q3])
        
        def w_h_optimize(u0, u1, u2, u3, w_init):
            """Вычисление через оптимизатор скоростей роторов"""
            def equation_u(p):
                x, y, z, u = p
                
                return ((self.L*self.k_L/2)*y**2 - (self.L*self.k_L/2)*u**2 - u0,
                       -(self.L*self.k_L/2)*x**2 + (self.L*self.k_L/2)*z**2 - u1, 
                        self.J_hh_0[2,2]*x+self.J_hh_1[2,2]*y+self.J_hh_2[2,2]*z+self.J_hh_3[2,2]*u - u2,
                        self.k_L*x**2 + self.k_L*y**2 +self.k_L*z**2 + self.k_L*u**2 - u3)
            
            x, y, z, u = fsolve(equation_u, (-1, 1, -1, 1))
            """wh0 = x, wh1 = y, wh2 = z, wh3 = u"""
            return np.array([x, y, z, u])

        """функции правой части динамической системы уравнений"""
        def dw_qq_0dt(wqq0, wqq1, wqq2, q0, q1, q2, q3):
            return (-(gamma*wqq0 + 1/2*(1*k_q*q1))/self.J_qq[0,0])
        
        def dw_qq_1dt(wqq0, wqq1, wqq2, q0, q1, q2, q3):
            return (-(gamma*wqq1 + 1/2*(1*k_q*q2))/self.J_qq[1,1])
        
        def dw_qq_2dt(wqq0, wqq1, wqq2, q0, q1, q2, q3):
            return (-(gamma*wqq2 + 1/2*(1*k_q*q3))/self.J_qq[2,2])
        
                
        def du2dt(wqq0, wqq1, wqq2, q0, q1, q2, q3, Mext):
            return (self.J_qq[0,0]*wqq0*wqq1 - self.J_qq[1,1]*wqq0*wqq1 + 
                    Mext[2] + gamma*wqq2 + 1/2*(1*k_q*q3))
        
        def u3(q0, q1, q2, q3, p0, p1, p2, p_like):
            return (2*self.beta*(q0*q1 - q2*q3)*(p1- p_like[1]) - 
                    2*self.beta*(q0*q2 + q1*q3)*(p0- p_like[0]) - 
                    (self.beta*(p2- p_like[2]) + self.g*self.m_qq)*
                    (q0**2 - q1**2 - q2**2 + q3**2))

        def dq0dt(w_0, w_1, w_2, q_0, q_1, q_2, q_3):
            """соответствует производным кватернионной части"""
            return (-0.5*q_1*w_0 - 0.5*q_2*w_1 - 0.5*q_3*w_2)
        """импульсы"""
        def dp0dt(p0, p1, p2, p_like):
            return - self.beta*(p0 - p_like)
        
        def dp1dt(p0, p1, p2, p_like):
            return - self.beta*(p1 - p_like)
        
        def dp2dt(p0, p1, p2, p_like):
            return - self.beta*(p2 - p_like)
        
        for i in range(0, N-1):
                        
            if i == 0:
                q[1:,i] = q_optimize(u_init[2], q[0,i], p[0,i], p[1,i], p[2,i], p_like[:,i], q_init[1],q_init[2], q_init[3], w[0,i], w[1,i], w[2,i], Mext[:,i], u_init[0], u_init[1], u_init[3])[3:]
                u[0,i] = q_optimize(u_init[2], q[0,i], p[0,i], p[1,i], p[2,i], p_like[:,i], q_init[1],q_init[2], q_init[3], w[0,i], w[1,i], w[2,i], Mext[:,i], u_init[0], u_init[1], u_init[3])[0]
                u[1,i] = q_optimize(u_init[2], q[0,i], p[0,i], p[1,i], p[2,i], p_like[:,i], q_init[1],q_init[2], q_init[3], w[0,i], w[1,i], w[2,i], Mext[:,i], u_init[0], u_init[1], u_init[3])[1]
                u[3,i] = q_optimize(u_init[2], q[0,i], p[0,i], p[1,i], p[2,i], p_like[:,i], q_init[1],q_init[2], q_init[3], w[0,i], w[1,i], w[2,i], Mext[:,i], u_init[0], u_init[1], u_init[3])[2]
            else:
                q[1:,i] = q_optimize(u[2,i], q[0,i], p[0,i], p[1,i], p[2,i], p_like[:,i], q[1,i-1], q[2,i-1], q[3,i-1], w[0,i], w[1,i], w[2,i], Mext[:,i], u[0,i-1], u[1,i-1], u[3,i-1])[3:]
                u[0,i] = q_optimize(u[2,i], q[0,i], p[0,i], p[1,i], p[2,i], p_like[:,i], q[1,i-1], q[2,i-1], q[3,i-1], w[0,i], w[1,i], w[2,i], Mext[:,i], u[0,i-1], u[1,i-1], u[3,i-1])[0]
                u[1,i] = q_optimize(u[2,i], q[0,i], p[0,i], p[1,i], p[2,i], p_like[:,i], q[1,i-1], q[2,i-1], q[3,i-1], w[0,i], w[1,i], w[2,i], Mext[:,i], u[0,i-1], u[1,i-1], u[3,i-1])[1]
                u[3,i] = q_optimize(u[2,i], q[0,i], p[0,i], p[1,i], p[2,i], p_like[:,i], q[1,i-1], q[2,i-1], q[3,i-1], w[0,i], w[1,i], w[2,i], Mext[:,i], u[0,i-1], u[1,i-1], u[3,i-1])[2]
            """перенормировка кватерниона"""
            q[0,i] = q[0,i] / (np.sqrt(q[0,i]**2 + q[1,i]**2 + q[2,i]**2 + q[3,i]**2))
            q[1,i] = q[1,i] / (np.sqrt(q[0,i]**2 + q[1,i]**2 + q[2,i]**2 + q[3,i]**2))
            q[2,i] = q[2,i] / (np.sqrt(q[0,i]**2 + q[1,i]**2 + q[2,i]**2 + q[3,i]**2))
            q[3,i] = q[3,i] / (np.sqrt(q[0,i]**2 + q[1,i]**2 + q[2,i]**2 + q[3,i]**2))
            """метод Рунге-Кутты 4-го порядка"""  
            """k1"""
            w0_k1 = h*dw_qq_0dt(w[0,i], w[1,i], w[2,i], q[0,i], q[1,i], q[2,i], q[3,i])
            w1_k1 = h*dw_qq_1dt(w[0,i], w[1,i], w[2,i], q[0,i], q[1,i], q[2,i], q[3,i])
            w2_k1 = h*dw_qq_2dt(w[0,i], w[1,i], w[2,i], q[0,i], q[1,i], q[2,i], q[3,i])
            
            u2_k1 = h*du2dt(w[0,i], w[1,i], w[2,i], q[0,i], q[1,i], q[2,i], q[3,i], Mext[:,i])
            
            p0_k1 = h*dp0dt(p[0,i], p[1,i], p[2,i], p_like[0,i])
            p1_k1 = h*dp1dt(p[0,i], p[1,i], p[2,i], p_like[1,i])
            p2_k1 = h*dp2dt(p[0,i], p[1,i], p[2,i], p_like[2,i])   
            
            q0_k1 = h*dq0dt(w[0,i], w[1,i], w[2,i], q[0,i], q[1,i], q[2,i], q[3,i])
            
            """k2"""
            w0_k2 = h*dw_qq_0dt(w[0,i]+w0_k1/2, w[1,i]+w1_k1/2, w[2,i]+w2_k1/2, q[0,i]+q0_k1/2, q[1,i], q[2,i], q[3,i])
            w1_k2 = h*dw_qq_1dt(w[0,i]+w0_k1/2, w[1,i]+w1_k1/2, w[2,i]+w2_k1/2, q[0,i]+q0_k1/2, q[1,i], q[2,i], q[3,i])
            w2_k2 = h*dw_qq_2dt(w[0,i]+w0_k1/2, w[1,i]+w1_k1/2, w[2,i]+w2_k1/2, q[0,i]+q0_k1/2, q[1,i], q[2,i], q[3,i])
            
            u2_k2 = h*du2dt(w[0,i]+w0_k1/2, w[1,i]+w1_k1/2, w[2,i]+w2_k1/2, q[0,i]+q0_k1/2, q[1,i], q[2,i], q[3,i], Mext[:,i])
            
            p0_k2 = h*dp0dt(p[0,i]+p0_k1/2, p[1,i]+p1_k1/2, p[2,i]+p2_k1/2, p_like[0,i])
            p1_k2 = h*dp1dt(p[0,i]+p0_k1/2, p[1,i]+p1_k1/2, p[2,i]+p2_k1/2, p_like[1,i])
            p2_k2 = h*dp2dt(p[0,i]+p0_k1/2, p[1,i]+p1_k1/2, p[2,i]+p2_k1/2, p_like[2,i])
            
            q0_k2 =  h*dq0dt(w[0,i]+w0_k1/2, w[1,i]+w1_k1/2, w[2,i]+w2_k1/2, q[0,i]+q0_k1/2, q[1,i], q[2,i], q[3,i])
            
            """k3"""
            w0_k3 = h*dw_qq_0dt(w[0,i]+w0_k2/2, w[1,i]+w1_k2/2, w[2,i]+w2_k2/2, q[0,i]+q0_k2/2, q[1,i], q[2,i], q[3,i])
            w1_k3 = h*dw_qq_1dt(w[0,i]+w0_k2/2, w[1,i]+w1_k2/2, w[2,i]+w2_k2/2, q[0,i]+q0_k2/2, q[1,i], q[2,i], q[3,i])
            w2_k3 = h*dw_qq_2dt(w[0,i]+w0_k2/2, w[1,i]+w1_k2/2, w[2,i]+w2_k2/2, q[0,i]+q0_k2/2, q[1,i], q[2,i], q[3,i])
            
            u2_k3 = h*du2dt(w[0,i]+w0_k2/2, w[1,i]+w1_k2/2, w[2,i]+w2_k2/2, q[0,i]+q0_k2/2, q[1,i], q[2,i], q[3,i], Mext[:,i])
            
            p0_k3 = h*dp0dt(p[0,i]+p0_k2/2, p[1,i]+p1_k2/2, p[2,i]+p2_k2/2, p_like[0,i])
            p1_k3 = h*dp1dt(p[0,i]+p0_k2/2, p[1,i]+p1_k2/2, p[2,i]+p2_k2/2, p_like[1,i])
            p2_k3 = h*dp2dt(p[0,i]+p0_k2/2, p[1,i]+p1_k2/2, p[2,i]+p2_k2/2, p_like[2,i])
            
            q0_k3 = h*dq0dt(w[0,i]+w0_k2/2, w[1,i]+w1_k2/2, w[2,i]+w2_k2/2, q[0,i]+q0_k2/2, q[1,i], q[2,i], q[3,i])
            
            """k4"""
            w0_k4 = h*dw_qq_0dt(w[0,i]+w0_k3, w[1,i]+w1_k3, w[2,i]+w2_k3, q[0,i]+q0_k3, q[1,i], q[2,i], q[3,i])
            w1_k4 = h*dw_qq_1dt(w[0,i]+w0_k3, w[1,i]+w1_k3, w[2,i]+w2_k3, q[0,i]+q0_k3, q[1,i], q[2,i], q[3,i])
            w2_k4 = h*dw_qq_2dt(w[0,i]+w0_k3, w[1,i]+w1_k3, w[2,i]+w2_k3, q[0,i]+q0_k3, q[1,i], q[2,i], q[3,i])
            
            u2_k4 = h*du2dt(w[0,i]+w0_k3, w[1,i]+w1_k3, w[2,i]+w2_k3, q[0,i]+q0_k3, q[1,i], q[2,i], q[3,i], Mext[:,i])
            
            p0_k4 = h*dp0dt(p[0,i]+p0_k3, p[1,i]+p1_k3, p[2,i]+p2_k3, p_like[0,i])
            p1_k4 = h*dp1dt(p[0,i]+p0_k3, p[1,i]+p1_k3, p[2,i]+p2_k3, p_like[1,i])
            p2_k4 = h*dp2dt(p[0,i]+p0_k3, p[1,i]+p1_k3, p[2,i]+p2_k3, p_like[2,i])         
            
            q0_k4 = h*dq0dt(w[0,i]+w0_k3, w[1,i]+w1_k3, w[2,i]+w2_k3, q[0,i]+q0_k3, q[1,i], q[2,i], q[3,i])
                       
            """i+1"""
            w[0,i+1] = w[0,i] + 1/6 * (w0_k1 + 2 * w0_k2 + 2 * w0_k3 + 4 * w0_k4)
            w[1,i+1] = w[1,i] + 1/6 * (w1_k1 + 2 * w1_k2 + 2 * w1_k3 + 4 * w1_k4)
            w[2,i+1] = w[2,i] + 1/6 * (w2_k1 + 2 * w2_k2 + 2 * w2_k3 + 4 * w2_k4)
            
            q[0,i+1] = q[0,i] + 1/6 * (q0_k1 + 2 * q0_k2 + 2 * q0_k3 + 4 * q0_k4)
            
            p[0,i+1] = p[0,i] + 1/6 * (p0_k1 + 2 * p0_k2 + 2 * p0_k3 + 4 * p0_k4)
            p[1,i+1] = p[1,i] + 1/6 * (p1_k1 + 2 * p1_k2 + 2 * p1_k3 + 4 * p1_k4)
            p[2,i+1] = p[2,i] + 1/6 * (p2_k1 + 2 * p2_k2 + 2 * p2_k3 + 4 * p2_k4)
            
            u[2,i+1] = u[2,i] + 1/6 * (u2_k1 + 2 * u2_k2 + 2 * u2_k3 + 4 * u2_k4)           
            if i == 0:
                w_h[:,i] = w_h_optimize(u[0,i], u[1,i], u[2,i], u[3,i], w_h[:,0])[:]
            else:
                w_h[:,i] = w_h_optimize(u[0,i], u[1,i], u[2,i], u[3,i], w_h[:,i-1])[:]
            
        self.uu = u
        self.ww_h = w_h
        
        self.ww = w
        self.qq = q
        self.pp = p