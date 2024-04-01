import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import random as rn


class quadrocopter:
    """Класс определения динамики квадрокоптера"""
    def __init__(self, m_qq = 1, m_hl = 0.01, k_L = 1, L = 1, L_hh = 0.1):
        """коэффициент  подъемной силы винта"""
        self.k_L = k_L 
        """ускорение силы тяжести"""
        self.g = -9.8
        """полная масса квадрокоптера"""
        self.m_qq = m_qq
        """масса винта"""
        self.m_hl = m_hl
        """длина плеча"""
        self.L = L
        """длина лопасти винта"""
        self.L_hh = L_hh
        
        """тензор инерции квадрокоптера по умолчанию"""
        self.J_qq = np.array([[(self.L/4)**2*self.m_qq/2, 0, 0],
                              [0, (self.L/4)**2*self.m_qq/2, 0],
                              [0, 0, (self.L/2)**2*self.m_qq/2]])
        
        """тензор инерции винтов по умолчанию, для всех одинаковый"""
        J_hh = np.array([[(self.L_hh/4)**2*self.m_qq/2, 0, 0],
                         [0, (self.L_hh/4)**2*self.m_qq/2, 0],
                         [0, 0, (self.L_hh/2)**2*self.m_qq/2]])
        self.J_hh_0 = J_hh
        self.J_hh_1 = J_hh
        self.J_hh_2 = J_hh
        self.J_hh_3 = J_hh
        
    def inertia_tensor(self, J_qq, J_hh_0, J_hh_1, J_hh_2, J_hh_3):
        """определение моментов инерции квадрокоптера и винтов, по умолчанию это 
        тензоры инерции плоского диска"""
        self.J_qq = J_qq
        self.J_hh_0 = J_hh_0
        self.J_hh_1 = J_hh_1
        self.J_hh_2 = J_hh_2
        self.J_hh_3 = J_hh_3
        
    def perturbation(self, Mext):
        """Определение возмущающих моментов"""
        self.Mextr = Mext
        
    def control(self, w_helix):
        """функции скоростей винтов - задание функций управления"""
        self.w_helixx = w_helix 
        
    def motion_simulation(self, h, N, w_init, q_init, p_init, r_init):
        """Определение шага интегрирования, количества итераций и 
        начальных условий моделирования"""
        
        """Определение векторов угловой скорости, кватернионной части,
        импульса и радикс-вектора"""            
        w = np.zeros((3,N))
        q = np.zeros((4,N))
        p = np.zeros((3,N))
        r = np.zeros((3,N))
        
        Mext = self.Mextr
        w_helix = self.w_helixx
        """Определение углового ускорения, создаваемого винтами"""
        dw_helix = np.zeros((4,len(w_helix[1,:])))
        dw_helix = np.diff(w_helix)/h
        
        """Определение начальных условий"""
        w[:,0] = w_init[:]
        q[:,0] = q_init[:]
        p[:,0] = p_init[:]
        r[:,0] = r_init[:]
                
        self.h = h
        self.N = N        
        """функции правой части динамической системы уравнений"""
        def f0(w_0, w_1, w_2, q_0, q_1, q_2, q_3, p_0, p_1, p_2, Mext, w_helix, dw_helix):
            """соответствует производным по омеге"""
            return ((self.J_qq[1,1]*w_1*w_2 + self.L*self.k_L*w_helix[1]**2/2 - self.L*self.k_L*w_helix[3]**2/2 + Mext[0] - 
                    (self.J_hh_0[2,2]*w_helix[0] + self.J_hh_1[2,2]*w_helix[1] + self.J_hh_2[2,2]*w_helix[2] + 
                     self.J_hh_3[2,2]*w_helix[3] + self.J_qq[2,2]*w_2)*w_1)/self.J_qq[0,0])

        def f1(w_0, w_1, w_2, q_0, q_1, q_2, q_3, p_0, p_1, p_2, Mext, w_helix, dw_helix):
            """соответствует производным по омеге"""
            return ((-self.J_qq[0,0]*w_0*w_2 - self.L*self.k_L*w_helix[0]**2/2 + self.L*self.k_L*w_helix[2]**2/2 + Mext[1] + 
                     (self.J_hh_0[2,2]*w_helix[0] + self.J_hh_1[2,2]*w_helix[1] + self.J_hh_2[2,2]*w_helix[2] + 
                      self.J_hh_3[2,2]*w_helix[3] + self.J_qq[2,2]*w_2)*w_0)/self.J_qq[1,1])

        def f2(w_0, w_1, w_2, q_0, q_1, q_2, q_3, p_0, p_1, p_2, Mext, w_helix, dw_helix):
            """соответствует производным по омеге"""
            return ((-self.J_hh_0[2,2]*dw_helix[0] - self.J_hh_1[2,2]*dw_helix[1] - self.J_hh_2[2,2]*dw_helix[2] - 
                      self.J_hh_3[2,2]*dw_helix[3] + self.J_qq[0,0]*w_0*w_1 - self.J_qq[1,1]*w_0*w_1 + Mext[2])/self.J_qq[2,2])

        def f3(w_0, w_1, w_2, q_0, q_1, q_2, q_3, p_0, p_1, p_2, Mext, w_helix, dw_helix):
            """соответствует производным кватернионной части"""
            return (-0.5*q_1*w_0 - 0.5*q_2*w_1 - 0.5*q_3*w_2)

        def f4(w_0, w_1, w_2, q_0, q_1, q_2, q_3, p_0, p_1, p_2, Mext, w_helix, dw_helix):
            """соответствует производным кватернионной части"""
            return (0.5*q_0*w_0 - 0.5*q_2*w_2 + 0.5*q_3*w_1)

        def f5(w_0, w_1, w_2, q_0, q_1, q_2, q_3, p_0, p_1, p_2, Mext, w_helix, dw_helix):
            """соответствует производным кватернионной части"""
            return (0.5*q_0*w_1 + 0.5*q_1*w_2 - 0.5*q_3*w_0)

        def f6(w_0, w_1, w_2, q_0, q_1, q_2, q_3, p_0, p_1, p_2, Mext, w_helix, dw_helix):
            """соответствует производным кватернионной части"""
            return (0.5*q_0*w_2 - 0.5*q_1*w_1 + 0.5*q_2*w_0)

        def f7(w_0, w_1, w_2, q_0, q_1, q_2, q_3, p_0, p_1, p_2, Mext, w_helix, dw_helix):
            """соответствует производным импульсной части"""
            return (self.k_L*(2*q_0*q_2 + 2*q_1*q_3)*(w_helix[0]**2 + w_helix[1]**2 + w_helix[2]**2 + w_helix[3]**2))

        def f8(w_0, w_1, w_2, q_0, q_1, q_2, q_3, p_0, p_1, p_2, Mext, w_helix, dw_helix):
            """соответствует производным импульсной части"""
            return (self.k_L*(-2*q_0*q_1 + 2*q_2*q_3)*(w_helix[0]**2 + w_helix[1]**2 + w_helix[2]**2 + w_helix[3]**2))

        def f9(w_0, w_1, w_2, q_0, q_1, q_2, q_3, p_0, p_1, p_2, Mext, w_helix, dw_helix):
            """соответствует производным импульсной части"""
            return (self.g*self.m_qq + 
                    self.k_L*(w_helix[0]**2 + w_helix[1]**2 + w_helix[2]**2 + w_helix[3]**2)*(q_0**2 - q_1**2 - q_2**2 + q_3**2))
    
        def coord_x(p_x):
            return p_x/self.m_qq 

        def coord_y(p_y):
            return p_y/self.m_qq 

        def coord_z(p_z):
            return p_z/self.m_qq 

        for i in range(0, N-1):
            """метод Рунге-Кутты 4-го порядка"""           
            w0_k1 = h*f0(w[0,i], w[1,i], w[2,i], q[0,i], q[1,i], q[2,i], q[3,i],
                         p[0,i], p[1,i], p[2,i], Mext[:,i], w_helix[:,i], dw_helix[:,i])
            w1_k1 = h*f1(w[0,i], w[1,i], w[2,i], q[0,i], q[1,i], q[2,i], q[3,i],
                         p[0,i], p[1,i], p[2,i], Mext[:,i], w_helix[:,i], dw_helix[:,i])
            w2_k1 = h*f2(w[0,i], w[1,i], w[2,i], q[0,i], q[1,i], q[2,i], q[3,i],
                         p[0,i], p[1,i], p[2,i], Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q0_k1 = h*f3(w[0,i], w[1,i], w[2,i], q[0,i], q[1,i], q[2,i], q[3,i],
                         p[0,i], p[1,i], p[2,i], Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q1_k1 = h*f4(w[0,i], w[1,i], w[2,i], q[0,i], q[1,i], q[2,i], q[3,i],
                         p[0,i], p[1,i], p[2,i], Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q2_k1 = h*f5(w[0,i], w[1,i], w[2,i], q[0,i], q[1,i], q[2,i], q[3,i],
                         p[0,i], p[1,i], p[2,i], Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q3_k1 = h*f6(w[0,i], w[1,i], w[2,i], q[0,i], q[1,i], q[2,i], q[3,i],
                         p[0,i], p[1,i], p[2,i], Mext[:,i], w_helix[:,i], dw_helix[:,i])
            p0_k1 = h*f7(w[0,i], w[1,i], w[2,i], q[0,i], q[1,i], q[2,i], q[3,i],
                         p[0,i], p[1,i], p[2,i], Mext[:,i], w_helix[:,i], dw_helix[:,i])
            p1_k1 = h*f8(w[0,i], w[1,i], w[2,i], q[0,i], q[1,i], q[2,i], q[3,i],
                         p[0,i], p[1,i], p[2,i], Mext[:,i], w_helix[:,i], dw_helix[:,i])
            p2_k1 = h*f9(w[0,i], w[1,i], w[2,i], q[0,i], q[1,i], q[2,i], q[3,i],
                         p[0,i], p[1,i], p[2,i], Mext[:,i], w_helix[:,i], dw_helix[:,i])           
            
            w0_k2 = h*f0(w[0,i]+w0_k1/2, w[1,i]+w1_k1/2, w[2,i]+w2_k1/2, q[0,i]+q0_k1/2, q[1,i]+q1_k1/2, q[2,i]+q2_k1/2,
                         q[3,i]+q3_k1/2, p[0,i]+p0_k1/2, p[1,i]+p1_k1/2, p[2,i]+p2_k1/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            w1_k2 = h*f1(w[0,i]+w0_k1/2, w[1,i]+w1_k1/2, w[2,i]+w2_k1/2, q[0,i]+q0_k1/2, q[1,i]+q1_k1/2, q[2,i]+q2_k1/2,
                         q[3,i]+q3_k1/2, p[0,i]+p0_k1/2, p[1,i]+p1_k1/2, p[2,i]+p2_k1/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            w2_k2 = h*f2(w[0,i]+w0_k1/2, w[1,i]+w1_k1/2, w[2,i]+w2_k1/2, q[0,i]+q0_k1/2, q[1,i]+q1_k1/2, q[2,i]+q2_k1/2,
                         q[3,i]+q3_k1/2, p[0,i]+p0_k1/2, p[1,i]+p1_k1/2, p[2,i]+p2_k1/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q0_k2 = h*f3(w[0,i]+w0_k1/2, w[1,i]+w1_k1/2, w[2,i]+w2_k1/2, q[0,i]+q0_k1/2, q[1,i]+q1_k1/2, q[2,i]+q2_k1/2,
                         q[3,i]+q3_k1/2, p[0,i]+p0_k1/2, p[1,i]+p1_k1/2, p[2,i]+p2_k1/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q1_k2 = h*f4(w[0,i]+w0_k1/2, w[1,i]+w1_k1/2, w[2,i]+w2_k1/2, q[0,i]+q0_k1/2, q[1,i]+q1_k1/2, q[2,i]+q2_k1/2,
                         q[3,i]+q3_k1/2, p[0,i]+p0_k1/2, p[1,i]+p1_k1/2, p[2,i]+p2_k1/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q2_k2 = h*f5(w[0,i]+w0_k1/2, w[1,i]+w1_k1/2, w[2,i]+w2_k1/2, q[0,i]+q0_k1/2, q[1,i]+q1_k1/2, q[2,i]+q2_k1/2,
                         q[3,i]+q3_k1/2, p[0,i]+p0_k1/2, p[1,i]+p1_k1/2, p[2,i]+p2_k1/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q3_k2 = h*f6(w[0,i]+w0_k1/2, w[1,i]+w1_k1/2, w[2,i]+w2_k1/2, q[0,i]+q0_k1/2, q[1,i]+q1_k1/2, q[2,i]+q2_k1/2,
                         q[3,i]+q3_k1/2, p[0,i]+p0_k1/2, p[1,i]+p1_k1/2, p[2,i]+p2_k1/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            p0_k2 = h*f7(w[0,i]+w0_k1/2, w[1,i]+w1_k1/2, w[2,i]+w2_k1/2, q[0,i]+q0_k1/2, q[1,i]+q1_k1/2, q[2,i]+q2_k1/2,
                         q[3,i]+q3_k1/2, p[0,i]+p0_k1/2, p[1,i]+p1_k1/2, p[2,i]+p2_k1/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            p1_k2 = h*f8(w[0,i]+w0_k1/2, w[1,i]+w1_k1/2, w[2,i]+w2_k1/2, q[0,i]+q0_k1/2, q[1,i]+q1_k1/2, q[2,i]+q2_k1/2,
                         q[3,i]+q3_k1/2, p[0,i]+p0_k1/2, p[1,i]+p1_k1/2, p[2,i]+p2_k1/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            p2_k2 = h*f9(w[0,i]+w0_k1/2, w[1,i]+w1_k1/2, w[2,i]+w2_k1/2, q[0,i]+q0_k1/2, q[1,i]+q1_k1/2, q[2,i]+q2_k1/2,
                         q[3,i]+q3_k1/2, p[0,i]+p0_k1/2, p[1,i]+p1_k1/2, p[2,i]+p2_k1/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            
            w0_k3 = h*f0(w[0,i]+w0_k2/2, w[1,i]+w1_k2/2, w[2,i]+w2_k2/2, q[0,i]+q0_k2/2, q[1,i]+q1_k2/2, q[2,i]+q2_k2/2,
                         q[3,i]+q3_k2/2, p[0,i]+p0_k2/2, p[1,i]+p1_k2/2, p[2,i]+p2_k2/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            w1_k3 = h*f1(w[0,i]+w0_k2/2, w[1,i]+w1_k2/2, w[2,i]+w2_k2/2, q[0,i]+q0_k2/2, q[1,i]+q1_k2/2, q[2,i]+q2_k2/2,
                         q[3,i]+q3_k2/2, p[0,i]+p0_k2/2, p[1,i]+p1_k2/2, p[2,i]+p2_k2/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            w2_k3 = h*f2(w[0,i]+w0_k2/2, w[1,i]+w1_k2/2, w[2,i]+w2_k2/2, q[0,i]+q0_k2/2, q[1,i]+q1_k2/2, q[2,i]+q2_k2/2,
                         q[3,i]+q3_k2/2, p[0,i]+p0_k2/2, p[1,i]+p1_k2/2, p[2,i]+p2_k2/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q0_k3 = h*f3(w[0,i]+w0_k2/2, w[1,i]+w1_k2/2, w[2,i]+w2_k2/2, q[0,i]+q0_k2/2, q[1,i]+q1_k2/2, q[2,i]+q2_k2/2,
                         q[3,i]+q3_k2/2, p[0,i]+p0_k2/2, p[1,i]+p1_k2/2, p[2,i]+p2_k2/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q1_k3 = h*f4(w[0,i]+w0_k2/2, w[1,i]+w1_k2/2, w[2,i]+w2_k2/2, q[0,i]+q0_k2/2, q[1,i]+q1_k2/2, q[2,i]+q2_k2/2,
                         q[3,i]+q3_k2/2, p[0,i]+p0_k2/2, p[1,i]+p1_k2/2, p[2,i]+p2_k2/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q2_k3 = h*f5(w[0,i]+w0_k2/2, w[1,i]+w1_k2/2, w[2,i]+w2_k2/2, q[0,i]+q0_k2/2, q[1,i]+q1_k2/2, q[2,i]+q2_k2/2,
                         q[3,i]+q3_k2/2, p[0,i]+p0_k2/2, p[1,i]+p1_k2/2, p[2,i]+p2_k2/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q3_k3 = h*f6(w[0,i]+w0_k2/2, w[1,i]+w1_k2/2, w[2,i]+w2_k2/2, q[0,i]+q0_k2/2, q[1,i]+q1_k2/2, q[2,i]+q2_k2/2,
                         q[3,i]+q3_k2/2, p[0,i]+p0_k2/2, p[1,i]+p1_k2/2, p[2,i]+p2_k2/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            p0_k3 = h*f7(w[0,i]+w0_k2/2, w[1,i]+w1_k2/2, w[2,i]+w2_k2/2, q[0,i]+q0_k2/2, q[1,i]+q1_k2/2, q[2,i]+q2_k2/2,
                         q[3,i]+q3_k2/2, p[0,i]+p0_k2/2, p[1,i]+p1_k2/2, p[2,i]+p2_k2/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            p1_k3 = h*f8(w[0,i]+w0_k2/2, w[1,i]+w1_k2/2, w[2,i]+w2_k2/2, q[0,i]+q0_k2/2, q[1,i]+q1_k2/2, q[2,i]+q2_k2/2,
                         q[3,i]+q3_k2/2, p[0,i]+p0_k2/2, p[1,i]+p1_k2/2, p[2,i]+p2_k2/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            p2_k3 = h*f9(w[0,i]+w0_k2/2, w[1,i]+w1_k2/2, w[2,i]+w2_k2/2, q[0,i]+q0_k2/2, q[1,i]+q1_k2/2, q[2,i]+q2_k2/2,
                         q[3,i]+q3_k2/2, p[0,i]+p0_k2/2, p[1,i]+p1_k2/2, p[2,i]+p2_k2/2, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            
            w0_k4 = h*f0(w[0,i]+w0_k3, w[1,i]+w1_k3, w[2,i]+w2_k3, q[0,i]+q0_k3, q[1,i]+q1_k3, q[2,i]+q2_k3,
                         q[3,i]+q3_k3, p[0,i]+p0_k3, p[1,i]+p1_k3, p[2,i]+p2_k3, Mext[:,i], w_helix[:,i], dw_helix[:,i])            
            w1_k4 = h*f1(w[0,i]+w0_k3, w[1,i]+w1_k3, w[2,i]+w2_k3, q[0,i]+q0_k3, q[1,i]+q1_k3, q[2,i]+q2_k3,
                         q[3,i]+q3_k3, p[0,i]+p0_k3, p[1,i]+p1_k3, p[2,i]+p2_k3, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            w2_k4 = h*f2(w[0,i]+w0_k3, w[1,i]+w1_k3, w[2,i]+w2_k3, q[0,i]+q0_k3, q[1,i]+q1_k3, q[2,i]+q2_k3,
                         q[3,i]+q3_k3, p[0,i]+p0_k3, p[1,i]+p1_k3, p[2,i]+p2_k3, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q0_k4 = h*f3(w[0,i]+w0_k3, w[1,i]+w1_k3, w[2,i]+w2_k3, q[0,i]+q0_k3, q[1,i]+q1_k3, q[2,i]+q2_k3,
                         q[3,i]+q3_k3, p[0,i]+p0_k3, p[1,i]+p1_k3, p[2,i]+p2_k3, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q1_k4 = h*f4(w[0,i]+w0_k3, w[1,i]+w1_k3, w[2,i]+w2_k3, q[0,i]+q0_k3, q[1,i]+q1_k3, q[2,i]+q2_k3,
                         q[3,i]+q3_k3, p[0,i]+p0_k3, p[1,i]+p1_k3, p[2,i]+p2_k3, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q2_k4 = h*f5(w[0,i]+w0_k3, w[1,i]+w1_k3, w[2,i]+w2_k3, q[0,i]+q0_k3, q[1,i]+q1_k3, q[2,i]+q2_k3,
                         q[3,i]+q3_k3, p[0,i]+p0_k3, p[1,i]+p1_k3, p[2,i]+p2_k3, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            q3_k4 = h*f6(w[0,i]+w0_k3, w[1,i]+w1_k3, w[2,i]+w2_k3, q[0,i]+q0_k3, q[1,i]+q1_k3, q[2,i]+q2_k3,
                         q[3,i]+q3_k3, p[0,i]+p0_k3, p[1,i]+p1_k3, p[2,i]+p2_k3, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            p0_k4 = h*f7(w[0,i]+w0_k3, w[1,i]+w1_k3, w[2,i]+w2_k3, q[0,i]+q0_k3, q[1,i]+q1_k3, q[2,i]+q2_k3,
                         q[3,i]+q3_k3, p[0,i]+p0_k3, p[1,i]+p1_k3, p[2,i]+p2_k3, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            p1_k4 = h*f8(w[0,i]+w0_k3, w[1,i]+w1_k3, w[2,i]+w2_k3, q[0,i]+q0_k3, q[1,i]+q1_k3, q[2,i]+q2_k3,
                         q[3,i]+q3_k3, p[0,i]+p0_k3, p[1,i]+p1_k3, p[2,i]+p2_k3, Mext[:,i], w_helix[:,i], dw_helix[:,i])
            p2_k4 = h*f9(w[0,i]+w0_k3, w[1,i]+w1_k3, w[2,i]+w2_k3, q[0,i]+q0_k3, q[1,i]+q1_k3, q[2,i]+q2_k3,
                         q[3,i]+q3_k3, p[0,i]+p0_k3, p[1,i]+p1_k3, p[2,i]+p2_k3, Mext[:,i], w_helix[:,i], dw_helix[:,i])
               
            w[0,i+1] = w[0,i] + 1/6 * (w0_k1 + 2 * w0_k2 + 2 * w0_k3 + 4 * w0_k4)
            w[1,i+1] = w[1,i] + 1/6 * (w1_k1 + 2 * w1_k2 + 2 * w1_k3 + 4 * w1_k4)
            w[2,i+1] = w[2,i] + 1/6 * (w2_k1 + 2 * w2_k2 + 2 * w2_k3 + 4 * w2_k4)
            
            q[0,i+1] = q[0,i] + 1/6 * (q0_k1 + 2 * q0_k2 + 2 * q0_k3 + 4 * q0_k4)
            q[1,i+1] = q[1,i] + 1/6 * (q1_k1 + 2 * q1_k2 + 2 * q1_k3 + 4 * q1_k4)
            q[2,i+1] = q[2,i] + 1/6 * (q2_k1 + 2 * q2_k2 + 2 * q2_k3 + 4 * q2_k4)
            q[3,i+1] = q[3,i] + 1/6 * (q3_k1 + 2 * q3_k2 + 2 * q3_k3 + 4 * q3_k4)
            """перенормировка кватерниона"""
            q[0,i+1] = q[0,i+1] / (np.sqrt(q[0,i+1]**2 + q[1,i+1]**2 + q[2,i+1]**2 + q[3,i+1]**2))
            q[1,i+1] = q[1,i+1] / (np.sqrt(q[0,i+1]**2 + q[1,i+1]**2 + q[2,i+1]**2 + q[3,i+1]**2))
            q[2,i+1] = q[2,i+1] / (np.sqrt(q[0,i+1]**2 + q[1,i+1]**2 + q[2,i+1]**2 + q[3,i+1]**2))
            q[3,i+1] = q[3,i+1] / (np.sqrt(q[0,i+1]**2 + q[1,i+1]**2 + q[2,i+1]**2 + q[3,i+1]**2))
            
            p[0,i+1] = p[0,i] + 1/6 * (p0_k1 + 2 * p0_k2 + 2 * p0_k3 + 4 * p0_k4)
            p[1,i+1] = p[1,i] + 1/6 * (p1_k1 + 2 * p1_k2 + 2 * p1_k3 + 4 * p1_k4)
            p[2,i+1] = p[2,i] + 1/6 * (p2_k1 + 2 * p2_k2 + 2 * p2_k3 + 4 * p2_k4)
            
            x_k1 = h*coord_x(p[0,i])
            x_k2 = h*coord_x(p[0,i]+x_k1/2)
            x_k3 = h*coord_x(p[0,i]+x_k2/2)
            x_k4 = h*coord_x(p[0,i]+x_k3)
            r[0,i+1] = r[0,i] + 1/6 * (x_k1 + 2 * x_k2 + 2 * x_k3 + x_k4)
            
            y_k1 = h*coord_y(p[1,i])
            y_k2 = h*coord_y(p[1,i]+y_k1/2)
            y_k3 = h*coord_y(p[1,i]+y_k2/2)
            y_k4 = h*coord_y(p[1,i]+y_k3)
            r[1,i+1] = r[1,i] + 1/6 * (y_k1 + 2 * y_k2 + 2 * y_k3 + y_k4)
            
            z_k1 = h*coord_z(p[2,i])
            z_k2 = h*coord_z(p[2,i]+z_k1/2)
            z_k3 = h*coord_z(p[2,i]+z_k2/2)
            z_k4 = h*coord_z(p[2,i]+z_k3)
            r[2,i+1] = r[2,i] + 1/6 * (z_k1 + 2 * z_k2 + 2 * z_k3 + z_k4)
    
        self.ww = w
        self.qq = q
        self.pp = p
        self.rr = r
            
    def simulation_animate(self):
        
        """создание фигуры креста квадракоптера """
        X = np.array([1/2,   1, 1/2,   0, 1/2,  1/2, 1/2, 1/2])
        Y = np.array([1/2, 1/2, 1/2, 1/2, 1/2,    1, 1/2,   0])
        Z = np.array([  0,   0,   0,   0,   0,    0,   0,   0])
        
        R = np.array([X-1/2,
                      Y-1/2,
                      Z])
        """отрисовка нормали"""
        normal = np.array([[0, 0],
                           [0, 0],
                           [0, 1/4]])
        
        self.OxyzR = np.zeros((3,len(X), self.N))
        self.Oxyz_normal = np.zeros((3,2, self.N))

        """отображение поворота модели"""
        for i in range(0, self.N):

            q0 = self.qq[0,i]
            q1 = self.qq[1,i]
            q2 = self.qq[2,i]
            q3 = self.qq[3,i]
            """расчет матрицы направляющих косинусов через кватернионы"""
            A = np.array([[q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q0*q2 + q1*q3)],
                          [2*(q0*q3 + q1*q2), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
                          [2*(q1*q3 - q0*q2), 2*(q0*q1 + q2*q3), q0**2 - q1**2 - q2**2 + q3**2]])
            """поворот как действие оператора"""
            self.OxyzR[:,:,i] = A.dot(R)
            self.Oxyz_normal[:,:,i] = A.dot(normal)

        
        def animate_func_ang(num):
            num = 10*num
            """анимация углового движения"""
            self.ax1.clear() 
            """Движение тела"""
            self.ax1.plot3D(self.OxyzR[0,:,num], self.OxyzR[1,:,num], self.OxyzR[2,:,num], c='blue')
            self.ax1.text2D(0.05, 0.95, "Поворот от начального положения",color='blue', transform=self.ax1.transAxes)
            """нормаль"""
            self.ax1.plot3D(self.Oxyz_normal[0,:,num], self.Oxyz_normal[1,:,num], self.Oxyz_normal[2,:,num], c='red')
            self.ax1.text2D(0.05, 0.9, "Нормаль",color='red', transform=self.ax1.transAxes)
            """Точки начального положения"""
            self.ax1.plot3D(self.OxyzR[0,:,0], self.OxyzR[1,:,0], self.OxyzR[2,:,0], c='black')
            self.ax1.text2D(0.05, 0.85, "Точки начального положения",color='black', transform=self.ax1.transAxes)
            self.ax1.set_xlim3d([-1, 1])
            self.ax1.set_ylim3d([-1, 1])
            self.ax1.set_zlim3d([-1, 1])
            """Добавляем метки"""
            self.ax1.set_title('rotation \nTime = ' + str((float('{:.5f}'.format(num*self.h)))) + ' sec')
            self.ax1.set_xlabel('X Label')
            self.ax1.set_ylabel('Y Label')
            self.ax1.set_zlabel('Z Label')
            
        def animate_func_tr(num):
            num = 10*num
            """анимация движения центральной точки"""
            self.ax2.clear()              
            self.ax2.plot3D(self.rr[0,:num+1], self.rr[1,:num+1],
                     self.rr[2,:num+1], c='blue')                       
            self.ax2.scatter(self.rr[0, num], self.rr[1,num],
                       self.rr[2, num], 
                      c='blue', marker='o')   
            """Добавляем постоянную начальную точку"""
            self.ax2.plot3D(self.rr[0, 0], self.rr[1,0],
                      self.rr[2, 0],    
                      c='black', marker='o')      
            
            self.ax2.set_xlim3d([-5, 5])
            self.ax2.set_ylim3d([-5, 5])
            self.ax2.set_zlim3d([0, 5])
            """Добавляем метки"""
            self.ax2.set_title('Trajectory \nTime = ' + str((float('{:.5f}'.format(num*self.h)))) + ' sec')
            self.ax2.set_xlabel('X Label')
            self.ax2.set_ylabel('Y Label')
            self.ax2.set_zlabel('Z Label')
  
        """отрисовка анимации"""
        self.fig = plt.figure(figsize=(10, 10))
        self.ax1 = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.line_ani_1 = animation.FuncAnimation(self.fig, animate_func_ang, interval=1, frames=round(self.N/10))
        self.ax2 = self.fig.add_subplot(1, 2, 2, projection='3d')
        self.line_ani_2 = animation.FuncAnimation(self.fig, animate_func_tr, interval=1, frames=round(self.N/10))
        plt.show()
        
    def motion_pattern_graph(self):
        """вывод всех рассчитанных  функций движения"""
        t = np.linspace(0, self.h*self.N, self.N)
        self.fig2 = plt.figure()
        """Под график для Угловые скорости""" 
        self.ax3 = self.fig2.add_subplot(2, 2, 1)
        self.ax3.plot(t, self.ww[0,:], label=r'$W_{0}$')
        self.ax3.plot(t, self.ww[1,:], label=r'$W_{1}$')
        self.ax3.plot(t, self.ww[2,:], label=r'$W_{2}$')
        self.ax3.legend()
        self.ax3.set_title('Угловые скорости')
        self.ax3.set_xlabel(r't')
        """Под график для координат центральной точки"""
        self.ax4 = self.fig2.add_subplot(2, 2, 2)
        self.ax4.plot(t, self.rr[0,:], label=r'$r_{x}$')
        self.ax4.plot(t, self.rr[1,:], label=r'$r_{y}$')
        self.ax4.plot(t, self.rr[2,:], label=r'$r_{z}$')
        self.ax4.legend()
        self.ax4.set_title('Координаты')
        self.ax4.set_xlabel(r't')
        """Под график для Импульсов"""
        self.ax5 = self.fig2.add_subplot(2, 2, 3)
        self.ax5.plot(t, self.pp[0,:], label=r'$p_{x}$')
        self.ax5.plot(t, self.pp[1,:], label=r'$p_{y}$')
        self.ax5.plot(t, self.pp[2,:], label=r'$p_{z}$')
        self.ax5.legend()
        self.ax5.set_title('Импульсы')
        self.ax5.set_xlabel(r't')
        """Под график для кватернионов"""
        self.ax6 = self.fig2.add_subplot(2, 2, 4)
        self.ax6.plot(t, self.qq[0,:], label=r'$q_{0}$')
        self.ax6.plot(t, self.qq[1,:], label=r'$q_{1}$')
        self.ax6.plot(t, self.qq[2,:], label=r'$q_{2}$')
        self.ax6.plot(t, self.qq[3,:], label=r'$q_{3}$')
        self.ax6.legend()
        self.ax6.set_title('кватернионная часть')
        self.ax6.set_xlabel(r't')
        plt.show()
            
            
        
        
    