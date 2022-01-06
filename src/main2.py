import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import lower
from numpy.lib.polynomial import polyfit
from mathutils import gaussian
import argparse
import scipy.optimize as sciopt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
# import pylab as pl
# import colorlover as cl
import matplotlib.collections as mc
import matplotlib

#from config import MU_1, SIGMA_1, MU_2, SIGMA_2, N_POINTS
#from scipy.optimize import newton


## distribution initiale
# MU_1 = 4
# SIGMA_1 = .7
# MU_2 = 12
# SIGMA_2 = .7
L_INTERVALLE =16
EPS = 0.1
PREC = 10 ** (-8)
PREC2 = 10 ** (-6)


N_POINTS = 500
L_INTERVALLE =16
EPS = 0.1
PEN_DISTANCE = 1
PEN_DENSITY = 10 #10 ** 4 ## MAX=10
MAX_ATTRACTIVENESS = 1
REWARD_ATTRACTIVENESS = 1
N_PEAKS=10

def newton(f, x0, jacobian, eps, args=()):
    x=x0
    f_value = f(x, *args)
    f_norm = np.linalg.norm(f_value, ord=2)  # l2 norm of vector
    n_iter = 0

    # plt.plot(f_value)
    # plt.show()
    while abs(f_norm) > (eps / 2) and n_iter < 100:

        delta = np.linalg.solve(jacobian(x, *args), -f_value)
        x = x + delta
        # if n_iter == 10:
        #     plt.imshow(jacobian(x, *args))
        #     plt.show()
        # if n_iter == 11:
        #     plt.imshow(jacobian(x, *args))
        #     plt.show()
        # print("min_delta", np.min(delta))
        f_value = f(x, *args)
        # print("borne", np.min(f_value), np.max(f_value), np.min(jacobian(x, *args)), np.max(jacobian(x, *args)))
        # print("step", np.linalg.norm(f_value - delta))
        f_norm = np.linalg.norm(f_value, ord=2)
        # print("norm", f_norm, )
        n_iter += 1

    # Here, either a solution is found, or too many iterations
    if abs(f_norm) > eps:
        # n_iter = -1
        print("Newton failed to converge")

    # plt.plot(x)
    # plt.show()
    return x#, n_iter

def newton_diagonal(f, x0, jacobian, eps, args=()):
    x=x0
    f_value = f(x, *args)
    f_norm = np.linalg.norm(f_value, ord=2)  # l2 norm of vector
    n_iter = 0
    while abs(f_norm) > (eps / 2) and n_iter < 100:

        delta = -f_value / np.diagonal(jacobian(x, *args)).reshape(-1, 1)
        x = x + delta

        # print("borne", np.min(f_value), np.max(f_value), np.min(jacobian(x, *args)), np.max(jacobian(x, *args)))

        print("min_delta", np.min(delta))
        f_value = f(x, *args)

        # print("step", np.linalg.norm(f_value - delta))
        f_norm = np.linalg.norm(f_value, ord=2)
        print("NORM : ", f_norm)
        print("ECART : ", np.linalg.norm(jacobian(x, *args) - jacobian(x - delta, *args), ord=1), np.linalg.norm(delta))
        n_iter += 1

    # Here, either a solution is found, or too many iterations
    if abs(f_norm) > eps:
        # n_iter = -1
        print("Newton failed to converge")
    return x#, n_iter

def dichotomy(f, lowerb, upperb, prec, args=()):
    f_middles = f(lowerb, *args)
    n_iter=0
    #print("ASSERT", np.max(f(lowerb, *args)) <= 0, np.min(f(upperb, *args)) >= 0, np.max(f(lowerb, *args)), np.min(f(upperb, *args)))
    while (np.linalg.norm(lowerb - upperb) > prec or np.linalg.norm(f_middles) > prec) and n_iter < 1000:
        n_iter += 1
        middles = (lowerb + upperb) / 2
        f_middles = f(middles, *args)
        upperb = np.where(f_middles >= 0, middles, upperb)
        lowerb = np.where(f_middles <= 0, middles, lowerb)
    # print(np.sum((lowerb + upperb) / 2))
    return (lowerb + upperb) / 2


class ConvexProximalMethod():

    def __init__(self, mu, costs, eps, capacity):
        self.mu = mu.reshape(-1, 1)
        self.costs = costs
        self.eps = eps
        self.capacity = capacity

    def prox_G1_KL(self, theta):
        return theta * self.mu * 1 / np.sum(theta, axis=1).reshape(-1, 1) #(self.mu.T @ theta).reshape(-1, 1) / np.sum(theta, axis=1).reshape(-1, 1) #TODO: vérifier dimension

    #prox 2
    def f_G2(self, nu, theta):
        # nu vecteur vertical
        return nu - np.sum(theta, axis=0).reshape(nu.shape) * np.exp( - (PEN_DENSITY * nu + self.phi.T @ nu) / self.eps)

    def jacobian_f_G2(self, nu, theta):  
        #nu est vertical
        A = np.sum(theta, axis=0).reshape(nu.shape) * np.exp(- (PEN_DENSITY * nu + self.phi.T @ nu) / self.eps)
        return np.eye(nu.size) + ((PEN_DENSITY * np.eye(nu.size) + self.phi) / self.eps) * np.tile(A, (1, nu.size))

    def prox_G2_KL(self, theta):  
        nu = np.sum(theta, axis=0)
        return theta * np.minimum(nu , self.capacity.reshape(1, theta.shape[1])) / nu#

    #Prox cyclique
    def prox_Gn_KL(self, n, theta, L=2):
        if n % L == 1:
            return self.prox_G1_KL(theta)

        elif n % L == 0:
            return self.prox_G2_KL(theta)

    #PROXIMAL ALGO
    def fit(self):
        L = 2
        gamma = np.exp(-self.costs / self.eps)
        z = [np.ones((self.mu.size, self.costs.shape[1])) for _ in range(L)]
        gap = np.inf
        n = 1
        break_loop = False
        mem_gap = []
        mem_nu = []
        while not break_loop: # todo : trouver une condition sur le gap qui ne soit pas dépendante du nombre de points
            #print(n, flush=True)
            new_gamma = self.prox_Gn_KL(n, gamma * z[n % L], L)
            #new_gamma = np.maximum(new_gamma, (10 ** -300))
            z[n % L] = z[n % L] * (gamma / new_gamma)
            #resume(new_gamma)
            gap = np.linalg.norm(np.sum(gamma, axis=0) - np.sum(new_gamma, axis=0), ord=1)
            mem_nu.append(np.sum(new_gamma, axis=0))
            mem_gap.append(np.log(gap))
            if n%L == 1:
                break_loop = gap < PREC2
                print("gap :", gap)
                #old_gamma = gamma
            n += 1
            gamma = new_gamma
        
        return gamma, np.sum(gamma, axis=0), mem_nu, mem_gap

class NonConvexProximalMethod():

    def __init__(self, mu, costs, eps, phi, capacity):
        self.mu = mu.reshape(-1, 1)
        self.costs = costs
        self.eps = eps
        self.phi = phi
        self.capacity = capacity

    def prox_G1_KL(self, theta):
        return theta * self.mu * 1 / np.sum(theta, axis=1).reshape(-1, 1)

    def f_G2(self, nu, theta):
        return nu - np.sum(theta, axis=0).reshape(nu.shape) * np.exp( - (PEN_DENSITY * nu + self.phi.T @ nu) / self.eps)

    def jacobian_f_G2(self, nu, theta):  
        #nu est vertical
        A = np.sum(theta, axis=0).reshape(nu.shape) * np.exp(- (PEN_DENSITY * nu + self.phi.T @ nu) / self.eps)
        return np.eye(nu.size) + ((PEN_DENSITY * np.eye(nu.size) + self.phi) / self.eps) * np.tile(A, (1, nu.size))

    def prox_G2_KL(self, theta):  
        nu = np.sum(theta, axis=0)
        return theta * np.minimum(nu , self.capacity.reshape(1, theta.shape[1])) / nu#

    #Prox cyclique
    def prox_Gn_KL(self, n, theta, L=2):
        if n % L == 1:
            return self.prox_G1_KL(theta)

        elif n % L == 0:
            return self.prox_G2_KL(theta)
    
    #PROXIMAL ALGO
    def proximal_scheme(self, costs):
        L = 2
        gamma = np.exp(-costs / self.eps)

        z = [np.ones((self.mu.size, self.costs.shape[1])) for _ in range(L)]
        gap = np.inf
        n = 1
        break_loop = False
        # mem_gap = []
        # mem_nu = []

        while not break_loop  and n < 30: # todo : trouver une condition sur le gap qui ne soit pas dépendante du nombre de points
            # print(n, flush=True)
            new_gamma = self.prox_Gn_KL(n, gamma * z[n % L], L)
            #new_gamma = np.maximum(new_gamma, (10 ** -300))
            z[n % L] = z[n % L] * (gamma / new_gamma)

            gap = np.linalg.norm(np.sum(gamma, axis=0) - np.sum(new_gamma, axis=0), ord=1)
            # mem_nu.append(np.sum(new_gamma, axis=0))
            # mem_gap.append(np.log(gap))
            if n%L == 1:
                break_loop = gap < PREC2
                # print("gap :", gap)

            n += 1

            gamma = new_gamma

        resume(new_gamma)
        return gamma, np.sum(gamma, axis=0)#, mem_nu, mem_gap

    def fit(self):
        nu = np.ones((self.costs.shape[1], 1))
        nu = nu / np.sum(nu)
        gap = np.inf
        n_iter = 1
        mem_gap = []
        mem_nu = []
        mem_potential = []
        while gap > PREC2 and n_iter < 100:
            print("ITER", n_iter)
            #print(n_iter)
            new_costs = self.costs + np.tile(nu.T @ self.phi, (self.costs.shape[0], 1))
            gamma, new_nu = self.proximal_scheme(new_costs)
            gap = np.linalg.norm(new_nu - nu, ord=1)
            nu = new_nu
            print("GAP :", gap)
            n_iter += 1

            mem_nu.append(nu)
            mem_gap.append(np.log(gap))
            mem_potential.append(nu.T @ self.phi)

            # if True:
            #     fig2 = plt.figure(f"Solution finale", figsize=(15, 10))
            #     ax2 = fig2.add_subplot(111)
            #     ax2.plot(nu.T @ self.phi, label="potential")

            # if True:
            #     y =  np.linspace(0, L_INTERVALLE, N_POINTS)
            #     attract_y = gaussian(L_INTERVALLE / 2, L_INTERVALLE / 20, y)
            #     attract_y = MAX_ATTRACTIVENESS * attract_y / np.max(attract_y)

            #     fig1 = plt.figure(f"Solution finale", figsize=(15, 10))
            #     ax1 = fig1.add_subplot(111)

            #     color3 = "grey"
            #     ax1.plot(self.capacity* N_POINTS / L_INTERVALLE, label="building limit", color=color3, linewidth=1.7)

            #     color1 = "blue"
            #     ax1.plot(nu * N_POINTS / L_INTERVALLE, label=r"$\nu$", color=color1, linewidth=1.7)
            #     ax1.grid(True)
            #     ax1.tick_params(axis='y', labelcolor=color1)
            #     # ax1.set_xlabel(r"$p_{dens} = $" + f"{PEN_DENSITY}")
            #     ax1.set_ylabel("density", color=color1)
            #     ax1.set_ylim(0,1)
            #     ax1.legend(loc="upper right")
                

            #     color2 = "forestgreen"
            #     ax2 = ax1.twinx()
            #     ax2.plot(attract_y, label="attractiveness", color=color2, linewidth=1.7)
            #     #ax2.plot(y, capacity, label="building zones", color=color2, linewidth=1.7)
            #     ax2.set_ylabel('attractiveness', color=color2)
            #     ax2.tick_params(axis='y', labelcolor=color2)
            #     ax1.legend(loc="upper left")
            #     ax1.set_title("Distribution of the agents according to the capacity and the attractiveness")
            #     fig1.tight_layout()
            #     plt.legend()
            #     plt.show()

        return gamma, nu, mem_nu, mem_gap, mem_potential

def ComputeGamma(max_attractiveness, reward_attractiveness, pen_distance, n_points, epsilon, l_intervalle, n_peaks):
    x = np.linspace(0, l_intervalle, n_points)
    # mu = gaussian(MU_1, SIGMA_1, x) + gaussian(MU_2, SIGMA_2, x)
    # mu = (mu / np.linalg.norm(mu, ord=1))

    mu = np.ones(x.shape) / x.shape

    y =  np.linspace(0, l_intervalle, n_points) # TODO: tester avec x et y de taille différente 
    attract_y = gaussian(l_intervalle / 2, l_intervalle / 20, y)
    #attract_y = #gaussian(MU_1, SIGMA_1, y) + gaussian(MU_2, SIGMA_2, y)
    attract_y = max_attractiveness * attract_y / np.max(attract_y)
    
    districts = n_points / n_peaks
    capacity = np.where(np.logical_and((n_points * x / l_intervalle) % districts >= (districts/4), (n_points * x / l_intervalle) % districts <=(3*districts/4)), 1, 10 ** (-4))
    capacity = 2*capacity / np.sum(capacity)
    
    # print("HELLO", np.sum(capacity))#capacity)#(n_points * x / l_intervalle) % 100)
    
    #gaussian(l_intervalle / 3, l_intervalle / 40, y) + gaussian(2 * l_intervalle / 3, l_intervalle / 40, y) # 
    # capacity = capacity / 1.5

    potential = reward_attractiveness * (1-attract_y).reshape(1, -1)#
    # phi = pen_distance * (y.reshape(-1, 1) - y.reshape(1, -1)) ** 2
    #gap_price_budget = attract_y.reshape(1, -1) - x.reshape(-1, 1)
    # costs = np.where(gap_price_budget < 0, (1 - reward_attractiveness) * attract_y, (1 - reward_attractiveness) * attract_y + pen_overcost * gap_price_budget)

    costs = pen_distance * (1 / l_intervalle ** 2) * (x.reshape(-1, 1) - y.reshape(1, -1)) ** 2 + np.tile(potential , (x.size, 1)) #np.where(gap_price_budget < 0, 0, pen_overcost * gap_price_budget) + np.tile(potential , (x.size, 1))
    # def E(nu, phi):
    #     return np.sum((1/2) * nu ** 2 + (1/2) * pen_density * (nu ** 2) + 0.5 * nu.T @ phi @ nu + potential)

    #+ np.tile(potential, (x.size, 1))

    method = ConvexProximalMethod(mu, costs, epsilon, capacity)
    gamma, nu, mem_nu, mem_gap= method.fit()

    plt.rc('font', size=16)

    if True:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))


        color2 = "forestgreen"
        # ax2 = ax1.twinx()
        ax1.plot(y, attract_y, label="attractiveness", color=color2, linewidth=1.7)
        #ax2.plot(y, capacity, label="building zones", color=color2, linewidth=1.7)
        ax1.set_ylabel('attractiveness', color=color2)
        ax1.tick_params(axis='y', labelcolor=color2)
        ax1.grid(True)
        ax1.legend(loc="upper right")
        
        ax1.set_title("Distribution of the agents according to the capacity and the attractiveness")

        color3 = "grey"
        ax2.plot(y, capacity* n_points / l_intervalle, label="capacity", color=color3, linewidth=1.7)
        

        color1 = "blue"
        ax2.plot(y, nu * n_points / l_intervalle, label=r"$\nu$", color=color1, linewidth=1.7)
        ax2.grid(True)
        ax2.tick_params(axis='y', labelcolor=color1)
        # ax1.set_xlabel(r"$p_{dens} = $" + f"{PEN_DENSITY}")
        ax2.set_ylabel("density", color=color1)
        ax2.set_xlabel("location")
        #ax2.set_ylim(0, 1)

        fig.tight_layout()
        plt.legend()

    if True:
        fig2 = plt.figure(f"Gap", figsize=(15, 10))
        ax3 = fig2.add_subplot(111)
        ax3.plot(mem_gap, label="logarithm of the gap", color=color1, linewidth=1.7)
        ax3.set_xlabel("Number of iterations")
        ax3.set_title("Evolution of the logarithm of the gap between two proximal steps")
        ax3.set_ylabel("logarithm of the gap between two proximal steps")
        ax3.grid(True)

    if True:
        fig3 = plt.figure(f"Evolution", figsize=(15, 10))
        ax4 = fig3.add_subplot(111)
        cmap = matplotlib.cm.get_cmap('Spectral')
        for i, nu in enumerate(mem_nu):
            ax4.plot(y, nu, color=cmap((i+1)/len(mem_nu)), linewidth=1.7)
        ax4.set_xlabel("x")
        ax4.set_title(r"Evolution $\nu$ throughout the proximal steps")
        ax4.set_ylabel(r"distribution $\nu$")

        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(ax4)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(cmap(np.array(range(1, len(mem_nu)+1))/len(mem_nu)), cax=cax)

        cbar4 = fig3.colorbar(matplotlib.cm.ScalarMappable(norm = matplotlib.cm.colors.Normalize(vmax=len(mem_nu), vmin=0), cmap=cmap), ax=ax4)
        cbar4.ax.set_ylabel('iteration rank')
        # ax4.colormap(cmap)
        ax4.grid(True)

    if True:
        fig4 = plt.figure(f"Gamma", figsize=(15, 10))
        ax5 = fig4.add_subplot(111)

        ax5.set_title("Matrix gamma")
        im = ax5.imshow(gamma, origin='lower', extent=[0, 16, 0, 16])#, extent=tlim+flim, aspect='auto')
        ax5.set_ylabel(r"$x$")
        ax5.set_xlabel(r"$y$")

        ax5.set_title(r"Values of the coefficients of $\gamma$")

        cbar = fig4.colorbar(im, ax=ax5)
        cbar.ax.set_ylabel(r"$\gamma_{ij}$")
        # im = ax5.imshow(gamma)

        # ax5.set_ylabel("density", color=color1)
        # ax5.set_xlabel("x")

        # divider = make_axes_locatable(ax5)
        # cax = divider.append_axes("right", size="5%", pad=0.05)

        # plt.colorbar(im, cax=cax)

        # gs = gridspec.GridSpec(2, 2, width_ratios=[1,3], height_ratios=[3,1], figure=fig4)
        # ax5 = plt.subplot(gs[0,1])
        # axl = plt.subplot(gs[0,0], sharey=ax5)
        # axb = plt.subplot(gs[1,1], sharex=ax5)

        # axl.grid(True)
        # axb.grid(True)

        # plt.xlim(tlim)

        # axl.plot(gamma.mean(1), y)
        # axb.plot(x, gamma.mean(0))
        # axl.set_ylabel(r"$\mu$")
        # axb.set_xlabel(r"$\nu$")










    plt.show()
    return gamma, nu, mu, x, y, attract_y, capacity, costs

def ComputeGammaPotential(max_attractiveness, reward_attractiveness, pen_distance, pen_density, n_points, epsilon, l_intervalle, n_peaks):
    x = np.linspace(0, l_intervalle, n_points)
    # mu = gaussian(MU_1, SIGMA_1, x) + gaussian(MU_2, SIGMA_2, x)
    # mu = (mu / np.linalg.norm(mu, ord=1))

    mu = np.ones(x.shape) / x.shape

    y =  np.linspace(0, l_intervalle, n_points) # TODO: tester avec x et y de taille différente 
    attract_y = gaussian(l_intervalle / 2, l_intervalle / 20, y)
    #attract_y = #gaussian(MU_1, SIGMA_1, y) + gaussian(MU_2, SIGMA_2, y)
    attract_y = max_attractiveness * attract_y / np.max(attract_y)
    
    districts = n_points / n_peaks
    capacity = np.where(np.logical_and((n_points * x / l_intervalle) % districts >= (districts/4), (n_points * x / l_intervalle) % districts <=(3*districts/4)), 1, 10 ** (-4))
    capacity = 2*capacity / np.sum(capacity)
    
    # print("HELLO", np.sum(capacity))#capacity)#(n_points * x / l_intervalle) % 100)
    
    #gaussian(l_intervalle / 3, l_intervalle / 40, y) + gaussian(2 * l_intervalle / 3, l_intervalle / 40, y) # 
    # capacity = capacity / 1.5

    potential = reward_attractiveness * (1-attract_y).reshape(1, -1)#

    dx = l_intervalle / n_points #if abs(i-j) < (districts/2) else 0
    phi = pen_density * np.array([[np.sqrt(dx / (abs(x[i]-x[j]) + dx))  for i in range(N_POINTS)] for j in range(N_POINTS)]) #(y.reshape(-1, 1) - y.reshape(1, -1)) ** 2
    #gap_price_budget = attract_y.reshape(1, -1) - x.reshape(-1, 1)
    # costs = np.where(gap_price_budget < 0, (1 - reward_attractiveness) * attract_y, (1 - reward_attractiveness) * attract_y + pen_overcost * gap_price_budget)

    costs = pen_distance * (1 / l_intervalle ** 2) * (x.reshape(-1, 1) - y.reshape(1, -1)) ** 2 + np.tile(potential , (x.size, 1)) #np.where(gap_price_budget < 0, 0, pen_overcost * gap_price_budget) + np.tile(potential , (x.size, 1))
    # def E(nu, phi):
    #     return np.sum((1/2) * nu ** 2 + (1/2) * pen_density * (nu ** 2) + 0.5 * nu.T @ phi @ nu + potential)

    #+ np.tile(potential, (x.size, 1))

    method = NonConvexProximalMethod(mu, costs, epsilon, phi, capacity)
    gamma, nu, mem_nu, mem_gap, mem_potential= method.fit()

    plt.rc('font', size=16)

    if True:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        color2 = "forestgreen"
        # ax2 = ax1.twinx()
        ax1.plot(y, attract_y, label="attractiveness", color=color2, linewidth=1.7)
        ax1.plot(y, mem_potential[-1], label=r"potential $\phi\nu$", color ="orange")
        #ax2.plot(y, capacity, label="building zones", color=color2, linewidth=1.7)
        ax1.set_ylabel('attractiveness', color=color2)
        ax1.tick_params(axis='y', labelcolor=color2)
        ax1.grid(True)
        ax1.legend(loc="upper right")
        
        ax1.set_title("Distribution of the agents according to the capacity and the attractiveness")

        color3 = "grey"
        ax2.plot(y, capacity* n_points / l_intervalle, label="capacity", color=color3, linewidth=1.7)
        

        color1 = "blue"
        ax2.plot(y, nu * n_points / l_intervalle, label=r"$\nu$", color=color1, linewidth=1.7)
        ax2.grid(True)
        ax2.tick_params(axis='y', labelcolor=color1)
        # ax1.set_xlabel(r"$p_{dens} = $" + f"{PEN_DENSITY}")
        ax2.set_ylabel("density", color=color1)
        ax2.set_xlabel("location")
        ax2.legend(loc="upper right")
        #ax2.set_ylim(0, 1)

        fig.tight_layout()
        plt.legend()

    if False:
        fig2 = plt.figure(f"Gap", figsize=(15, 10))
        ax3 = fig2.add_subplot(111)
        ax3.plot(mem_gap, label="logarithm of the gap", color=color1, linewidth=1.7)
        ax3.set_xlabel("Number of iterations")
        ax3.set_title("Evolution of the logarithm of the gap between two proximal steps")
        ax3.set_ylabel("logarithm of the gap between two proximal steps")
        ax3.grid(True)

    if True:
        fig3 = plt.figure(f"Evolution potential", figsize=(15, 10))
        ax4 = fig3.add_subplot(111)
        cmap = matplotlib.cm.get_cmap('Spectral')
        for i, potential in enumerate(mem_potential):
            ax4.plot(y, potential, label="logarithm of the gap", color=cmap((i+1)/len(mem_nu)), linewidth=1.7)
        ax4.set_xlabel("x")
        ax4.set_title(r"Evolution of the potentaial $\phi \nu$ throughout the proximal steps")
        ax4.set_ylabel(r"distribution $\nu$")

        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(ax4)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(cmap(np.array(range(1, len(mem_nu)+1))/len(mem_nu)), cax=cax)

        cbar4 = fig3.colorbar(matplotlib.cm.ScalarMappable(norm = matplotlib.cm.colors.Normalize(vmax=len(mem_nu), vmin=0), cmap=cmap), ax=ax4)
        cbar4.ax.set_ylabel('rang de l\'itération')
        ax4.grid(True)

    if True:
        fig4 = plt.figure(f"Evolution value", figsize=(15, 10))
        ax4 = fig4.add_subplot(111)
        cmap = matplotlib.cm.get_cmap('Spectral')
        for i, nu in enumerate(mem_nu):
            ax4.plot(y, nu, color=cmap((i+1)/len(mem_nu)), linewidth=1.7)
        ax4.set_xlabel("x")
        ax4.set_title(r"Evolution $\nu$ throughout the proximal steps")
        ax4.set_ylabel(r"distribution $\nu$")

        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(ax4)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(cmap(np.array(range(1, len(mem_nu)+1))/len(mem_nu)), cax=cax)

        cbar4 = fig3.colorbar(matplotlib.cm.ScalarMappable(norm = matplotlib.cm.colors.Normalize(vmax=len(mem_nu), vmin=0), cmap=cmap), ax=ax4)
        cbar4.ax.set_ylabel('rang de l\'itération')
        # ax4.colormap(cmap)
        ax4.grid(True)

    plt.show()
    return gamma, nu, mu, x, y, attract_y, capacity, costs

def resume(gamma):
    x = np.linspace(0, 1, N_POINTS)
    y =  np.linspace(0, L_INTERVALLE, N_POINTS) # TODO: tester avec x et y de taille différente
    districts = N_POINTS / N_PEAKS
    attract_y = gaussian(L_INTERVALLE / 2, L_INTERVALLE / 20, y)
    phi = PEN_DENSITY * np.array([[1 if abs(i-j) < (districts/2) else 0 for i in range(N_POINTS)] for j in range(N_POINTS)])
    nu = np.sum(gamma, axis=0)
    print("somme : ", np.sum(nu))
    #print("penalisation_densite moyenne : ", np.mean(PEN_DENSITY * (nu **2)))
    print("apport transport moyen : ", np.sum(gamma * PEN_DISTANCE * (1 / L_INTERVALLE ** 2) * (x.reshape(-1, 1) - y.reshape(1, -1)) ** 2))# MAX_ATTRACTIVENESS * (1-attract_y) @ nu)
    print("penalisation distance moyenne: ", PEN_DENSITY * nu.T @ phi @ nu)

if __name__ == "__main__":
        
        # ComputeGamma(MAX_ATTRACTIVENESS, REWARD_ATTRACTIVENESS, PEN_DISTANCE, N_POINTS, EPS, L_INTERVALLE, N_PEAKS)

        ComputeGammaPotential(MAX_ATTRACTIVENESS, REWARD_ATTRACTIVENESS, PEN_DISTANCE, PEN_DENSITY, N_POINTS, EPS, L_INTERVALLE, N_PEAKS)














        # gamma, nu, mu, x, y, attract_y, capacity, costs = 
        # np.sum(gamma), np.max(np.sum(gamma, axis=1) - mu)




        # plt.rc('font', size=16)
        # fig1 = plt.figure(f"Solution finale", figsize=(15, 10))
        # ax1 = fig1.add_subplot(111)

        # color3 = "grey"
        # ax1.plot(y, capacity* N_POINTS / L_INTERVALLE, label="building limit", color=color3, linewidth=1.7)

        # color1 = "blue"
        # ax1.plot(y, nu * N_POINTS / L_INTERVALLE, label="nu", color=color1, linewidth=1.7)
        # ax1.grid(True)
        # ax1.tick_params(axis='y', labelcolor=color1)
        # # ax1.set_xlabel(r"$p_{dens} = $" + f"{PEN_DENSITY}")
        # ax1.set_ylabel("density", color=color1)
        # ax1.set_ylim(0,1)
        # ax1.legend(loc="upper right")
        


        # # ax1.set_ylabel('attractiveness', color=color3)
        # # ax1.tick_params(axis='y', labelcolor=color3)

        # color2 = "forestgreen"
        # ax2 = ax1.twinx()
        # ax2.plot(y, attract_y, label="attractiveness", color=color2, linewidth=1.7)
        # #ax2.plot(y, capacity, label="building zones", color=color2, linewidth=1.7)
        # ax2.set_ylabel('attractiveness', color=color2)
        # ax2.tick_params(axis='y', labelcolor=color2)
        # ax1.legend(loc="upper left")
        # ax1.set_title("Distribution of the agents according to the capacity and the attractiveness")
        # fig1.tight_layout()
        # plt.legend()
        # plt.show()

        # color3 = 2
        # ax2.plot(y, capacity, label="building zones", color=color3, linewidth=1.7)
        # ax2.set_ylabel('attractiveness', color=color3)
        # ax2.tick_params(axis='y', labelcolor=color3)
        # fig1.tight_layout()
        # plt.show()