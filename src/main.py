import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import lower
from numpy.lib.polynomial import polyfit
from mathutils import gaussian
import argparse
import scipy.optimize as sciopt
# import pylab as pl
import matplotlib.collections as mc
#from config import MU_1, SIGMA_1, MU_2, SIGMA_2, N_POINTS
#from scipy.optimize import newton

N_POINTS = 500

## distribution initiale
MU_1 = 4
SIGMA_1 = .7
MU_2 = 12
SIGMA_2 = .7
L_INTERVALLE =16
EPS = 0.1
PREC = 10 ** (-15)
PREC2 = 10 ** (-5)

def newton(f, x0, jacobian, eps, args=()):
    x=x0
    f_value = f(x, *args)
    f_norm = np.linalg.norm(f_value, ord=2)  # l2 norm of vector
    n_iter = 0
    while abs(f_norm) > (eps / 2) and n_iter < 100:

        delta = np.linalg.solve(jacobian(x, *args), -f_value)
        x = x + delta
        # if n_iter == 10:
        #     plt.imshow(jacobian(x, *args))
        #     plt.show()
        # if n_iter == 11:
        #     plt.imshow(jacobian(x, *args))
        #     plt.show()
        print("min_delta", np.min(delta))
        f_value = f(x, *args)
        # print("borne", np.min(f_value), np.max(f_value), np.min(jacobian(x, *args)), np.max(jacobian(x, *args)))
        # print("step", np.linalg.norm(f_value - delta))
        f_norm = np.linalg.norm(f_value, ord=2)
        print("norm", f_norm, )
        n_iter += 1

    # Here, either a solution is found, or too many iterations
    if abs(f_norm) > eps:
        # n_iter = -1
        print("Newton failed to converge")
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
    while (np.linalg.norm(lowerb - upperb) > prec or np.linalg.norm(f_middles) > prec) and n_iter < 1000000:
        n_iter += 1
        middles = (lowerb + upperb) / 2
        f_middles = f(middles, *args)
        
        upperb = np.where(f_middles >= 0, middles, upperb)
        lowerb = np.where(f_middles <= 0, middles, lowerb)
    # print(np.sum((lowerb + upperb) / 2))
    return (lowerb + upperb) / 2

class ConvexProximalMethod():

    def __init__(self, mu, costs, phi, eps, h_G3, hprim_G3, E):
        self.mu = mu.reshape(-1, 1)
        self.costs = costs
        self.phi = phi
        self.eps = eps
        self.h_G3 = h_G3
        self.hprim_G3 = hprim_G3
        self.E = E

    def prox_G1_KL(self, theta):
        sum_theta = np.sum(theta, axis=1)
        return theta * self.mu * np.nan_to_num(1 / sum_theta).reshape(-1, 1) #(self.mu.T @ theta).reshape(-1, 1) / np.sum(theta, axis=1).reshape(-1, 1) #TODO: vérifier dimension

    def log_prox_G1_KL(self, theta):
        sum_theta = np.sum(theta, axis=1)
        return np.log(theta) + np.log(self.mu) - np.nan_to_num(np.log(sum_theta).reshape(-1, 1)) #(self.mu.T @ theta).reshape(-1, 1) / np.sum(theta, axis=1).reshape(-1, 1) #TODO: vérifier dimension

    #prox 2
    def f_G2(self, nu, theta):
        # nu vecteur vertical
        return nu - np.sum(theta, axis=0).reshape(nu.shape) * np.exp( - (nu + self.phi.T @ nu) / self.eps)

    def jacobian_f_G2(self, nu, theta):
        #nu est vertical
        A = np.sum(theta, axis=0).reshape(nu.shape) * np.exp(- (nu + self.phi.T @ nu) / self.eps)
        return np.eye(nu.size) + ((np.eye(nu.size) + self.phi) / self.eps) * np.tile(A, (1, nu.size))

    def prox_G2_KL(self, theta):  
        assert np.sum(phi ** 2) < 1 #TODO remplacer par une erreur
        nu = newton(self.f_G2, np.zeros((theta.shape[1], 1)), self.jacobian_f_G2, self.eps, args=(theta,))
        return theta * np.exp(-(nu + nu.T @ self.phi) / self.eps)

    def log_prox_G2_KL(self, theta):  
        assert np.sum(phi ** 2) < 1 #TODO remplacer par une erreur
        nu = newton(self.f_G2, np.zeros((theta.shape[1], 1)), self.jacobian_f_G2, self.eps, args=(theta,))
        return np.log(theta) - (nu + nu.T @ self.phi) / self.eps

    #prox 3
    def f_G3(self, nu, theta):
        return nu - np.sum(theta, axis=0).reshape(nu.shape) * np.exp(- self.h_G3(nu) / self.eps)

    def jacobian_f_G3(self, nu, theta):
        A = np.sum(theta, axis=0).reshape(nu.shape) * np.exp(- self.h_G3(nu) / self.eps)
        return np.eye(nu.size) + self.hprim_G3(nu) * np.tile(A, (1, nu.size))

    def prox_G3_KL(self, theta):
        nu = newton(self.f_G3, np.zeros((theta.shape[1], 1)), self.jacobian_f_G3, self.eps, args=(theta,))
        return theta * np.exp(- self.h_G3(nu) / self.eps)

    def log_prox_G3_KL(self, theta):
        nu = newton(self.f_G3, np.zeros((theta.shape[1], 1)), self.jacobian_f_G3, self.eps, args=(theta,))
        return np.log(theta) - self.h_G3(nu) / self.eps

    #Prox cyclique
    def prox_Gn_KL(self, n, theta):
        if n % 3 == 1:
            return self.prox_G1_KL(theta)

        elif n % 3 == 2:
            return self.prox_G2_KL(theta)

        elif n % 3 == 0:
            return self.prox_G3_KL(theta)

    def log_prox_Gn_KL(self, n, theta):
        if n % 3 == 1:
            return self.log_prox_G1_KL(theta)

        elif n % 3 == 2:
            return self.log_prox_G2_KL(theta)

        elif n % 3 == 0:
            return self.log_prox_G3_KL(theta)


    #PROXIMAL ALGO
    def fit(self):
        L = 3
        gamma = np.exp(-self.costs / self.eps)
        z = [np.ones((self.mu.size, self.costs.shape[1])) for _ in range(L)]
        gap = np.inf
        n = 1
        # print("gamma", np.linalg.norm(gamma))
        while gap > self.eps or n%L != 1: # todo : trouver une condition sur le gap qui ne soit pas dépendante du nombre de points
            #print(n)
            new_gamma = self.prox_Gn_KL(n, gamma * z[n % L])
            z[n % L] = z[(n-1) % L] * (gamma / new_gamma)
            # print("new_gamma : ", np.linalg.norm(new_gamma, ord=1), new_gamma.shape)
            # print("gamma : ", np.linalg.norm(gamma, ord=1), new_gamma.shape)
            gap = np.linalg.norm(gamma - new_gamma, ord=np.inf)
            # print("gap : ", gap)
            gamma = new_gamma
            n += 1
        
        return gamma, np.sum(gamma, axis=0)

    def fit_log(self):
        L = 3
        log_gamma = -self.costs / self.eps
        z = [np.ones((self.mu.size, self.costs.shape[1])) for _ in range(L)]
        gap = np.inf
        n = 1
        # print("gamma", np.linalg.norm(gamma))
        while gap > self.eps or n%L != 1: # todo : trouver une condition sur le gap qui ne soit pas dépendante du nombre de points
            #print(n)
            log_new_gamma = self.log_prox_Gn_KL(n, log_gamma * z[n % L])
            z[n % L] = z[(n-1) % L] * np.exp(log_gamma - log_new_gamma)
            # print("new_gamma : ", np.linalg.norm(log_new_gamma, ord=1), log_new_gamma.shape)
            # print("gamma : ", np.linalg.norm(log_gamma, ord=1), log_new_gamma.shape)
            gap = np.linalg.norm(np.exp(log_gamma) - np.exp(log_new_gamma), ord=np.inf)
            # print("gap : ", gap)
            log_gamma = log_new_gamma
            n += 1
        
        gamma = np.exp(log_gamma)

        return gamma, np.sum(gamma, axis=0)

    def fit_log2(self):
        L = 3
        gamma = np.exp(-self.costs / self.eps)
        z = [np.ones((self.mu.size, self.costs.shape[1])) for _ in range(L)]
        gap = np.inf
        n = 1
        # print("gamma", np.linalg.norm(gamma))
        while gap > self.eps or n%L != 1: # todo : trouver une condition sur le gap qui ne soit pas dépendante du nombre de points
            #print(n)
            new_gamma = self.prox_Gn_KL(n, gamma * z[n % L])
            z[n % L] = z[(n-1) % L] * np.exp(np.log(gamma) -np.log(new_gamma))
            # print("new_gamma : ", np.linalg.norm(new_gamma, ord=1), new_gamma.shape)
            # print("gamma : ", np.linalg.norm(gamma, ord=1), new_gamma.shape)
            gap = np.linalg.norm(gamma - new_gamma, ord=np.inf)
            # print("gap : ", gap)
            gamma = new_gamma
            n += 1
        
        return gamma, np.sum(gamma, axis=0)

class SemiImplicitMethod():

    def __init__(self, mu, costs, phi, eps, h_G2, hprim_G2, E):
        self.mu = mu.reshape(-1, 1)
        self.costs = costs
        self.phi = phi
        self.eps = eps
        self.E = E
        self.h_G2 = h_G2
        self.hprim_G2 = hprim_G2

    def prox_G1_KL(self, theta):
        sum_theta = np.sum(theta, axis=1)
        return theta * self.mu * np.nan_to_num(1 / sum_theta).reshape(-1, 1)

    def f_G2(self, nu, theta):
        return nu - np.sum(theta, axis=0).reshape(nu.shape) * np.exp(- self.h_G2(nu) / self.eps)

    def jacobian_f_G2(self, nu, theta):
        A = np.sum(theta, axis=0).reshape(nu.shape) * np.exp(- self.h_G2(nu) / self.eps)
        return np.eye(nu.size) + (1 / self.eps) * np.diag((self.hprim_G2(nu) * A).ravel()) #np.eye(nu.size) + self.hprim_G2(self.mu) * np.tile(A, (1, nu.size))

    def prox_G2_KL(self, theta):
        # nu = newton_diagonal(self.f_G2, np.zeros((theta.shape[1], 1)), self.jacobian_f_G2, self.eps, args=(theta,))
        #nu = sciopt.newton(self.f_G2, np.zeros((theta.shape[1], 1)), fprime=self.jacobian_f_G2, args=(theta,)) #
        # nu = newton_diagonal(self.f_G2, np.zeros((theta.shape[1], 1)), self.jacobian_f_G2, self.eps, args=(theta,))
        nu = dichotomy(self.f_G2, np.zeros((theta.shape[1], 1)), np.ones((theta.shape[1], 1)), PREC, args=(theta,))
        print("HELLO", np.min(np.exp(- self.h_G2(nu) / self.eps)), flush=True)
        return theta * np.exp(- self.h_G2(nu) / self.eps)

    def prox_Gn_KL(self, n, theta):
        if n % 2 == 1:
            return self.prox_G1_KL(theta)

        elif n % 2 == 0:
            return self.prox_G2_KL(theta)

    def proximal_step(self, costs):
        L = 2
        gamma = np.exp(-costs / self.eps) # TODO: revoir point de départ
        old_gamma = gamma 
        z = [np.ones((self.mu.size, self.costs.shape[1])) for _ in range(L)]
        break_loop = False
        n = 1
        while not break_loop:
            print(n, flush=True)
            new_gamma = self.prox_Gn_KL(n, gamma * z[n % L])
            new_gamma = np.maximum(new_gamma, (10 ** -300))
            # print("ZEROS :", np.count_nonzero(new_gamma==0))
            print("evol", np.linalg.norm(gamma- new_gamma), flush=True)
            z[n % L] = z[n % L] * np.nan_to_num(gamma / new_gamma)

            if n % L == 0:
                print("TEST", np.mean(np.exp(- self.h_G2(np.sum(new_gamma, axis=0)) / self.eps)))
            # plt.plot(np.sum(new_gamma, axis=0))
            # plt.title(str(n) + " " + str(np.sum(new_gamma)))
            # plt.show()

            # print("TRUE MULTIPLY", np.max(np.nan_to_num(gamma / new_gamma)))
            # print("new_gamma : ", np.linalg.norm(new_gamma, ord=1), new_gamma.shape)
            # print("gamma : ", np.linalg.norm(new_gamma, ord=1), new_gamma.shape)
            


            gamma = new_gamma
            if n >= 3 and n%L == 0:
                gap = np.linalg.norm(np.nan_to_num(old_gamma - new_gamma), ord=np.inf)
                break_loop = gap < PREC2
                old_gamma = gamma
            n += 1
        
        return gamma, np.sum(gamma, axis=0)

    def fit(self):
        nu = np.ones((self.costs.shape[1], 1))
        nu = nu / np.sum(nu)
        gap = np.inf
        n_iter = 1
        while gap > PREC2:
            print("ITER", n_iter)
            #print(n_iter)
            new_costs = self.costs + np.tile(nu.T @ self.phi, (self.costs.shape[0], 1))
            gamma, new_nu = self.proximal_step(new_costs)
            gap = np.linalg.norm(new_nu - nu, ord=1)
            nu = new_nu
            print("gap 2", gap)
            n_iter += 1
        return gamma, nu

def ComputeGamma(max_attractiveness, reward_attractiveness, pen_overcost, pen_density, pen_distance, n_points, epsilon, l_intervalle):
    x = np.linspace(0, 1, n_points)
    # mu = gaussian(MU_1, SIGMA_1, x) + gaussian(MU_2, SIGMA_2, x)
    # mu = (mu / np.linalg.norm(mu, ord=1))

    mu = np.ones(x.shape) / x.shape

    y =  np.linspace(0, l_intervalle, n_points) # TODO: tester avec x et y de taille différente
    attract_y = gaussian(l_intervalle / 2, l_intervalle / 20, y)#gaussian(MU_1, SIGMA_1, y) + gaussian(MU_2, SIGMA_2, y)
    attract_y = max_attractiveness * attract_y / np.max(attract_y)

    phi = pen_distance * (y.T - y) ** 2

    gap_price_budget = attract_y.reshape(1, -1) - x.reshape(-1, 1)
    costs = np.where(gap_price_budget < 0, (1 - reward_attractiveness) * attract_y, (1 - reward_attractiveness) * attract_y + pen_overcost * gap_price_budget)

    def E(nu, phi):
        return np.sum(pen_density * (nu ** 3) + 0.5 * nu.T @ phi @ nu)

    #+ np.tile(potential, (x.size, 1))
    def h_G2(nu): # fonction issue du potentiel
        return 3 * pen_density * (nu ** 2)
    
    def hprim_G2(nu):
        return 6 * pen_density * nu#56 * (nu ** 6)

    method = SemiImplicitMethod(mu, costs, phi, epsilon, h_G2, hprim_G2, E)
    gamma, nu = method.fit()
    return gamma, nu, mu, x, y, attract_y, costs, phi

def ComputeGamma2(pen_mobility, pen_density, pen_distance, n_points, epsilon, l_intervalle):
    x = np.linspace(0, l_intervalle, n_points)
    mu = gaussian(MU_1, SIGMA_1, x) + gaussian(MU_2, SIGMA_2, x)
    mu = (mu / (np.linalg.norm(mu, ord=1))) #+ eps

    # mu = np.ones(x.shape) / x.shape

    y =  np.linspace(0, l_intervalle, n_points) # TODO: tester avec x et y de taille différente
    # attract_y = gaussian(l_intervalle / 2, l_intervalle / 20, y)#gaussian(MU_1, SIGMA_1, y) + gaussian(MU_2, SIGMA_2, y)
    # attract_y = max_attractiveness * attract_y / np.max(attract_y)

    phi = pen_distance * (y.T - y) ** 2

    costs = pen_mobility * (y.reshape(1, -1) - x.reshape(-1, 1)) ** 2

    def E(nu, phi):
        return np.sum( (1/3) * pen_density * (nu ** 3) + 0.5 * nu.T @ phi @ nu)

    #+ np.tile(potential, (x.size, 1))
    def h_G2(nu): # fonction issue du potentiel
        return pen_density * (nu ** 2)
    
    def hprim_G2(nu):
        return 2 * pen_density * nu   #56 * (nu ** 6)

    method = SemiImplicitMethod(mu, costs, phi, epsilon, h_G2, hprim_G2, E)
    gamma, nu = method.fit()
    return gamma, nu, mu, x, y, costs, phi

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="mode of the execution",
                        choices=["reprod", "holidays", "debug"], default="holidays")
    args = parser.parse_args()
    mode = args.mode

    # Reproduction de l'algo de l'article
    if mode == "reprod":
        x = np.linspace(0, L_INTERVALLE, N_POINTS)
        mu = gaussian(MU_1, SIGMA_1, x) + gaussian(MU_2, SIGMA_2, x)

        EPS2 = 0.05
        mu = (1 - EPS2) * (mu / np.linalg.norm(mu, ord=1)) + (EPS2 / N_POINTS)
        

        y = x # TODO: tester avec x et y de taille différente

        phi = (10 ** -4) * (y.T - y) ** 2
        potential = ((y-9) ** 4).reshape(1, -1)
        costs = (x.reshape(-1, 1) - y.reshape(1, -1)) ** 2 + np.tile(potential, (x.size, 1))

        def E(nu, phi, potential):
            return np.sum(nu ** 8 + 0.5 * nu.T @ phi @ nu + potential)

        # def h_G3(nu): # fonction issue du potentiel
        #     return 8 * (nu ** 7) - nu

        # def hprim_G3(nu):
        #     return 56 * (nu ** 6) - 1

        # method = ConvexProximalMethod(mu, costs, phi, EPS, h_G3, hprim_G3, E)
        # gamma, nu = method.fit()

        def h_G2(x): # fonction issue du potentiel
            return 8 * (x ** 7)
        
        def hprim_G2(nu):
            return 56 * (x ** 6)

        method = SemiImplicitMethod(mu, costs, phi, EPS, h_G2, hprim_G2, E)
        gamma, nu = method.fit()


        #print("somme gamma", np.sum(gamma))
        print("somme mu", np.sum(mu))
        print("somme nu", np.sum(nu))
        
        fig = plt.figure(f"Solution finale", figsize=(15, 10))
        ax1 = fig.add_subplot(111)
        ax1.plot(x, mu * N_POINTS / L_INTERVALLE, label="mu")
        ax1.plot(x, nu * N_POINTS / L_INTERVALLE, label="nu")
        ax1.grid(True)
        ax1.set_xlabel("x")
        ax1.set_ylabel("density")
        ax1.set_title("Illustration de la valeur de nu en fonction de celle de mu")
        ax1.legend()
        plt.show()


    elif mode == "holidays":

        MAX_ATTRACTIVENESS = 1
        REWARD_ATTRACTIVENESS = 2
        PEN_OVERCOST = 3
        PEN_DENSITY = 10
        PEN_DISTANCE = 10

        

        # x = np.linspace(0, 1, N_POINTS)
        # # mu = gaussian(MU_1, SIGMA_1, x) + gaussian(MU_2, SIGMA_2, x)
        # # mu = (mu / np.linalg.norm(mu, ord=1))

        # mu = np.ones(x.shape) / x.shape

        # y =  np.linspace(0, L_INTERVALLE, N_POINTS) # TODO: tester avec x et y de taille différente
        # attract_y = gaussian(L_INTERVALLE / 2, L_INTERVALLE / 20, y)#gaussian(MU_1, SIGMA_1, y) + gaussian(MU_2, SIGMA_2, y)
        # attract_y = MAX_ATTRACTIVENESS * attract_y / np.max(attract_y)

        # phi = PEN_DISTANCE * (y.T - y) ** 2

        # gap_price_budget = attract_y.reshape(1, -1) - x.reshape(-1, 1)
        # costs = np.where(gap_price_budget < 0, (1 - REWARD_ATTRACTIVENESS) * attract_y, (1 - REWARD_ATTRACTIVENESS) * attract_y + PEN_OVERCOST * gap_price_budget)

        # def E(nu, phi):
        #     return np.sum( PEN_DENSITY * (nu ** 3) + 0.5 * nu.T @ phi @ nu)

        # #+ np.tile(potential, (x.size, 1))
        # def h_G2(nu): # fonction issue du potentiel
        #     return 3 * PEN_DENSITY * (nu ** 2)
        
        # def hprim_G2(nu):
        #     return 6 * PEN_DENSITY *  nu#56 * (nu ** 6)

        # method = SemiImplicitMethod(mu, costs, phi, EPS, h_G2, hprim_G2, E)
        # gamma, nu = method.fit()

        gamma, nu, mu, x, y, attract_y, costs, phi = ComputeGamma(MAX_ATTRACTIVENESS, REWARD_ATTRACTIVENESS, PEN_OVERCOST, PEN_DENSITY, PEN_DISTANCE, N_POINTS, EPS, L_INTERVALLE)

        plt.rc('font', size=16)

        # if True:
        #     
        #     plt.plot(y, costs[0,:])#np.sum(gamma, axis=1))
        #     plt.show()

        #     plt.imshow(np.where(gap_price_budget < 0, 0, PEN_OVERCOST * gap_price_budget))
        #     plt.colorbar()
        #     plt.show()

        #     plt.imshow(gamma)
        #     plt.colorbar()
        #     plt.show()
            
        if True:            
            fig1 = plt.figure(f"Solution finale", figsize=(15, 10))
            ax1 = fig1.add_subplot(111)
            #ax1.plot(x, mu * N_POINTS / L_INTERVALLE, label="mu")
            #ax1.plot(y, attract_y, label="attractiveness")
            color1 = "blue"
            ax1.plot(y, nu * N_POINTS / L_INTERVALLE, label="nu", color=color1, linewidth=1.7)
            ax1.grid(True)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.set_xlabel("abscisse of the destination")
            ax1.set_ylabel("density", color=color1)
            ax1.set_title("Equilibrium distribution of the players")

            color2 = "darkorange"
            ax2 = ax1.twinx()
            ax2.plot(y, attract_y, label="attractiveness", color=color2, linewidth=1.7)
            ax2.set_ylabel('attractiveness', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            fig1.tight_layout()


            polys = []
            values = []
            for j, (ys, ye) in enumerate(zip(y[:-1], y[1:])):
                z = np.cumsum(gamma[:, j])
                zp = np.c_[z[:-1], z[:-1], z[1:], z[1:]] #concatenates along the second axis

                yp = np.repeat([[ys, ye, ye, ys]], len(zp), axis=0)
                points = np.dstack((yp, zp))
                polys.append(points)

                values.append(.5 * (x[:-1] + x[1:]))

            polys = np.concatenate(polys, 0)
            values = np.concatenate(values, 0)

            pc = mc.PolyCollection(polys)
            pc.set_array(values)

            fig, ax = plt.subplots(figsize=(15, 10))
            ax.add_collection(pc)
            ax.autoscale()    
            plt.colorbar(mappable=pc)
            plt.grid(True)



            fig2 = plt.figure(f"dépassements de budget", figsize=(15, 10))
            ax1 = fig2.add_subplot(111)
            #ax1.plot(x, mu * N_POINTS / L_INTERVALLE, label="mu")
            #ax1.plot(y, attract_y, label="attractiveness")
            color1 = "blue"
            # ax1.plot(x, np.sum(gamma * np.where(gap_price_budget < 0, 0, PEN_OVERCOST * gap_price_budget), axis=1), label="nu", color=color1, linewidth=1.7)
            # ax1.grid(True)
            # ax1.tick_params(axis='y', labelcolor=color1)
            # ax1.set_xlabel("budget")
            # ax1.set_ylabel("mean overcost", color=color1)
            # ax1.set_title("Mean_overcost")

            ax1.plot(x, np.sum(gamma * attract_y.reshape(1, -1) / mu, axis=1), label="nu", color=color1, linewidth=1.7)
            ax1.grid(True)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.set_xlabel("budget")
            ax1.set_ylabel("mean overcost", color=color1)
            ax1.set_title("Attractivity of the destination chosen against the budget")

            # color2 = "darkorange"
            # ax2 = ax1.twinx()
            # ax2.plot(x, np.sum(gamma * attract_y.reshape(1, -1), axis=1), label="attractiveness", color=color2, linewidth=1.7)
            # ax2.set_ylabel('attractiveness', color=color2)
            # ax2.tick_params(axis='y', labelcolor=color2)
            # fig1.tight_layout()


            plt.show()

    if mode == "debug" :

        a = 1
        b = 0.0001
        c = 0.0001

        # def h_G2(nu): # fonction issue du potentiel
        #     return 8 * (nu ** 7)
        
        # def hprim_G2(nu):
        #     return 56 * (nu ** 6)

        def h_G2(nu): # fonction issue du potentiel
            return 3 * c * (nu ** 2)

        def f(x, A, B):
            return x - A * np.exp(- h_G2(x) / B)
        
        def jacobian(x, A, B):
            return np.diag((1 + (A / B) * hprim_G2(x) * np.exp(- h_G2(x) / B)).ravel())



        def newton_debug(f, x0, jacobian, eps, args=()):
            x=x0
            f_value = f(x, *args)
            f_norm = np.linalg.norm(f_value, ord=2)  # l2 norm of vector
            n_iter = 0
            list_x = [x0[0]]
            list_delta = []

            while abs(f_norm) > (eps / 2) and n_iter < 100:

                delta = np.linalg.solve(jacobian(x, *args), -f_value)
                list_delta.append(delta)
                x = x + delta
                list_x.append(x[0])

                print("min_delta", np.min(delta))
                f_value = f(x, *args)
                f_norm = np.linalg.norm(f_value, ord=2)
                print("norm", f_norm, )
                n_iter += 1

            delta = np.linalg.solve(jacobian(x, *args), -f_value)
            list_delta.append(delta)

            # Here, either a solution is found, or too many iterations
            if abs(f_norm) > eps:
                # n_iter = -1
                print("Newton failed to converge")
            return x, list_x#, n_iter


        def dichotomy_debug(f, lowerb, upperb, eps, args=()):
            
            list_x = []
            print(np.max(f(lowerb, *args)) <= 0, np.min(f(upperb, *args)) >= 0)
            while np.linalg.norm(lowerb - upperb) > eps:
                middles = (lowerb + upperb) / 2
                list_x.append(middles[0])
                f_middles = f(middles, *args)
                upperb = np.where(f_middles >= 0, middles, upperb)
                lowerb = np.where(f_middles <= 0, middles, lowerb)

            return (lowerb + upperb) / 2, list_x

        res, list_x = dichotomy_debug(f, np.zeros((1, 1)), np.ones((1, 1)), eps=10 ** (-3), args=(a, b))#newton_debug(f, np.zeros((1, 1)), jacobian, eps=10 ** (-3), args=(a, b))

        # print(list_x)


        axis = np.linspace(0, 2, 1000)
        plt.plot(axis, f(axis, a, b))
        plt.scatter(list_x, f(np.array(list_x), a, b))
        plt.grid(True)
        plt.show()

