import numpy as np
import matplotlib.pyplot as plt
from mathutils import gaussian
#from config import MU_1, SIGMA_1, MU_2, SIGMA_2, N_POINTS
#from scipy.optimize import newton

N_POINTS = 1000

## distribution initiale
MU_1 = 4
SIGMA_1 = .7
MU_2 = 12
SIGMA_2 = .7
L_INTERVALLE =16

EPS = 0.5
def newton(f, x0, jacobian, eps, args=()):
    x=x0
    f_value = f(x, *args)
    f_norm = np.linalg.norm(f_value, ord=2)  # l2 norm of vector
    n_iter = 0
    while abs(f_norm) > (eps / 2) and n_iter < 100:
        delta = np.linalg.solve(jacobian(x, *args), -f_value)
        x = x + delta
        f_value = f(x, *args)
        f_norm = np.linalg.norm(f_value, ord=2)
        n_iter += 1

    # Here, either a solution is found, or too many iterations
    if abs(f_norm) > eps:
        # n_iter = -1
        print("Newton failed to converge")
    return x#, n_iter

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
        return theta * self.mu * np.nan_to_num(1 / sum_theta).reshape(-1, 1)#(self.mu.T @ theta).reshape(-1, 1) / np.sum(theta, axis=1).reshape(-1, 1) #TODO: vérifier dimension

    #prox 2
    def f_G2(self, nu, theta):
        # nu vecteur vertical
        return nu - np.sum(theta, axis=0).reshape(nu.shape) * np.exp( - (nu + self.phi.T @ nu) / self.eps)

    def jacobian_f_G2(self, nu, theta):
        #nu est vertical
        A = np.sum(theta, axis=0).reshape(nu.shape) * np.exp(- (nu + self.phi.T @ nu) / self.eps)
        return np.eye(nu.size) + ((np.eye(nu.size) + self.phi) / self.eps) * np.tile(A, (1, nu.size))

    def prox_G2_KL(self, theta):  
        assert np.sum(phi ** 2) < 1 #TODO remplacer par un erreur
        nu = newton(self.f_G2, np.zeros((theta.shape[1], 1)), self.jacobian_f_G2, self.eps, args=(theta,))
        return theta * (np.exp(-(nu + nu.T @ self.phi) / self.eps))

    #prox 3
    def f_G3(self, nu, theta):
        return nu - np.sum(theta, axis=0).reshape(nu.shape) * np.exp(- self.h_G3(nu) / self.eps)

    def jacobian_f_G3(self, nu, theta):
        A = np.sum(theta, axis=0).reshape(nu.shape) * np.exp(- self.h_G3(nu) / self.eps)
        return np.eye(nu.size) + self.hprim_G3(mu) * np.tile(A, (1, nu.size))

    def prox_G3_KL(self, theta):
        #TODO: newton pour calculer nu
        nu = newton(self.f_G3, np.zeros((theta.shape[1], 1)), self.jacobian_f_G3, self.eps, args=(theta,))
        return theta * np.exp(- self.h_G3(nu) / self.eps)

    #Prox cyclique
    def prox_Gn_KL(self, n, theta):
        if n % 3 == 1:
            return self.prox_G1_KL(theta)

        elif n % 3 == 2:
            return self.prox_G2_KL(theta)

        elif n % 3 == 0:
            return self.prox_G3_KL(theta)

    #PROXIMAL ALGO
    def fit(self):
        L = 3
        gamma = np.exp(-self.costs / self.eps)
        z = [np.ones((self.mu.size, self.costs.shape[1])) for _ in range(L)]
        gap = np.inf
        n = 1
        # print("gamma", np.linalg.norm(gamma))
        while gap > self.eps or n%L != 1: # todo : trouver une condition sur le gap qui ne soit pas dépendante du nombre de points
            print(n)
            new_gamma = self.prox_Gn_KL(n, gamma * z[n % L])
            z[n % L] = z[(n-1) % L] * (gamma / new_gamma)
            print("new_gamma : ", np.linalg.norm(new_gamma, ord=1), new_gamma.shape)
            print("gamma : ", np.linalg.norm(gamma, ord=1), new_gamma.shape)
            gap = np.linalg.norm(gamma - new_gamma, ord=np.inf)
            print("gap : ", gap)
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
        return np.eye(nu.size) + self.hprim_G2(mu) * np.tile(A, (1, nu.size))

    def prox_G2_KL(self, theta):
        nu = newton(self.f_G2, np.zeros((theta.shape[1], 1)), self.jacobian_f_G2, self.eps, args=(theta,))
        return theta * np.exp(- self.h_G2(nu) / self.eps)

    def prox_Gn_KL(self, n, theta):
        if n % 3 == 1:
            return self.prox_G1_KL(theta)

        elif n % 3 == 2:
            return self.prox_G2_KL(theta)

    def promimal_step(self, costs):
        L = 2
        gamma = np.exp(-costs / self.eps) # TODO: revoir point de départ
        z = [np.ones((self.mu.size, self.costs.shape[1])) for _ in range(L)]
        gap = np.inf
        n = 1
        while gap > self.eps or n%L != 1: # todo : trouver une condition sur le gap qui ne soit pas dépendante du nombre de points
            print(n)
            new_gamma = self.prox_Gn_KL(n, gamma * z[n % L])
            z[n % L] = z[(n-1) % L] * (gamma / new_gamma)
            # print("new_gamma : ", np.linalg.norm(new_gamma, ord=1), new_gamma.shape)
            # print("gamma : ", np.linalg.norm(gamma, ord=1), new_gamma.shape)
            gap = np.linalg.norm(gamma - new_gamma, ord=np.inf)
            # print("gap : ", gap)
            gamma = new_gamma
            n += 1
        
        return gamma, np.sum(gamma, axis=0)

    def fit(self):
        nu = np.ones((self.costs.shape[1], 1))
        nu = nu / np.sum(nu)
        gap = np.inf
        n_iter = 1
        while gap > self.eps:
            print(n_iter)
            new_costs = self.costs + np.tile(nu.T @ self.phi, (self.costs.shape[0], 1))
            gamma, new_nu = self.promimal_step(new_costs)
            gap = np.linalg.norm(new_nu - nu, ord=np.inf)
            nu = new_nu
            print("gap", gap)
            n_iter += 1
        return gamma, nu


if __name__ == "__main__":
    x = np.linspace(0, L_INTERVALLE, N_POINTS)
    mu = gaussian(MU_1, SIGMA_1, x) + gaussian(MU_2, SIGMA_2, x)
    mu = (mu / np.linalg.norm(mu, ord=1))

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

    def h_G2(nu): # fonction issue du potentiel
        return 8 * (nu ** 7)
    
    def hprim_G2(nu):
        return 56 * (nu ** 6)

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


    # plt.imshow(costs)
    # plt.colorbar()
    # plt.show()
    # print(np.exp( - costs / EPS).sum()/ 10000)