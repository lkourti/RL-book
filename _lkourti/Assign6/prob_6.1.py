import matplotlib.pyplot as plt
from numpy import linspace

def opt_z(alpha,mu=0.1,r=0.07,sigma=0.18):
    return (mu-r)*(1-alpha*(1+r)) / (alpha*((mu-r)**2 + sigma**2))

if __name__ == '__main__':
    mu = 0.1
    r = 0.07
    sigma = 0.18
    l_lim = (mu-r) / ((mu-r)**2 + sigma**2 + (mu-r)*(1+r))
    u_lim = 1 / (1+r)
    x = [i for i in linspace(l_lim, u_lim)]
    z = [opt_z(i) for i in x]
    plt.scatter(x,z, s=7)
    plt.ylabel("z in millions")
    plt.xlabel("alpha")
    plt.show()