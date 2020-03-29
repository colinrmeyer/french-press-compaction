# # compaction model # # 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import diags

class parameters():
    def __init__(self):
        # Mesh Information
        # domain from from z=a to z=b
        self.l = 0 # nondimensional bottom
        self.r = 1 # nondimensional top
        self.N = (2**6)*(self.r-self.l) # Number of grid cells
        self.dy = (self.r-self.l)/self.N # grid cell width
        self.yedges = np.linspace(self.l,self.r,self.N+1) # cell edges
        self.y = (self.yedges[:self.N]+self.yedges[-self.N:])/2 # cell centers
        self.n = 3 # plastic effective pressure exponent
        self.m = 2 # plastic effective pressure exponent
        self.a = 3 # kozeny-carman exponent
        self.b = 2 # kozeny-carman exponent
        self.chi0 = 0.01 # initial solid fraction
        self.gamma = 1000 # nondimensional compaction rate

def compactfun(time,c,Nmbr,dz,zedges,gamma,a,b,n,m):
    # Transmissability
    Dv = (n+(m-n)*c)*(c**(n-b))*((1-c)**(a-m-1))
    Td = (2/dz)*(((1/Dv[:-1])+(1/Dv[1:]))**(-1)) # Transmissability vector
    Tplus = np.append(Td,0)
    Tminus = np.append(0,Td)
    
    # Compile into diffusion matrix
    diagonals = [np.transpose(Td), -(Tminus+Tplus), np.transpose(Td)]
    D = diags(diagonals,[-1,0,1],[Nmbr,Nmbr])

    # Boundary condition vector
    bcs = np.zeros(Nmbr)
    bcs[-1] = c[-1] # Fixed velocity, z=1
    bcs[0] = 0 # Fixed velocity, z=0
    
    # advection
    hdot = -1
    apart = np.append(c[1:],c[-1])*zedges[1:] - np.append(c[0],c[:-1])*zedges[:-1]
    advf = hdot*(apart-c*dz)
    advf[0]=0
    advf[-1]=0
    h = 1-time # height with time
    dcdt = (1/(h*dz))*(advf + (gamma/h)*D@c + bcs)
    return dcdt
    
if __name__ == "__main__":
    prms = parameters()
    chi = prms.chi0*np.ones(prms.N) # initial solid fraction profile
    sol = solve_ivp(lambda t,x: compactfun(t,x,prms.N,prms.dy,prms.yedges,prms.gamma,prms.a,prms.b,prms.n,prms.m),[0,0.98],chi,method='LSODA')
    
    # plot the solution
    f = plt.figure()
    for i in range(np.size(sol.t)):
        h = 1-sol.t[i] 
        plt.semilogx(sol.y[:,i],prms.y*h,'k-')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.ylim(0,1)
        plt.xlim(1e-2,1e0)
        plt.ylabel('depth, $z/h_0$')
        plt.xlabel('solid fraction, $1-\phi$')
    f.savefig("compact_slow.pdf")
