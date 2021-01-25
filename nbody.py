import numpy as np
import matplotlib.pyplot as plt

def acceleration( pos, mass, G, mindist ):

    """ Calculate acceleration of particles
    pos is a Nx3 matrix
    mass is a Nx1 matrix
    G is Newton's constant
    mindist is to avoid division by zero
    returns Acceleration as Nx3 matrix
    """

    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]

    # matrix that stores r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    den = (dx**2 + dy**2 + dz**2 + mindist**2)

    den[den>0]=den[den>0]**(-1.5)

    ax = G * dx * den @ mass 
    ay = G * dy * den @ mass
    az = G * dz * den @ mass

    a = np.hstack((ax,ay,az))

    return a

def getEnergy ( pos, vel, mass, G):
    """ Calculate total energy
    pos is a Nx3 matrix
    vel is a Nx3 matrix
    mass is a Nx1 matrix
    G is gravitational constant
    """

    # kinetic energy
    KE = 0.5*np.sum(np.sum(mass*vel**2))

    # potential energy
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]
    # matrix that stores r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    den = (dx**2 + dy**2 + dz**2)

    den[den>0]=den[den>0]**(-0.5)
    
    # sum over upper triangle, to count each interaction only once
    PE = G*np.sum(np.sum(np.triu(-(mass*mass.T)*den,1)))

    return KE, PE


def main():
    """ Simulation"""

    # Parameters:
    N = 100
    t = 0.
    tEnd = 10.
    dt = 0.01
    softening = 0.1
    G = 1.0 # Newton's G Constant
    plotRealTime = True

    #Generate initial conditions
    np.random.seed(17)

    mass = 20*np.ones((N,1))/N
    pos = np.random.randn(N,3)
    vel = np.random.randn(N,3)

    # Convert vel to center of mass frame
    #vel -= np.mean(mass*vel,0) / np.mean(mass)

    # Get initial acceleration (First one)
    acc = acceleration(pos, mass, G, softening)
    
    # Get initial energy
    KE, PE = getEnergy(pos, vel, mass, G)

    # Time steps
    Nt = int(np.ceil(tEnd/dt))

    # save energy and partile orbits for plots
    pos_save = np.zeros((N,3,Nt+1))
    pos_save[:,:,0] = pos
    KE_save = np.zeros(Nt+1)
    KE_save[0] = KE
    PE_save = np.zeros(Nt+1)
    PE_save[0] = PE
    t_all = np.arange(Nt+1)*dt
    
    #prep figure
    fig = plt.figure(figsize=(6,7),dpi = 80)
    grid = plt.GridSpec(3,1,wspace =0.0, hspace =0.3)
    ax1 = plt.subplot(grid[0:2,0])
    ax2 = plt.subplot(grid[2,0])

    # Simulation loop
    for i in range(Nt):
        #new vel    half way
        vel += dt*acc/2.
        #new pos
        pos += vel*dt
        #new acc
        acc = acceleration(pos,mass,G,softening)
        # update vel another half way
        vel += dt*acc/2

        t+=dt
        
        #Get new energies
        KE,PE = getEnergy(pos,vel,mass,G)
        
        #save pos and energy
        pos_save[:,:,i+1] = pos
        KE_save[i+1]=KE
        PE_save[i+1]=PE

        # Plot in real time
        if plotRealTime  or (i==Nt-1):
            plt.sca(ax1)
            plt.cla()
            xx = pos_save[:,0,max(i-50,0):i+1]
            yy = pos_save[:,1,max(i-50,0):i+1]
            plt.scatter(xx,yy,s=1,color=[.7,.7,1.])
            plt.scatter(pos[:,0],pos[:,1],s=10,color='blue')
            ax1.set(xlim=(-2,2),ylim=(-2,2),xlabel=('x'),ylabel=('y'))
            ax1.set_aspect('equal','box')
            ax1.set_xticks([-2,-1,0,1,2])
            ax1.set_yticks([-2,-1,0,1,2])

            plt.sca(ax2)
            plt.cla()
            plt.scatter(t_all,KE_save,color='red',s=1,label='KE' if i == Nt-1 else "")
            plt.scatter(t_all,PE_save,color='blue',s=1,label='PE' if i == Nt-1 else "")
            plt.scatter(t_all,KE_save+PE_save,color='black',s=1,label='Etot' if i == Nt-1 else "")
            ax2.set(xlim=(0,tEnd),ylim=(-300,300))
            ax2.set_aspect(0.007)

            plt.pause(0.001)

    plt.sca(ax2)
    plt.xlabel('time')
    plt.ylabel('energy')
    ax2.legend(loc='upper right')

    #Save fig
    plt.savefig('nbody.png',dpi=240)
    plt.show()

    return 0

if __name__ == "__main__":
    main()
        
        

