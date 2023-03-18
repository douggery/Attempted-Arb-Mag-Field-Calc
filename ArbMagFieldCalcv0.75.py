#############################################################################
# Doug Bopp
# Last Updated 10/25/2017


#This program is meant to simulate the magnetic field originating from
#two planes which are separated by a distance z and which contain
#electrode geometries which produce arbitrary magnetic fields. In order
#to develop this, an optimization algorithm will be used to find a suitable
#geometry. Two particular cases are of interest will guide this project: 
#a spatially uniform magnetic field and a magnetic gradient as would be used 
#in a magneto-optical trap. The constraints on these goal magnetic field
#configurations have azimuthal symmetry which restricts the basis set which
#may be used to ones containing this same symmetry.

#This project can grow nearly without bound so it should contain focus points.
#The following is a simple description of interesting milestones:
#1) calculate the magnetic field off axis of a current loop.
#2) develop an algorithm to optimize two coils of variable position and radii
#   and fixed height to produce a desired magnetic field. Replicate the Helmholtz
#   coil configuration as a known solution. 
#3) add arrays of non-intersecting current loops
#4) develop a code to calculate the magnetic field from an arbitrary line
#   current at any point within a given volume
#5) develop a route-finding algorithm to generate non-overlapping electrode
#   configurations
#6) combine steps 4) and 5) with a suitable algorithm to develop magnetic
#   field geometries
#7) potentially expand the optimiztion algorithm. Two obvious candidates
#   are evolutionary strategy algorithms and downhill simplex methods
#8) improvement in accuracy and runtime may be required. For accuracy, Richardson
#   extrapolation should be used. Runtime efficiency may be improved by using
#   a generalized finite current wire in nondimensionalized units with a 
#   function to translate the coordinates.

#############################################################################
#               To Do List

# Add two plots, one for closeness of helmholtz configuration one for fit
# Add a changing mutation rate as a function of iteration number
# Work on the fitness function, specifically focus on the gradient
# Double check the negative x for the magnetic field magnitude
# Convert the magnetic field components in radial into x,y,z
# Add magnetic field of a line current

#############################################################################

import matplotlib.pyplot as plt
import random
from scipy.special import ellipk, ellipe, ellipkm1
from numpy import pi, sqrt, linspace
import numpy as np
from pylab import plot, xlabel, ylabel, suptitle, legend, show

#############################################################################
#               System Parameters

# z=5*10**-3      # mm in separation between planes
# Rmean=9*10**-3  # average expected radius
# Rmax=15*10**-3  # maximum radius in mm of a loop
# xmax=5*10**-3   # max shift in x for the center of a loop
# ymax=5*10**-3   # max shift in y for the center of a loop
# I=1*10**0       # 1 A of excitation current

z=5*10**-3     # mm in separation between planes
Rmean=5  # average expected radius
Rmax=15  # maximum radius in mm of a loop
xmax=5   # max shift in x for the center of a loop
ymax=5  # max shift in y for the center of a loop
I=1*10**0 

#############################################################################
#               Magnetic Field Calculation Math

# The magnetic field of a current-carrying loop can be calculated analytically
# using elliptic integrals which are built into scipy. For the source code
# that inspired this code see http://nbviewer.jupyter.org/github/tiggerntatie/emagnet-py/blob/master/offaxis/off_axis_loop.ipynb

uo = 4E-7*pi                            # Permeability [H/m]
Bo = lambda i, a, u=uo: i*u/2./a        # Central field = f(current, loop radius, perm. constant)
Alpha = lambda r, a: r/a                # Alpha = f(radius of measurement point, radius of loop)
Beta = lambda x, a: x/a                 # Beta = f(axial distance to meas. point, radius of loop)
Gamma = lambda x, r: x/r                # Gamma = f(axial distance, radius to meas. point)
Q = lambda r, x, a: (1 + Alpha(r,a))**2 + Beta(x,a)**2 +0.0000001   # Q = f(radius, distance to meas. point, loop radius)
k = lambda r, x, a: sqrt(4*Alpha(r,a)/Q(r,x,a))       # k = f(radius, distance to meas. point, loop radius)
K = lambda k: ellipk(k**2.0)            # Elliptic integral, first kind, as a function of k, scipy takes in m=k**2
E = lambda k: ellipe(k**2.0)            # Elliptic integral, second kind, as a function of k, scipy takes in m=k**2

# On-Axis field = f(current and radius of loop, x of measurement point)

def Baxial(i, a, x, u=uo):
    if a == 0:
        if x == 0:
            return NaN
        else:
            return 0.0
    else:
        return (u*i*a**2)/2.0/(a**2 + x**2)**(1.5)

# Axial field component = f(current i, loop radius a, r and x of desired point)

def Bx(i, a, x, r):
    if r<0:
        r=abs(r)
    #if x<0:
    #    x=abs(x)
    if r == 0:
        if x == 0:
            return Bo(i,a)         # central field
        else:
            return Baxial(i,a,-x)   # axial field
    else:                          # axial component, any location
        return Bo(i,a)*\
            (E(k(r,x,a))*((1.0-Alpha(r,a)**2-Beta(x,a)**2)/(Q(r,x,a)-4*Alpha(r,a))) + K(k(r,x,a)))\
            /pi/sqrt(Q(r,x,a))
        
# Radial field component = f(current and radius of loop, r and x of meas. point)

def Br(i, a, x, r):
    sign=1
    if r>0:
        r=abs(r)
        sign*=-1
    if r<0:
        r=abs(r)
    #if x<0:
    #    x=abs(x)
    if r == 0:
        return 0                   # no radial component on axis!
    else:                          # radial component, any location other than axis.
        return sign*Bo(i,a)*Gamma(x,r)*\
            (E(k(r,x,a))*((1.0+Alpha(r,a)**2+Beta(x,a)**2)/(Q(r,x,a)-4*Alpha(r,a))) - K(k(r,x,a)))\
            /pi/sqrt(Q(r,x,a))


A= lambda i,a,x,r: [Bx(i,a,x,r)+Bx(i,a,(x-z),r),Br(i,a,x,r)+Br(i,a,(x-z),r)]

#############################################################################

#               Genetic Algorithm Code

# Generate a population of individuals. At this point, the individuals have
# the following parameters:  radius of each coil, z dimension of each coil and 
# sign of the current in the loop. 
# Each individual should have the following parameter list: 
# individualX=[I1,R1,z1,I2,R2,z2] 

def gen_population(pop_num,param_num=6):
    pop=np.zeros((pop_num,param_num+1))
    for i in range(pop_num):
        pop[i]=[1,abs(random.gauss(Rmean,Rmax/5)),0,1,abs(random.gauss(Rmean,Rmax/5)),5,0]
    return pop

# Test the fitness of a population of individuals, order them by fitness
# The field strength from the two coils along the axis is given by 
# Bx(pop[i][0]*I,pop[i][1],pop[i][2],0)

def test_fitness(pop,param_num=6):
    for i in range(pop.shape[0]):
        Bfield_at_point= lambda h:(Bx(pop[i][0]*I,pop[i][1]*10**-3,pop[i][2]*10**-3+h,0)+Bx(pop[i][3]*I,pop[i][4]*10**-3,pop[i][5]*10**-3-h,0))
        A=Bfield_at_point(2.48*10**-3)
        B=Bfield_at_point(2.495*10**-3)
        C=Bfield_at_point(2.50*10**-3)
        D=Bfield_at_point(2.505*10**-3)
        pop[i][pop.shape[1]-1]=1*((pop[i][1]-5)**2+(pop[i][4]-5)**2)
    pop=np.asarray(sorted(pop,key=lambda x: x[6]))
    return pop

# Reproduce using the most fit individuals and randomly selecting traits
# among this set of fit individuals

def reproduce(pop,percent_survival):
    num_of_survivors=pop.shape[0]//(100/percent_survival)
    parents=np.zeros((num_of_survivors,pop.shape[1]))
    for i in range(num_of_survivors):
        parents[i]=pop[i]
    children=np.zeros((pop.shape[0]-num_of_survivors,pop.shape[1]))
    for i in range(pop.shape[0]-num_of_survivors):
        child=np.zeros(pop.shape[1])
        P1=parents[np.random.randint(0,num_of_survivors-1)]
        P2=parents[np.random.randint(0,num_of_survivors-1)]
        P1andP2=np.stack([P1,P2])
        randints=np.random.choice([0,1],2)      #gen a list of random ints with a size of two which is the number of chromosomes
        child=np.concatenate((np.split(P1andP2[randints[0]],[3])[0],np.split(P1andP2[randints[1]],[3])[1]))
        #individual[j]=np.random.choice([parents[np.random.randint(0,num_of_survivors-1)][j],parents[np.random.randint(0,num_of_survivors-1)][j]])
        children[i]=child
    newpop=np.concatenate((parents,children))
    return newpop

# Mutate the invidual traits in order to expand the search

def mutate(pop,mutation_rate,mutation_step):
    newpop=pop
    for i in range(pop.shape[0]):
        for j in [1,4]:
            if np.random.uniform(0,1)<mutation_rate:
                newpop[i][j]=pop[i][j]*(1+random.gauss(0,mutation_step))
            else:
                newpop[i][j]=pop[i][j]
    return newpop

def calc_avg_fitness(pop):
    sum=0
    for i in range(pop.shape[0]):
        sum+=1*pop[i][pop.shape[1]-1]
    sum=sum/pop.shape[0]
    return sum

#############################################################################
#               Running The Code And Plotting 

def run_GA(N):
    pop=gen_population(50)
    pop=test_fitness(pop)
    print(pop)
    fitness=np.zeros(N)
    for i in range(N/2):
        pop=test_fitness(pop)
        fitness[i]=calc_avg_fitness(pop)
        pop=reproduce(pop,25)
        pop=mutate(pop,0.25,0.45)
    print("Halfway Done! Reducing the mutation rate and mutation step size!")
    for i in range(N/2,N):
        pop=test_fitness(pop)
        fitness[i]=calc_avg_fitness(pop)
        pop=reproduce(pop,25)
        pop=mutate(pop,0.1,0.05)
    pop=test_fitness(pop)
    return fitness,pop




M=100

output=run_GA(M)
fit=output[0]
fitpop=output[1]

print(fit)


print("The five fittest individuals are:")

for i in range(5):
    print(fitpop[i])

iterations=np.linspace(0,M,fit.shape[0])

plt.close("all")

f,ax =plt.subplots()
ax.plot(iterations,fit)
ax.set_title('Simple plot of fitness vs iteration num')
axes = plt.gca()
axes.set_ylim([min(fit),max(fit)])
axes.set_yscale('log')

plt.show()
#############################################################################
# axiallimit = z # meters from center
# radiallimit = Rmax*10**-3 # maximum radius to investigate
# curveqty = 5
# X = linspace(-axiallimit, 2*axiallimit)
# R = linspace(0, radiallimit, curveqty)
# [plot(X, [A(1*10**-3,5*10**-3,x,r)[0] for x in X], label="r={0}".format(r)) for r in R]
# #[plot(X, [Bx(-1*10**-3,5*10**-3,x,r) for x in X], label="r={0}".format(r)) for r in R]
# xlabel("Axial Position (m)")
# ylabel("Axial B field (T)")
# suptitle("Axial component of unit coil (5*10**-3 radius, 1mA current) B field for various measurement radii")
# legend()
# show()

#############################################################################
#   Quiver Plotting the Magnetic Field

nx=50
nz=50
x=np.linspace(-50,50)
z=np.linspace(-50,50)
X,Z=np.meshgrid(x,z)

B_x,B_z=np.zeros((nz,nx)),np.zeros((nz,nx))

for i in range(nx):
    for j in range(nz):
        B_x[i][j]=Br(-100,20,Z[i][j],X[i][j])+0*Br(-1,1,X[j][i],Z[j][i])
        B_z[i][j]=Bx(100,20,Z[i][j],X[i][j])+0*Bx(1,1,X[j][i],Z[j][i])


print(B_x[24,24])
print(B_x[25,35])
print(B_x[25,15])
print(B_x[20,25])
print(B_x[30,25])



# fig=plt.figure()
# ax=fig.add_subplot(211)
# color1=np.log(np.sqrt(B_x**2+B_z**2))
# ax.streamplot(x,z,B_x,B_z,color='k',linewidth=1,cmap=plt.cm.inferno,density=2,arrowstyle='->',arrowsize=1.5)

# # ax.set_xlabel('$x$')
# # ax.set_ylabel('$y$')
# # ax.set_xlim(-20,20)
# # ax.set_ylim(-20,20)
# # ax.set_aspect('equal')
# plt.show()



#############################################################################
#Useful testing bits



#np.c_[np.array([1,2,3]), np.array([4,5,6])]
#np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])]

# f,ax =plt.subplots()

# plt.figure(1)

# plt.subplot(422)
# plt.streamplot(x,z,B_x,B_z,color='k',linewidth=1,cmap=plt.cm.inferno,density=2,arrowstyle='->',arrowsize=1.5)
# plt.subplot(421)
# ax.plot(iterations,fit)


# plt.subplot(423)
# plt.imshow(DoubleSlit)
# plt.subplot(424)
# plt.imshow(abs(fftshift((fft2(DoubleSlit)))))
# #plt.subplot(224)
# #plt.imshow(abs(fftshift((fft2(np.multiply(Gauss,KnifeEdge))))))

# DoubleSlitWide=add_slit(10,30)+add_slit(10,-30)
# plt.subplot(425)
# plt.imshow(KnifeEdge)
# plt.subplot(426)
# plt.imshow(abs(fftshift(ifft2((np.multiply(fft2(KnifeEdge),fft2(Gauss)))))))

# DoubleSlitNarrow=add_slit(5,5)+add_slit(5,-5)
# plt.subplot(427)
# plt.imshow(abs(fftshift((fft2(DoubleSlitNarrow)))))
# plt.subplot(428)
# plt.imshow(abs(ifft2((abs(fft2(DoubleSlitNarrow))))))

#plt.show()