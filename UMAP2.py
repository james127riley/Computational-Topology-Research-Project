import requests, json
from pprint import pprint
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import kneighbors_graph
import pandas as pd
import random
import math
from numpy import linalg as LA
from numpy import random
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from scipy.linalg import sqrtm
from sklearn.manifold import SpectralEmbedding
import generate_datasets
pd.set_option('display.max_columns', None)


class UMAP():

    def __init__(self,base_url, target_dimension, neighbours, epochs, min_dist, alpha0, points, resolution, bound):
        #print("Generating data")
        loop = self.generate_loop(3,points)
        #dpoints,labels = generate_datasets.make_point_clouds(1,int(math.sqrt(points)),0.1)
        #torus = np.array(dpoints[2])
        #print(torus.shape)
        print(loop.shape)
        #points = 48
        #players = self.get_player_data(base_url)[:points]
        data = loop
        print(data.shape)
        size = data.shape[0]
        fs_set = self.process_matrix(data,neighbours,size,points)
        fs_set = self.make_symmetric(fs_set,size)
        scaler = MinMaxScaler()
        #print("Creating spectral embedding")
        LD_candidate2 = self.spectral_embedding(fs_set,target_dimension)
        LD_candidate = scaler.fit_transform(pd.DataFrame(LD_candidate2))
        #print("optimising")
        plots = np.array(self.optimise_embedding(LD_candidate[:],fs_set,min_dist,epochs,alpha0, resolution, bound, target_dimension))
        #print("plotting")
        self.calculate_homology(data)
        self.calculate_homology(LD_candidate2)
        self.calculate_homology(plots[-1])
        if target_dimension == 2:
            self.plot_LD_2D(plots[-1],"UMAP")
        elif target_dimension == 3:
            self.plot_LD_3D(plots[-1],"UMAP")
        self.plot_all_322(data,LD_candidate,plots[-1])
        #self.plot_key_LD(LD_candidate,plots[-1])
        self.plot_all_stages(LD_candidate,plots)
        self.LD = plots[-1].copy()

    def get_LD(self):
        return self.LD

    def local_fuzzy_simplicial_set(self,X,knn_indices,knn_dists,size,i,n):
        x = X[i] 
        p = knn_dists[0]
        sigma = self.smoothKNNdist(knn_dists,n,p)
        #print(sigma)
        #fs_set0 = X.copy()
        fs_set1 = np.array([[i,j,0] for j in range(size)])
        for j in knn_indices:
            d_xy=max(0,LA.norm(x-X[j])-p)/sigma
            fs_set1 = np.vstack([fs_set1,[i,j,math.exp(-d_xy)]])
        return fs_set1

    def smoothKNNdist(self,knn_dists,n,p):
        sigma = 1
        ep = 0.01
        uv = self.smooth_dist(knn_dists,p,ep)
        lv = self.smooth_dist(knn_dists,p,-ep)
        if uv<lv:
            sig_lb = ep
            sig_ub = 2*ep
            lv = self.smooth_dist(knn_dists,p,sig_lb)
            uv = self.smooth_dist(knn_dists,p,sig_ub)
            while (uv-math.log2(n))*(math.log2(n)-lv) < 0:
                sig_ub += 1
                uv = self.smooth_dist(knn_dists,p,sig_ub)
            sigma = self.binary_search(knn_dists,p,n,sig_lb,sig_ub)
        else:
            sig_ub = -ep
            sig_lb = -2*ep
            lv = self.smooth_dist(knn_dists,p,sig_lb)
            uv = self.smooth_dist(knn_dists,p,sig_ub)
            while (uv-math.log2(n))*(math.log2(n)-lv) < 0:
                sig_lb -= 1
                lv = self.smooth_dist(knn_dists,p,sig_lb)
            sigma = self.binary_search(knn_dists,p,n,sig_lb,sig_ub)
        return sigma

    def smooth_dist(self,knn_dists,p,sigma):
        return sum([math.exp(-(d-p)/sigma) for d in knn_dists])

    def binary_search(self,knn_dists,p,n,lb,ub):
        ep = 0.01
        mb = (lb+ub)/2
        mv = self.smooth_dist(knn_dists,p,mb)
        lv = self.smooth_dist(knn_dists,p,lb)
        uv = self.smooth_dist(knn_dists,p,ub)
        while abs(mv-math.log2(n))>ep:
            if (mv-math.log2(n))*(lv-math.log2(n))>0:
                lb = mb
                mb = (mb+ub)/2
            else:
                ub = mb
                mb = (mb+lb)/2
            mv = self.smooth_dist(knn_dists,p,mb)
            lv = self.smooth_dist(knn_dists,p,lb)
        return mb

    def process_matrix(self,data,neighbours,size,points):
        knn_graph = kneighbors_graph(data, neighbours, mode="distance",p=2,include_self=False).toarray()
        #print(knn_graph[:10])
        fs_set1=[]
        for i in range(size):
            knn,knn_dists = self.get_neighbours(knn_graph,i)
            #print(knn)
            fs_set1.append(self.local_fuzzy_simplicial_set(data,knn,knn_dists,size,i,neighbours))
        fs_set1 = np.array(fs_set1)
        #print(fs_set1.shape)
        for i in range(size):
            for n in range(neighbours):
                line = fs_set1[i][-n-1]
                #print(line)
                fs_set1[i][int(line[1])][2] = line[2]
        fs_set = fs_set1[0][:size]
        for i in range(1,size):
            fs_set = np.vstack([fs_set,fs_set1[i][:size]])
        print(fs_set.shape, points)
        fs_set = fs_set.reshape(points,points,3)
        return fs_set

    def get_neighbours(self,knn,i):
        line = knn[i].copy()
        indices = np.array(np.nonzero(line)).flatten()
        line_sort = np.sort(line[np.nonzero(line)])
        return (indices,line_sort)
    
    def make_symmetric(self, fs_set,size):
        for i in range(size):
            for j in range(i,size):
                d1 = fs_set[i][j][2]
                d2 = fs_set[j][i][2]
                d3 = d1+d2-d1*d2
                fs_set[i][j][2] = d3
                fs_set[j][i][2] = d3
        return fs_set

    def get_player_data(self,url):
        r = requests.get(url+'bootstrap-static/.').json()
        players = pd.json_normalize(r['elements'])
        teams = pd.json_normalize(r['teams'])
        df = pd.merge(left=players,
                    right=teams,
                    left_on='team',
                    right_on='id')
        df = df.rename(columns={'name':'team_name'})
        filters = (df['minutes'] > 0)
        players = df.loc[filters,['first_name', 'second_name','team_name','goals_scored','minutes','now_cost','assists','creativity']]
        players['first_name'] = players['first_name'].str.replace(" ","")
        players['second_name'] = players['second_name'].str.replace(" ","")
        players['tag'] = players['first_name'].str[:4]+'_'+players['second_name'].str[:4]+'_'+players['team_name']
        players.drop(columns=['first_name','second_name','team_name'],inplace=True)
        tags = list(players['tag'])
        players.drop(columns=['tag'],inplace=True)
        players.reset_index(drop=True, inplace=True)
        scaler = MinMaxScaler()
        players_scaled = scaler.fit_transform(players)
        print(players_scaled.shape)
        print(players.shape)
        return players_scaled
    
    def generate_loop(self,d, p):
        points = np.zeros((p,d))
        current = np.zeros(d)
        for i in range(d):
            vector = np.zeros(d)
            vector[i]=1
            for j in range(math.floor(0.5*p/d)):
                points[i*math.floor(0.5*p/d)+j] = np.round(current.copy(),2)
                current = np.add(vector,current)
        for i in range(d):
            vector = np.zeros(d)
            vector[i]=-1
            for j in range(math.floor(0.5*p/d)):
                points[int(0.5*p+i*math.floor(0.5*p/d)+j)] = np.round(current.copy(),2)
                current = np.add(vector,current)
        return points

    def spectral_embedding(self,fs_set,d):
        A = self.one_skeleton(fs_set)
        n = A.shape[0]
        D = np.zeros(A.shape)
        for i in range(n):
            D[i][i] = np.sum(A[i])
        L = np.matmul(np.matmul(LA.inv(sqrtm(D)),np.subtract(D,A)),LA.inv(sqrtm(D)))
        evals,evects = LA.eig(L)
        evects = np.transpose(evects)
        ar = np.c_[evals,evects]
        indices = np.argsort(evals)
        ar = ar[indices]
        evects = evects[indices]
        LD = evects[1:d+1]
        LD = np.transpose(LD)
        return LD
    
    def one_skeleton(self,fs_set):
        size = fs_set.shape[0]
        A = np.zeros((size,size))
        for i in range(size):
            for j in range(i, size):
                A[i][j] += fs_set[i][j][2]
                A[j][i] += fs_set[i][j][2]
        return A

    def phi(self,X,a,b):
        n = LA.norm(X)
        return 1/(1+a*(n**(2*b)))

    def psi(self,X, min_dist):
        n = LA.norm(X)
        if n <= min_dist: return 1
        else: return math.exp(-n+min_dist)

    def fit_phi_2D(self,min_dist,resolution,bd):
        seeds_x,seeds_y = np.meshgrid(np.linspace(-bd,bd,resolution+1),np.linspace(-bd,bd,resolution+1))
        seeds_x = seeds_x.flatten()
        seeds_y = seeds_y.flatten()
        grid = np.c_[seeds_x,seeds_y]
        psi_output = []
        for i in range(resolution**2):
            psi_output.append(self.psi(grid[i],min_dist))
        params,cov = curve_fit(self.phi,grid, np.array(psi_output),p0=np.array([5,0.5]))
        return params

    def fit_phi_3D(self,min_dist,resolution,bd):
        seeds_x,seeds_y,seeds_z = np.meshgrid(np.linspace(-bd,bd,resolution+1),np.linspace(-bd,bd,resolution+1),np.linspace(-bd,bd,resolution+1))
        seeds_x = seeds_x.flatten()
        seeds_y = seeds_y.flatten()
        seeds_z = seeds_z.flatten()
        grid = np.c_[seeds_x,seeds_y,seeds_z]
        psi_output = []
        for i in range(resolution**3):
            psi_output.append(self.psi(grid[i],min_dist))
        params,cov = curve_fit(self.phi,grid, np.array(psi_output),p0=np.array([5,0.5]))
        return params
    
    def test_phi_2D(self,a,b,min_dist,resolution,bd):
        bd=1
        seeds_x,seeds_y = np.meshgrid(np.linspace(-bd,bd,resolution),np.linspace(-bd,bd,resolution))
        seeds_x = seeds_x.flatten()
        seeds_y = seeds_y.flatten()
        grid = np.c_[seeds_x,seeds_y]
        psis = []
        phis = []
        for i in range(resolution**2):
            psi_output = self.psi(grid[i],min_dist)
            phi_output = self.phi(grid[i],a,b)
            psis.append(psi_output)
            phis.append(phi_output)
        self.plot_2d(grid[:,0],grid[:,1],np.array(psis),np.array(phis))

    def test_phi_3D(self,a,b,min_dist,resolution, bd):
        bd=1
        seeds_x,seeds_y,seeds_z = np.meshgrid(np.linspace(-bd,bd,resolution),np.linspace(-bd,bd,resolution),np.linspace(-bd,bd,resolution))
        seeds_x = seeds_x.flatten()
        seeds_y = seeds_y.flatten()
        seeds_z = seeds_z.flatten()
        grid = np.c_[seeds_x,seeds_y,seeds_z]
        psis = []
        phis = []
        for i in range(resolution**3):
            psi_output = self.psi(grid[i],min_dist)
            phi_output = self.phi(grid[i],a,b)
            psis.append(psi_output)
            phis.append(phi_output)
        self.plot_3d(grid[:,0],grid[:,1],grid[:,2],np.array(psis),np.array(phis))

    def plot_2d(self,x,y,z1,z2):
        fig, ax = plt.subplots(2,2,subplot_kw={"projection":"3d"})
        x=x.flatten()
        y=y.flatten()
        surf1 = ax[0][0].plot_trisurf(x, y, z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        surf2 = ax[0][1].plot_trisurf(x, y, z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        surf3 = ax[1][0].plot_trisurf(x, y, np.log(z1), cmap=cm.coolwarm, linewidth=0, antialiased=False)
        surf4 = ax[1][1].plot_trisurf(x, y, np.log(z2), cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax[0][0].set_title("phi2")
        ax[0][1].set_title("phi")
        ax[1][0].set_title("grad_phi2")
        ax[1][1].set_title("grad_phi")
        fig.colorbar(surf1, shrink=0.5, aspect=5)
        fig.colorbar(surf2, shrink=0.5, aspect=5)
        fig.colorbar(surf3, shrink=0.5, aspect=5)
        fig.colorbar(surf4, shrink=0.5, aspect=5)
        plt.show()

    def plot_3d(self,x,y,z,z1,z2):
        fig1, ax1 = plt.subplots(3,subplot_kw={"projection":"3d"})
        fig2, ax2 = plt.subplots(3,subplot_kw={"projection":"3d"})
        x=x.flatten()
        y=y.flatten()
        z=z.flatten()
        surf1 = ax1[0].plot_trisurf(x, y, z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        surf2 = ax1[1].plot_trisurf(y, z, z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        surf3 = ax1[2].plot_trisurf(z, x, z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        surf4 = ax2[0].plot_trisurf(x, y, z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        surf5 = ax2[1].plot_trisurf(y, z, z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        surf6 = ax2[2].plot_trisurf(z, x, z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig1.colorbar(surf1, shrink=0.5, aspect=5)
        fig1.colorbar(surf2, shrink=0.5, aspect=5)
        fig1.colorbar(surf3, shrink=0.5, aspect=5)
        fig2.colorbar(surf4, shrink=0.5, aspect=5)
        fig2.colorbar(surf5, shrink=0.5, aspect=5)
        fig2.colorbar(surf6, shrink=0.5, aspect=5)
        plt.show()

    def grad_phi(self, X, Y, a, b):
        V = np.subtract(X,Y)
        size = V.shape[0]
        n = LA.norm(V)
        grad = []
        u = 1+a*(n**2)**b
        for i in range(size):
            grad_i = -(2*a*b*(n**2)**(b-1)*V[i])/u
            grad.append(grad_i)
        return np.array(grad)

    def grad_phi2(self, X, Y, a, b):
        V = np.subtract(X,Y)
        size = V.shape[0]
        n = LA.norm(V)
        grad = []
        u = 1+a*(n**2)**b
        for i in range(size):
            grad_i = (2*a*b*(n**2)**(b-1)*V[i])/(u*(u-1))
            grad.append(grad_i)
        return np.array(grad)

    def test_grad_phi_2D(self,a,b,min_dist,resolution,bd):
        X = np.linspace(-bd,bd,resolution)
        Y = np.linspace(-bd,bd,resolution)
        seeds_x,seeds_y = np.meshgrid(X,Y)
        seeds_x = seeds_x.flatten()
        seeds_y = seeds_y.flatten()
        grid = np.c_[seeds_x,seeds_y]
        gphis = np.zeros((resolution,resolution,2))
        gphis2 = np.zeros((resolution,resolution,2))
        for i in range(resolution):
            for j in range(resolution):
                gphis[i][j] = self.grad_phi(np.array([1,1])+grid[i*resolution+j],np.array([1,1]),a,b)
                gphis2[i][j] = self.grad_phi2(np.array([1,1])+grid[i*resolution+j],np.array([1,1]),a,b)
        self.plot_2d_grad(X,Y,gphis[:,:,0],gphis[:,:,1],gphis2[:,:,0],gphis[:,:,1])

    def plot_2d_grad(self,x,y,u1,u2,v1,v2):
        fig, ax = plt.subplots(2,1)
        x=x.flatten()
        y=y.flatten()
        X = np.arange(-10, 10, 1)
        Y = np.arange(-10, 10, 1)
        U, V = np.meshgrid(X, Y)
        #print(U[:10],V[:10])
        #u1 = np.reshape(u1,(50,50))
        surf1 = ax[0].quiver(x,y,u1,u2)
        surf2 = ax[1].quiver(x,y,v1,v2)
        ax[0].set_title("grad_phi")
        ax[1].set_title("grad_phi2")
        fig.colorbar(surf1, shrink=0.5, aspect=5)
        fig.colorbar(surf2, shrink=0.5, aspect=5)
        plt.show()

    def optimise_embedding(self, low_rep, top_rep, min_dist, epochs, alpha0, resolution, bound, target_dimension):
        LD = low_rep.copy()
        A = self.one_skeleton(top_rep)
        size = A.shape[0]
        if target_dimension==3:
            a,b = self.fit_phi_3D(min_dist,resolution,bound)[:2]
            #self.test_phi_3D(a,b,min_dist,resolution,bound)
        elif target_dimension==2:
            a,b = self.fit_phi_2D(min_dist,resolution,bound)[:2]
            #self.test_phi_2D(a,b,min_dist,resolution,bound)
            #self.test_grad_phi_2D(a,b,min_dist,resolution,bound)
        else: print("Target dimension ERROR")
        #print("a = {}\nb = {}\nmin_dist = {}\nsize = {}".format(a,b, min_dist,size))
        alpha = alpha0
        plots = []
        for e in range(epochs):
            pos_alt = 0
            neg_alt = 0
            pos_diff = 0
            neg_diff = 0
            alts = []
            direction = -1#+2*(e%2)
            for i in range(0,size,direction):
                for j in range(i,size,direction):
                    if A[i][j] == 0: continue
                    print(round(100*(e*size**2+i*size+j)/(epochs*size**2),2), end="\r")
                    r = random.random()
                    if r < A[i][j]:
                        pos_alt += 1
                        diff = self.grad_phi(LD[i],LD[j],a,b)
                        LD[i] = LD[i]+alpha*diff
                        alts.append([i,j,list(diff),"+"])
                        pos_diff += LA.norm(diff)
                        line = A[i]
                        zero_indices = np.array(np.nonzero(line==0)).flatten()
                        neg_samples = zero_indices.size
                        for n in range(size):
                            k = random.randint(0,neg_samples-1)
                            while zero_indices[k] == i:
                                k = random.randint(0,neg_samples-1)
                            l = zero_indices[k]                            
                            diff = self.grad_phi2(LD[i],LD[l],a,b)
                            LD[i] = LD[i]+alpha*diff
                            neg_alt += 1
                            neg_diff += LA.norm(diff)
                            alts.append([i,k,list(diff),"-"])
            alpha = alpha0*(1-0.5*e/epochs)
            if e%5==0:
                plots.append(LD.copy())
        return plots
    
    def plot_LD_3D(self, LD, title):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(LD[:,0],LD[:,1],LD[:,2])
        plt.title(title)
        plt.show()

    def plot_LD_2D(self, LD, title="plot"):
        fig = plt.figure()
        plt.scatter(LD[:,0],LD[:,1])
        for i in range(LD.shape[0]):
            plt.annotate(i,LD[i])
        plt.title(title)
        plt.show()

    def plot_all_322(self,HD,LD1,LD2):
        fig= plt.figure()
        ax1 = fig.add_subplot(2,2,1,projection="3d")
        ax1.scatter(HD[:,0],HD[:,1],HD[:,2])
        ax2 = fig.add_subplot(2,2,2)
        ax2.scatter(LD1[:,0],LD1[:,1])
        ax3 = fig.add_subplot(2,2,3)
        ax3.scatter(LD2[:,0],LD2[:,1])
        #for i in range(LD1.shape[0]):
        #    ax2.annotate(i,LD1[i])
        #    ax3.annotate(i,LD2[i])
        ax1.set_title("Initial")
        ax2.set_title("Spectral")
        ax3.set_title("Final")
        plt.subplots_adjust(wspace=0.5,hspace=0.5)
        plt.show()

    def plot_key_LD(self,LD1,LD2):
        fig= plt.figure()
        ax2 = fig.add_subplot(2,1,1)
        ax2.scatter(LD1[:,0],LD1[:,1])
        ax3 = fig.add_subplot(2,1,2)
        ax3.scatter(LD2[:,0],LD2[:,1])
        for i in range(LD1.shape[0]):
            ax2.annotate(i,LD1[i])
            ax3.annotate(i,LD2[i])
        ax2.set_title("Initial")
        ax3.set_title("Final")
        plt.subplots_adjust(wspace=0.5,hspace=0.5)
        plt.show()

    def plot_all_stages(self,LD1,plots):
        fig= plt.figure()
        nplots = plots.shape[0]+1
        ax = [0 for i in range(nplots)]
        h = math.ceil(math.sqrt(nplots))
        ax[0] = fig.add_subplot(h,h,1)
        ax[0].scatter(LD1[:,0],LD1[:,1])
        ax[0].set_title(0)
        for i in range(1,nplots-1):
            p = plots[i]
            ax[i] = fig.add_subplot(h,h,i+1)
            ax[i].scatter(p[:,0],p[:,1])
            #ax[i].annotate(i,p[i])
            ax[i].set_title(i)
        plt.subplots_adjust(wspace=0.5,hspace=1.5)
        plt.show()

    def calculate_homology(self,array):
        array = np.array([array])
        rips = VietorisRipsPersistence(metric="euclidean",homology_dimensions=[0,1,2])
        diagram = rips.fit_transform(array)
        fig = plot_diagram(diagram[0])
        fig.show()

class TestNN():

    def __init__(self,min,max,target_dim):
        if min % 2 == 0:
            print("K must be odd")
            return None
        plots = [0 for i in range(min,max+1,2)]
        for i in range(min,max+1,2):
            print(round(100*(i-min)/(max-min),2),end="\r")
            umap = UMAP(base_url,
                        target_dimension=target_dim,
                        neighbours=i,
                        epochs=100,
                        min_dist=0.01,
                        alpha0=1,
                        points=48,
                        resolution=50,
                        bound=4)
            print(umap.get_LD().shape)
            plots[int((i-min)/2)]=umap.get_LD()
        if target_dim == 2:
            self.plot_all_2D(np.array(plots),min)
        elif target_dim == 3:
            self.plot_all_3D(np.array(plots),min)
        
    def plot_all_2D(self,plots,min):
        fig= plt.figure()
        nplots = plots.shape[0]
        print(plots.shape)
        ax = [0 for i in range(nplots)]
        h = math.ceil(math.sqrt(nplots))
        for i in range(nplots):
            p = plots[i]
            ax[i] = fig.add_subplot(h,h,i+1)
            ax[i].scatter(p[:,0],p[:,1])
            #ax[i].annotate(i,p[i])
            ax[i].set_title(i*2+min)
        plt.subplots_adjust(wspace=0.5,hspace=1.5)
        plt.show()

    def plot_all_3D(self,plots,min):
        fig= plt.figure()
        nplots = plots.shape[0]
        print(plots.shape)
        ax = [0 for i in range(nplots)]
        h = math.ceil(math.sqrt(nplots))
        for i in range(nplots):
            p = plots[i]
            ax[i] = fig.add_subplot(h,h,i+1,projection="3d")
            ax[i].scatter(p[:,0],p[:,1],p[:,2])
            #ax[i].annotate(i,p[i])
            ax[i].set_title(i*2+min)
        plt.subplots_adjust(wspace=0.5,hspace=1.5)
        plt.show()

class TestEpochs():

    def __init__(self):
        pass

if __name__ =="__main__":
    base_url = 'https://fantasy.premierleague.com/api/'
    mode = 0
    if mode == 0:
        UMAP(base_url,
         target_dimension=2,
         neighbours=13,
         epochs=100,
         min_dist=0.01,
         alpha0=1,
         points=48,
         resolution=50,
         bound=4)
    elif mode == 1:
        TestNN(min=5,max=11,target_dim=3)
    #TestEpochs()

#GAUSSIAN MIXTURE MODEL
#COLOUR BLIND FRIENDLY COLOUR SCHEME
#EXPLORE 2D SIMPLICES