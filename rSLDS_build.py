#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:51:00 2020

@author: sam
"""
# In[]
from ssm import SLDS
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
"""
load data
"""
train_all=0
t_range = [-5,10]
#Lag time embedding
tp = 6
data0 = np.load('otherData/VAE/012420_uv30s_hiRes_align_28.npy')
light = np.load('otherData/VAE/012420_uv30s_hiRes_align_28_light.npy')
data0 = data0/data0.max()

t = np.linspace(t_range[0],t_range[-1],120*(t_range[-1]-t_range[0]))
if tp==1:
    data = data0
else:
    data = []
    for i in tqdm(range(tp,data0.shape[0]),position=0):
        data.append(data0[i-tp:i,...])

data = np.array(data)
data=data/data.max()


#get light data
#vidName='../../../../media/sam/internalHDD/behavior/hiRes/012420_uv30s/'
#light_files=sorted(os.listdir(vidName+'light/'))
#light = []
#for f in light_files:
#    light.extend(np.load(vidName+'light/'+f))
#light = np.array(light)
#light=light>15.8
t_on=[0,]
for l in light[1:]:
    if l:
        t_on.append(t_on[-1]+1)
    else:
        t_on.append(0)
t_on = np.array(t_on)


raw = []
U = []
if train_all:
    U.append(light[:data.shape[0],None])
    raw.append(data)
else:
    ind = np.where(t_on==1)[0]
    for loc in ind:
        if loc<-t_range[0]*120 or loc+t_range[-1]*120>data.shape[0]:
            continue
        ind = np.arange(loc+t_range[0]*120,loc+t_range[-1]*120)
        raw.append(data[ind])
        U.append(light[ind][:,None])

latent = []
for dat in raw:
    latent.append(encoder.predict(dat))

# In[]
"""
Look at VAE latent
"""
from mpl_toolkits.mplot3d import Axes3D 

fig = plt.figure()
if latent[0].shape[-1]>2:
    ax = fig.add_subplot(111, projection='3d')

t_bin=np.linspace(-1,10,10)
uv=5

for i in range(1,t_bin.size):
    ind = np.where((t>=t_bin[i-1])&(t<=t_bin[i]))[0]
    if t_bin[i]<.5:
        c='r'
    else:
        c = plt.cm.viridis(i/t_bin.size)
    
    for dat in latent:
        if latent[0].shape[-1]>2:
            plt.plot(dat[ind,0],dat[ind,1],dat[ind,2],c=c,alpha=.3)
        else:
            plt.plot(dat[ind,0],dat[ind,1],c=c,alpha=.3)
        


# In[]
"""
build rSLDS
""" 
K=3
D=2
num_iters=10
dynamicsOnly=0
new = 1

if new:
#    slds = SLDS(latent[0].shape[-1], K, D, M=1,
#                     transitions="recurrent",
#                     dynamics="diagonal_gaussian",
#                     emissions="gaussian",
#                     single_subspace=True)
    slds = SLDS(latent[0].shape[-1], K, D, M=1,
                     transitions="recurrent",
                     dynamics="gaussian",
                     emissions="gaussian_id",
                     single_subspace=True)

slds.fit(latent, method="laplace_em", 
        variational_posterior="structured_meanfield",
        num_iters=num_iters, initialize=new,
        inputs=U,learn_emissions=(not dynamicsOnly))













# In[]
response=[]
discrete=[]
for u,dat in zip(U,latent):
    elbos, posteriors = slds.approximate_posterior(dat,num_iters=1,inputs=u)
    z = posteriors.mean_continuous_states[0]
    z_disc = slds.most_likely_states(z,dat)
    y_smooth = slds.smooth(z, dat)
    response.append(z)
    discrete.append(z_disc)
response=np.array(response)
discrete=np.array(discrete)


# In[]
"""
D2 QUIVER
"""

colors=['firebrick','cornflowerblue']
colors=['r','b','g']
def plot_most_likely_dynamics(model, uv=False,
    xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=20,
    alpha=0.8, ax=None, figsize=(3, 3)):
    
    K = model.K
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))
    print(xy.shape)
    # Get the probability of each state at each xy location
    if 'r' in model.transitions.__dict__:
        r = model.transitions.r
    else:
        r=0
    
    if uv:
       p = xy.dot(model.transitions.Rs.T) + r + model.transitions.Ws[:,0]
       z = np.argmax(xy.dot(model.transitions.Rs.T) + r + model.transitions.Ws[:,0], axis=1)
    else:
        p = xy.dot(model.transitions.Rs.T) + r
        z = np.argmax(xy.dot(model.transitions.Rs.T) + r, axis=1)
    p=np.exp(p)
#    print(p[0],p[-1])
    p=p/p.sum(axis=1)[:,None]
#    print(p[0],p[-1])
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    delta = np.zeros(xy.shape)
    if uv:
        ax.set_title('UV on')
        for k, (A, b, V) in enumerate(zip(model.dynamics.As, model.dynamics.bs, model.dynamics.Vs)):
            dxydt_m = xy.dot(A.T) + b + V[:,0] - xy
            delta += dxydt_m*p[:,k][:,None]
#            print(k,delta[-1])
            zk = z == k
            if zk.sum(0) > 0:
                ax.quiver(xy[zk, 0], xy[zk, 1],
                          dxydt_m[zk, 0], dxydt_m[zk, 1],
                          color=colors[k % len(colors)], alpha=alpha)
    else:
        ax.set_title('UV off')
        for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
            dxydt_m = xy.dot(A.T) + b - xy
            delta =delta + p[:,k][:,None]*dxydt_m
            print(k,delta[0])
#            print(p[:,k][:,None][0])
            zk = z == k
            if zk.sum(0) > 0:
                ax.quiver(xy[zk, 0], xy[zk, 1],
                          dxydt_m[zk, 0], dxydt_m[zk, 1],
                          color=colors[k % len(colors)], alpha=alpha)

    ax.set_xlabel('$Z_1$')
    ax.set_ylabel('$Z_2$')
    ax.quiver(xy[:, 0], xy[:, 1],
                          delta[:, 0], delta[:, 1],
                          color='k', alpha=1)
    plt.tight_layout()

    return ax

fig, ax = plt.subplots(ncols=3)
xlim=(np.mean(response[:,:,0],axis=0).min()-2,np.mean(response[:,:,0],axis=0).max()+3)
ylim=(np.mean(response[:,:,1],axis=0).min()-2,np.mean(response[:,:,1],axis=0).max()+3)
ax[0].plot(t,np.median(response[:,:,0],axis=0), label='$ \\langle Z1_t \\rangle$')
ax[0].plot(t,np.median(response[:,:,1],axis=0), label='$ \\langle Z2_t \\rangle$')
#ax[0].plot(t,np.mean(discrete,axis=0),c='k', label='$P(S=1|t)$')
plot_most_likely_dynamics(slds,ax=ax[1],
                          xlim=xlim,ylim=ylim)
#                          xlim=(response[:,:,0].min(),response[:,:,0].max()),
#                          ylim=(response[:,:,1].min(),response[:,:,1].max()))

#ax[2].imshow(Activity.classifier.dynamics._As[0],cmap='RdBu',clim=(-.1,.1))
plot_most_likely_dynamics(slds,ax=ax[2],uv=True,
                          xlim=xlim,ylim=ylim)
ax[0].legend()
print(slds.dynamics._As)

# In[]
"""
D2 UV RESPONSE
"""
fig, ax = plt.subplots(ncols=2)
plot_most_likely_dynamics(slds,
                          xlim=xlim,ylim=ylim,uv=1,ax=ax[0])
plot_most_likely_dynamics(slds,
                          xlim=xlim,ylim=ylim,uv=0,ax=ax[1])
t_bin=np.linspace(-.01,10,20)
prev=np.where(t>t_bin[0])[0][0]
for i in range(1,t_bin.size):
    this=np.where(t>t_bin[i])[0][0]
    if t_bin[i-1]<uv:
        c='r'
        ax[0].plot(np.median(response[:,prev:this,0],axis=0),np.median(response[:,prev:this,1],axis=0),c=c)
    else:
        c=plt.cm.viridis_r(i/t_bin.size)
        ax[1].plot(np.median(response[:,prev:this,0],axis=0),np.median(response[:,prev:this,1],axis=0),c=c)
    prev=this.copy()-1



# In[]
"""
simulate worm
""" 
n=9 
uv = 30/60
import matplotlib.animation as animation
t2 = np.arange(-120,15*120)/120
u = np.zeros((t2.size,1))
u[(t2>=0)&(t2<=uv)] = 1
sim = []
sim_latent = []
for i in tqdm(range(n)):
    states_z, states_x, emissions = slds.sample(u.size,input=u)
    imgs = decoder.predict(emissions) 
    sim.append(imgs)
    sim_latent.append(emissions)
plt.plot(t2,np.median(np.array(sim_latent),axis=0)) 
#plt.fill_between(t2,np.percentile(np.array(sim_latent),25,axis=0),np.percentile(np.array(sim_latent),75,axis=0),alpha=.3) 
# In[] 
s=3
saveTo = 'PLOTS/vae/DNE/'
st = np.where(u==1)[0][0]-60 
en = int(st+5*120)  
l1=-4
l2=4
sub=3

 
if saveTo == None:
    fig = plt.figure(constrained_layout=True,figsize=(12,4))    
    gs = GridSpec(s, 2*s, figure=fig)
    ax = []
    for i in range(s):
        ax.append([])
        for j in range(s):
            a = fig.add_subplot(gs[i,j])
            ax[-1].append(a)
    if sim_latent[0].shape[-1]>2:
        ax_tr = fig.add_subplot(gs[:,s:], projection='3d')
    else:
        ax_tr = fig.add_subplot(gs[:,s:],)
    ims = []
    for i in range(st,en):
        if i %sub ==0: continue
        art = []
        for j in range(n):
            if j>=s**2: break
            if len(imgs[0].shape)==3:
                pic = sim[j][i,:,:,0]
            else:
                pic = sim[j][i,-1,:,:,0]
#                pic = np.median(sim[j][i,:,:,:,0],axis=0)
            art.append(ax[j//s][j%s].imshow(pic,cmap="Greys_r",clim=(0,1)))
            art.append(ax[j//s][j%s].imshow(np.ones(pic.shape),cmap='plasma',alpha = u[i]*.3))
            art.append(ax[j//s][j%s].imshow(np.ones(pic.shape)*j,cmap='gist_rainbow',clim=(0,n),alpha = .1))
            lat = sim_latent[j][i-30:i]
            if sim_latent[0].shape[-1]>2:
                art.append(ax_tr.plot(lat[:,0],lat[:,1],lat[:,2],alpha=.3,c='grey')[0])
                art.append(ax_tr.scatter(lat[-1,0],lat[-1,1],lat[-1,2],alpha=.6,color=plt.cm.gist_rainbow(j/n)))
            else:
                art.append(ax_tr.plot(lat[:,0],lat[:,1],alpha=.3,c='grey')[0])
                art.append(ax_tr.scatter(lat[-1,0],lat[-1,1],alpha=.6,color=plt.cm.gist_rainbow(j/n)))
        
        ims.append(art)
        
    ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True,
                                    repeat_delay=100)
    plt.show()   

else:
    plt.ioff()
    fig = plt.figure(constrained_layout=True,figsize=(12,4))    
    gs = GridSpec(s, 2*s, figure=fig)
    ax = []
    for i in range(s):
        ax.append([])
        for j in range(s):
            a = fig.add_subplot(gs[i,j])
            ax[-1].append(a)
    if sim_latent[0].shape[-1]>2:
        ax_tr = fig.add_subplot(gs[:,s:], projection='3d')
    else:
        ax_tr = fig.add_subplot(gs[:,s:],)
#    ax_tr.axes.set_xlim3d(l1,l2)
#    ax_tr.axes.set_ylim3d(l1,l2)
#    ax_tr.axes.set_zlim3d(l1,l2)
    for i in tqdm(range(st,en),position=0,leave=True):
        ax_tr.clear()
        for j in range(n):
            if j>=s**2: break
            if len(imgs[0].shape)==3:
                pic = sim[j][i,:,:,0]
            else:
                pic = np.median(sim[j][i,:,:,:,0],axis=0)
            ax[j//s][j%s].clear()
            ax[j//s][j%s].imshow(pic,cmap="Greys_r",clim=(0,1))
            ax[j//s][j%s].imshow(np.ones(pic.shape),cmap='plasma',alpha = u[i]*.3)
            ax[j//s][j%s].imshow(np.ones(pic.shape)*j,cmap='gist_rainbow',clim=(0,n),alpha = .01)
            lat = sim_latent[j][i-30:i]
            if sim_latent[0].shape[-1]>2:
                ax_tr.plot(lat[:,0],lat[:,1],lat[:,2],alpha=.3,c='grey')
                ax_tr.scatter(lat[-1,0],lat[-1,1],lat[-1,2],alpha=.6,color=plt.cm.gist_rainbow(j/n))
            else:
                ax_tr.plot(lat[:,0],lat[:,1],alpha=.3,c='grey')
                ax_tr.scatter(lat[-1,0],lat[-1,1],alpha=.6,color=plt.cm.gist_rainbow(j/n))
        if sim_latent[0].shape[-1]>2:
            ax_tr.axes.set_xlim3d(l1,l2)
            ax_tr.axes.set_ylim3d(l1,l2)
            ax_tr.axes.set_zlim3d(l1,l2)
        else:
            ax_tr.set_xlim(l1,l2)
            ax_tr.set_ylim(l1,l2)
        fig.suptitle(str((i-np.where(u==1)[0][0])/2)+' s')
        fig.savefig(saveTo+f'{i:05}'+'.jpg')
#        plt.close('all')
#        del fig
#        del ax
#        del ax_tr
    plt.ion()
    plt.close('all')        
            
        


# In[]
fig = plt.figure()
if latent[0].shape[-1]>2:
    ax = fig.add_subplot(111, projection='3d')
    
for em in sim_latent:
    t_bin=np.linspace(-1,5,10)


for i in range(1,t_bin.size):
    ind = np.where((t2>=t_bin[i-1])&(t2<=t_bin[i]))[0]
    if t_bin[i]<uv:
        c='r'
    else:
        c = plt.cm.viridis(i/t_bin.size)
    
    for dat in sim_latent:
        if latent[0].shape[-1]>2:
            plt.plot(dat[ind,0],dat[ind,1],dat[ind,2],c=c,alpha=.3)
        else:
            plt.plot(dat[ind,0],dat[ind,1],c=c,alpha=.3)
    

