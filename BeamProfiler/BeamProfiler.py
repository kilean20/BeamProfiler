import numpy as np
import scipy.optimize as opt
from copy import deepcopy as copy
from scipy.special import roots_legendre

pi = np.pi
intvec = np.vectorize(int)
  

def integrate(func,a,b,n_sub_interval=8,n_order=32):
  '''
  1D function Gaussian qaudrature integration from a to b
  '''
  L = b-a
  dL = L/n_sub_interval
  X0,W0 = roots_legendre(n_order)
  X = dL*(X0+1.)/2. 
  y = 0.
  for i in range(n_sub_interval):
    x_min = a + i*dL
    y = y + np.sum(W0*func(X+x_min), axis=-1)
  return y*dL/2.0


def getDFT(Z,K):
  '''
  [sum(Z*np.exp(-1j*k*np.arange(len(Z)))) for k in K]
  '''
  n=len(Z)
  if isinstance(K,np.float):
    return np.sum(Z*np.exp(-1j*K*np.arange(0,n)))
  else:
    sample = len(K)
    out = np.zeros(sample,dtype=np.complex)
    for i,k in enumerate(K):
        out[i] = np.sum(Z*np.exp(-1j*k*np.arange(0,n)))
    return out 





def getEmittance(X):
  '''
  input:
    X: 2-dim array representing (x,p) data
  '''
  sigx = np.std(X[:,0])
  sigp = np.std(X[:,1])
  sigxp = np.sum((X[:,0]-X[:,0].mean())*(X[:,1]-X[:,1].mean()))/len(X)
  return np.sqrt(sigx*sigx*sigp*sigp - sigxp*sigxp)






def naff(nmode,signal,window_id=1):
  """
  tunes,amps,substracted_signals = naff(nmode,signal)
  t=[0,T-1]
  amp = signal*np.exp(-2j*pi*tune*np.arange(T))
  """
  T = len(signal)
  window = (1.0+np.cos(np.pi*(-1.0+2.0/(T+1.0)*np.arange(1,T+1))))**window_id
  window = window/np.sum(window)


  def getPeakInfo(signal):
    T = len(signal)
    def loss(tune):
      return -np.abs(np.sum(signal*window*np.exp(-2j*pi*tune*np.arange(T))))
    tune = np.argmax(np.abs(np.fft.fft(signal)))/T
    result = opt.differential_evolution(loss,((tune-2.2/T,tune+2.2/T),),popsize=9)
    return result

  tunes = []
  amps = []
  subtracted_signals = []

  X = copy(signal)
  for i in range(nmode):
    result = getPeakInfo(X)
    if result.message!='Optimization terminated successfully.':
      print('Optimization failed at '+str(i+1)+'-th mode')
      break
    tunes.append(copy(result.x))
    amps.append(np.sum(X*np.exp(-2j*pi*tunes[-1]*np.arange(T)))/T)

    X = X - amps[-1]*np.exp(2j*pi*tunes[-1]*np.arange(T))
    subtracted_signals.append(copy(X))

  return tunes,amps,subtracted_signals




def getProfile_of_normal_coordiante(Z1,Z2, n, nsub, norder, lr, old_sigma=None):
  '''
  meaasure beam profile in normalized coordinate of initial kick direction. 
  Inputs:
    Z1 : 1D complex numpy array represeting <x(t)-ip(t)>. Z1(0) = x_0
    Z2 : 1D complex numpy array represeting x(t) - ip(t). Z2(0) = x_0 + dx
  '''
  nturn = len(Z1)
  
  x0 = z1[0].real
  bar_x0 = z2[0].real
  
  I0 = 0.5*x0*x0
  bar_I0 = 0.5*bar_x0*bar_x0
  dI = bar_I0-I0
  
  nu0_naff = nu0_naff[0]
  mu0_naff = 2*np.pi*nu0_naff
  
  bar_nu0_naff,_,_ = naff(1,Z2)
  bar_nu0_naff = bar_nu0_naff[0]
  bar_mu0_naff = 2*np.pi*bar_nu0_naff
  
  mu1_naff = (bar_mu0_naff - mu0_naff)/(bar_I0-I0)
  
  
  # initial sigma estimation
  if isinstance(old_sigma,float):
    new_sigma = old_sigma
  else:
    from scipy.signal import savgol_filter
    window_length = int(nturn/50)
    if window_length%2 == 0:
      window_length = window_length +1
    window_length = max([11,window_length])
    polyorder = 3
    smothedEnv1 = savgol_filter(np.abs(centroid1),window_length,polyorder) 
    for t in range(nturn):
        if smothedEnv1[t] < 0.5*smothedEnv1[t]:
            break
    new_sigma = np.abs(1.1774/(x0*t*mu1_naff))
   
  
  def iterate(new_mu0, new_bar_mu0, new_mu1, new_sigma, nsub=8, norder=32, lr=0.9, epochs=30):
    
    lr0 = lr
    Loss = {'err_mu0':[],
            'err_bar_mu0':[],
            'mu1':[],
            'sigma':[]}
    Loss['mu1'].append(new_mu1)
    Loss['sigma'].append(new_sigma)
    
    for i in range(epochs):
        cr = 5 + 10*(epochs-i-1)/(epochs-1)
      
        def integrand(k):
            return k*(getDFT(centroid1,k+new_mu0).real/x0-0.5)

        tmp = integrate(integrand, -cr*new_sigma*new_mu1*x0, +cr*new_sigma*new_mu1*x0, n_sub_interval=nsub,n_order=norder)
        new_mu0 = new_mu0 + lr*tmp/np.pi*np.sign(new_mu1*x0)
        Loss['err_mu0'].append(tmp/np.pi*np.sign(new_mu1*x0))
        

        def integrand(k):
            return k*(getDFT(centroid2,k+new_bar_mu0).real/x0-0.5)

        tmp = integrate(integrand, -cr*new_sigma*new_mu1*bar_x0, +cr*new_sigma*new_mu1*bar_x0,n_sub_interval=nsub,n_order=norder)
        new_bar_mu0 = new_bar_mu0 + lr*tmp/np.pi*np.sign(new_mu1*bar_x0)
        Loss['err_bar_mu0'].append(tmp/np.pi*np.sign(new_mu1*bar_x0))

        
        new_mu1 = (new_bar_mu0 - new_mu0)/(bar_I0-I0)
        Loss['mu1'].append(new_mu1)
        
        
        intk2F1 = integrate(integrand,-cr*new_sigma*new_mu1*x0,cr*new_sigma*new_mu1*x0)
        new_sigma = np.sqrt(np.abs(intk2F1)/np.pi)/(new_mu1*x0)
        Loss['sigma'].append(new_sigma)
        
        if i>epochs/2:
             lr = lr0*(i-epochs/2)/((epochs+1)/2)    
            
    return new_mu0, new_bar_mu0, new_mu1, new_sigma, Loss
  
  
  new_mu0 = mu0_naff
  new_bar_mu0 = bar_mu0_naff
  new_mu1 = mu1_naff

  new_mu0, new_bar_mu0, new_mu1, new_sigma, Loss = iterate(new_mu0, new_bar_mu0, new_mu1, n=30, nsub=8, norder=32,lr=0.9)
  
  K1 = new_mu0 + new_sigma*new_mu1*x0*np.linspace(-5,5,2048)
  DFT1 = getDFT(centroid1/x0,K1).real-0.5
  
  return (K1-new_mu0)/(new_mu1*x0), DFT1.real*np.abs(new_mu1*x0)/np.pi, new_mu0, new_bar_mu0, new_mu1, new_sigma, Loss