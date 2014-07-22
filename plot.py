import matplotlib
from matplotlib import pyplot as plt
import os,sys
import numpy
from scipy.fftpack import rfft, irfft, fftfreq, fft
from scipy.interpolate import interp1d

def loadData():
  def _makeFloat(i):
    try:
      return float(i)
    except ValueError:
      return i

  with open(sys.argv[1]) as fp:
    lines = [L for L in fp.readlines() if L and not L.startswith('#')]
  res=map(lambda L: map(lambda i: _makeFloat(i) ,L.strip().split()), lines)

  D = {}
  D['t'] = [r[0] for r in res] 
  D['dt'] = [r[1] for r in res]
  D['qtime'] = [r[2] for r in res]
  D['nfound'] = [r[3] for r in res]
  D['worker_id'] = [r[4] for r in res]
  return D

def bin_data(x,y,nbins):
  bin_size = ( (max(x)-min(x)) /nbins )
  bins = [min(x)]
  values = []

  while bins[-1] < max(x):
    bins.append(bins[-1]+bin_size)
    t = [j for i,j in zip(x,y) if i<=bins[-1] and i>bins[-2]]
    v = numpy.sum(t)
    values.append(v)
  return bins,values,bin_size

def main():
  data = loadData()
  t = [i-data['t'][0] for i in data['t']]
  print t[-10:-1],max(t)

  plt.subplot(2,1,1)
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
  hist, bin_edges = numpy.histogram(data['qtime'], bins=100)
  csum = numpy.cumsum(hist)
  csum = csum / (float(csum.max()))
  func=interp1d(csum,bin_edges[:-1])
  m=0.95
  res = func(m)
  plt.plot(bin_edges[:-1], csum,'-k')
  plt.axhline(y=1,xmin=bin_edges[0],xmax=bin_edges[-1],ls='--',color='red')
  ylim = [min(csum)-0.05,max(csum)+.05]
  plt.plot([res*0.99,res*1.01],[ylim[0],m],'--r')
  plt.ylim(ylim)
  workers = len(set(data['worker_id']))
  req_sec = len(t)/t[-1]
  text = '%s%% Qtime=%0.1f \nmedian=%0.2f \nmean=%0.2f \nN=%s (%s workers) \nReq/sec=%0.1f' % \
    (m*100,res,numpy.median(data['qtime']),numpy.mean(data['qtime']),len(data['qtime']),workers,req_sec)
  plt.text(res*1.5,(ylim[1]+ylim[0])/2.3,text)
  plt.ylabel('Cumulative N')
  plt.xlabel('Qtime (ms)')

  plt.subplot(2,1,2)
  plt.plot(t,data['qtime'],'-k')
  plt.xlabel('t (s)')
  plt.ylabel('Qtime (ms)')
  #plt.yscale("log")
  plt.xlim([min(t),max(t)])

  # plt.subplot(3,1,3)
  # bins,values,bin_size = bin_data(t,data['qtime'],len(t))
  
  # ps = numpy.abs(numpy.fft.fft(values))**2
  # time_step = bin_size
  # freqs = numpy.fft.fftfreq(len(values), time_step)*1000
  # idx = numpy.argsort(freqs)
  # plt.plot(freqs[idx], ps[idx],'-k')
  # #plt.yscale("log")
  # plt.xlabel('Hz')
  # plt.ylabel('Power')
 
  plt.savefig('results.png')
  plt.show()

if __name__ == '__main__':
  main()