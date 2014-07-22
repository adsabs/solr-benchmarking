import matplotlib
from matplotlib import pyplot as plt

import os,sys
import numpy
from matplotlib.font_manager import FontProperties

from scipy.fftpack import rfft, irfft, fftfreq, fft
from scipy.interpolate import interp1d

def loadData(f):
  def _makeFloat(i):
    try:
      return float(i)
    except ValueError:
      return i

  with open(f) as fp:
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


def interpolate(data,m=0.95):
  hist, bin_edges = numpy.histogram(data, bins=100)
  csum = numpy.cumsum(hist)
  csum = csum / (float(csum.max()))
  func=interp1d(csum,bin_edges[:-1])
  res = func(m)
  return res

def plot_aggregated(datadirs,plotx='req_per_sec',ploty='q95'):
  data = {}
  px, py = {},{}
  for dd in datadirs:
    data[dd] = {}
    for d in os.listdir(dd):
      for f in os.listdir(os.path.join(dd,d)):
        if f.endswith('.dat'):
          data[dd][os.path.join(d,f)] = {}
          data[dd][os.path.join(d,f)]['raw']=loadData(os.path.join(dd,d,f))

    for experiment_id in data[dd]:
      results = data[dd][experiment_id]['raw']
      t=[i-results['t'][0] for i in results['t']]
      data[dd][experiment_id]['q95'] = interpolate(results['qtime'])
      data[dd][experiment_id]['dt95'] = interpolate(results['dt'])
      data[dd][experiment_id]['nworkers'] = len(set(results['worker_id']))
      data[dd][experiment_id]['req_per_sec'] = len(t)/t[-1]
      data[dd][experiment_id]['median_qtime'] = numpy.median(results['qtime'])
      data[dd][experiment_id]['mean_qtime'] = numpy.mean(results['qtime'])
    
    x,y = [],[]
    for experiment_id in data[dd]:
      results = data[dd][experiment_id]
      for k,v in results.iteritems():
        if k==plotx:
          x.append(v)
        if k==ploty:
          y.append(v)
    px[dd] = x
    py[dd] = y

  fig = plt.figure()
  ax = fig.gca()
  plt.rc('axes',color_cycle=['b','r','g','y','k'])
  for dd in px:
    ax.plot(px[dd],py[dd],'.',markeredgewidth=5,markersize=10,label=dd)
  fontP = FontProperties()
  fontP.set_size('small')
  
  box = ax.get_position()
  ax.set_position([box.x0, box.y0 + box.height*0,
                   box.width, box.height*1])
  ax.legend(loc="upper center",prop=fontP,numpoints=1,scatterpoints=1,bbox_to_anchor=(0.5, 1.10),ncol=len(px))
  plt.ylabel(ploty)
  plt.xlabel(plotx)
  plt.savefig('%s_vs_%s.png' % (ploty,plotx))



def main():
  data = loadData(sys.argv[1])
  t = [i-data['t'][0] for i in data['t']]

  plt.subplot(2,1,1)
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
  hist, bin_edges = numpy.histogram(data['qtime'], bins=100)
  csum = numpy.cumsum(hist)
  csum = csum / (float(csum.max()))
  m=0.95
  func=interp1d(csum,bin_edges[:-1])
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
  plt.text(res*1.5,(ylim[1]+ylim[0])*(1/2.0),text)
  plt.ylabel('Cumulative N')
  plt.xlabel('Qtime (ms)')

  plt.subplot(2,1,2)
  plt.plot(t,data['qtime'],'-k')
  plt.xlabel('t (s)')
  plt.ylabel('Qtime (ms)')
  #plt.yscale("log")
  plt.xlim([min(t),max(t)])

  # plt.subplot(3,1,3)
  # fig = plt.figure()
  # ax = fig.gca()
  # bins,values,bin_size = bin_data(t,data['qtime'],len(t))
  
  # ps = numpy.abs(numpy.fft.fft(values))**2
  # time_step = bin_size
  # freqs = numpy.fft.fftfreq(len(values), time_step)*1000
  # idx = numpy.argsort(freqs)
  # ax.plot(freqs[idx], ps[idx],'-k')
  # #plt.yscale("log")
  # ax.set_xlabel('Hz')
  # ax.setylabel('Power')
 
  plt.savefig('results.png')
  #plt.show()

if __name__ == '__main__':
  main()