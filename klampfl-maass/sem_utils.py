
import numpy
import pylab,matplotlib
import sys
import colorsys

eps = numpy.finfo('float64').eps

def select_random_element(sequence):
    return sequence[numpy.random.randint(len(sequence))]

def printf(s, o=sys.stdout):
    o.write(s)
    o.flush()

def entropy(p):
    #eps = numpy.finfo('float64').eps
    p = numpy.asarray(p)
    p /= numpy.sum(p)
    return -numpy.sum(p*numpy.log2(p+eps))

def kld(p,q):
    #eps = numpy.finfo('float64').eps
    p = numpy.asarray(p)
    return numpy.sum(p*numpy.log2(p/q))

def conditional_entropy(y, x):
    T,K = y.shape
    assert(x.shape==(T,))
    targets = numpy.unique(x)
    #print targets
    result = 0.0
    for t in targets:
        ids = numpy.where(x==t)[0]
        nt = len(ids)
        yp = numpy.sum(y[ids,:], axis=0)
        yp = yp/numpy.sum(yp)
        assert(yp.shape == (K,))
        #print t, nt, yp
        result += float(nt)/float(T)*entropy(yp)
    return result

def distance(pos1, pos2):
    pos1 = numpy.array(pos1)
    pos2 = numpy.array(pos2)
    numpy.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
    return numpy.linalg.norm(pos1-pos2)
    #return numpy.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)

def plot_spike_trains(st,ax,I=None,sub_neurons=1):
    st = numpy.asarray(st)
    if sub_neurons<1 and not type(sub_neurons)==int:
        sub_neurons=1
    st = st[:,::sub_neurons]
    T,N = st.shape
    pylab.axes(ax)
    pylab.hold(True)
    colors = ['r','g','b','m','c','y','k']
    maxI = numpy.max(I)
    for n in range(N):
        ts = numpy.where(st[:,n])[0]
        for t in ts:
            dy = 1.0/N
            c = 'k'
            if I is not None:
                if len(I)==T:
                    c = colors[I[t]%len(colors)]
                if len(I)==N:
                    c = matplotlib.cm.hsv(float(I[n])/float(maxI+1))
            pylab.axvline(t, n*dy, (n+sub_neurons)*dy, c=c)
    if N<=10:
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    pylab.xlim([0,T])
    pylab.ylim([0,N*sub_neurons])

def plot_spike_trains2(st,T,ax,I=None,hlines=None,vlines=None,sub_neurons=1,ms=1,offset=0,spkh=0.9,patterns=None,c='k',hsv=True):
    if sub_neurons<1 and not type(sub_neurons)==int:
        sub_neurons=1
    N = len(st)
    pylab.axes(ax)
    pylab.hold(True)
    colors = ['r','g','m','b','c','y','k']
    dy = 1.0/(N+1)
    maxI = numpy.max(I)
    for n in range(0,N,sub_neurons):
        ts = st[n]
        if I is not None:
            if len(I)==N:
                if hsv:
                    c = matplotlib.cm.hsv(float(I[n])/float(maxI+1))
                    c = reduce_hsv_value(c, 0.8)
                else:
                    c = colors[I[n]%len(colors)]
                if ms>0:
                    pylab.plot(ts,(n+offset)*numpy.ones(len(ts)), marker='.', ls='', ms=ms, c=c)
                else:
                    for t in ts:
                        pylab.axvline(t, (n+offset-sub_neurons*spkh/2)*dy, (n+offset+sub_neurons*spkh/2)*dy, c=c)
            elif len(I)==T:
                for t in ts:
                    #print t,n,I[t]
                    if hasattr(I[t],'__iter__'):
                        c = colors[I[t][n]%len(colors)]
                    else:
                        c = colors[I[t]%len(colors)]
                    if ms>0:
                        pylab.plot([t],[n+offset], marker='.', ls='', ms=ms, c=c)
                    else:
                        pylab.axvline(t, (n+offset-sub_neurons*spkh/2)*dy, (n+offset+sub_neurons*spkh/2)*dy, c=c)
        else:
            if ms>0:
                pylab.plot(ts,(n+offset)*numpy.ones(len(ts)), marker='.', ls='', ms=ms, c=c)
            else:
                for t in ts:
                    pylab.axvline(t, (n+offset-sub_neurons*spkh/2)*dy, (n+offset+sub_neurons*spkh/2)*dy, c=c)
        #for t in ts:
            #dy = 1.0/N
            #c = 'k'
            #if I is not None:
                #if len(I)==N:
                    #c = matplotlib.cm.hsv(float(I[n])/float(maxI+1))
                #elif len(I)==T:
                    #c = colors[I[t]%len(colors)]
            #pylab.axvline(t, n*dy, (n+1)*dy, c=c)
    if hlines is not None:
        for hl in hlines:
            pylab.axhline(hl, 0, 1, c='k', ls=':')
    if vlines is not None:
        for vl in vlines:
            pylab.axvline(vl, 0, 1, c='k', ls=':')
    if patterns is not None:
        p,t = patterns
        for pi,pt in enumerate(p):
            if (max(I[pt])>-1):
                c1 = colors[max(I[pt])%len(colors)]
                pylab.axvline(pt, 0, 1, c=c1, ls='--')
                if pt+t[pi]<T:
                    c2 = colors[max(I[pt+t[pi]-1])%len(colors)]
                    pylab.axvline(pt+t[pi], 0, 1, c=c2, ls='--')
                    if c1==c2:
                        line = matplotlib.lines.Line2D([pt,pt+t[pi]], [N+1,N+1], lw=3, color=c1)
                        line.set_clip_on(False)
                        pylab.gca().add_line(line)
                    else:
                        line1 = matplotlib.lines.Line2D([pt,pt+t[pi]/2], [N+1,N+1], lw=3, color=c1)
                        line2 = matplotlib.lines.Line2D([pt+t[pi]/2,pt+t[pi]], [N+1,N+1], lw=3, color=c2)
                        line1.set_clip_on(False)
                        line2.set_clip_on(False)
                        pylab.gca().add_line(line1)
                        pylab.gca().add_line(line2)
    if N<=10:
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    #pylab.xlim([0,T])
    #pylab.ylim([-0.5+offset,N+0.5+offset])

def generatePoissonSpikeTrain(T, f, n=-1, tau_refract=0.0):
    if n<0:
        n = int(numpy.ceil(5*f*T))
    if f==0:
        return numpy.array([])
    lambd = 1.0/f - tau_refract
    st=numpy.cumsum(tau_refract + numpy.random.exponential(lambd, (1,n)))
    st=st.compress((st<(T)).flat)
    return st

def convert_spike_train(st, T, dt):
    nsteps = (int)(T/dt)
    x = numpy.zeros(nsteps)
    for tp in range(nsteps):
        x[tp] = (int)(numpy.any((st>=tp*dt) & (st<(tp+1)*dt)))
    return x

def convolve_spike_train(st, T, dt, tau):
    x = convert_spike_train(st, T, dt)
    ts = numpy.arange(-0.1,0.1+dt/2,dt)
    g = numpy.exp(-(ts/tau)**2)
    return numpy.convolve(x,g,mode="same")

def spike_train_distance(st1, st2, T, dt, tau):
    g1 = convolve_spike_train(st1, T, dt, tau)
    g2 = convolve_spike_train(st2, T, dt, tau)
    d = numpy.linalg.norm(g1-g2)
    return d

def filter_spikes(Z, tfrom, tto, reset=False):
    Zout = []
    for i in range(len(Z)):
        if reset:
            Zout.append(numpy.asarray([t-tfrom for t in Z[i] if t>=tfrom and t<=tto]))
        else:
            Zout.append(numpy.asarray([t for t in Z[i] if t>=tfrom and t<=tto]))
    return Zout

def format_time(time):
    tmp = int(time)
    s = tmp%60
    s_str = "%ds" % (s)
    tmp /= 60
    m = tmp%60
    m_str = {True:"%dm "%(m),False:""}[m>0]
    tmp /= 60
    h = tmp
    h_str = {True:"%dh "%(h),False:""}[h>0]
    return h_str+m_str+s_str

def subsample_equal(r1, r2):
    from scipy import interpolate
    T1,N1 = r1.shape
    T2,N2 = r2.shape
    assert(N1==N2)
    if T1>=T2:
        ip = interpolate.interp1d(numpy.arange(T2),r2,axis=0)
        ri = ip(numpy.linspace(0,T2-1,T1))
        result = (r1,ri)
    else:
        ip = interpolate.interp1d(numpy.arange(T1),r1,axis=0)
        ri = ip(numpy.linspace(0,T1-1,T2))
        result = (ri,r2)
    assert(result[0].shape==result[1].shape)
    return result

def subsample_equal_list(rlist):
    result = []
    rimax = numpy.argmax([resp.shape[0] for resp in rlist])
    for i,resp in enumerate(rlist):
        if i!=rimax:
            resp1, resp2 = subsample_equal(rlist[i], rlist[rimax])
            result.append(resp1)
        else:
            result.append(rlist[rimax])
    return result

def spike_correlation(r, p, t, I, r_thresh=0):
    nsteps,N = r.shape
    pts = []
    for pti,ptt in enumerate(p[::1]):
        pt = ptt
        pi = numpy.max(I[pt])
        pl = t[pti]
        if pi<0:
            continue
        if pt+pl<=nsteps:
            pts.append(numpy.arange(pt,pt+pl).astype(int))
    if len(pts)<3:
        return (0.0, 0.0)
    resp1 = r[pts[0],:]
    resp2 = r[pts[1],:]
    resp3 = r[pts[2],:]

    resp1, resp2, resp3 = tuple(subsample_equal_list([resp1, resp2, resp3]))
    T = resp1.shape[0]
    assert(N == resp1.shape[1])
    assert(T,N == resp2.shape)
    cc12 = numpy.corrcoef(resp1.T, resp2.T)
    assert(cc12.shape == 2*N,2*N)
    ccs12 = [cc12[i,N+i] for i in range(N) if numpy.max(resp1[:,i])>r_thresh and numpy.max(resp2[:,i])>r_thresh]

    assert(T,N == resp3.shape)
    cc13 = numpy.corrcoef(resp1.T, resp3.T)
    assert(cc13.shape == 2*N,2*N)
    ccs13 = [cc13[i,N+i] for i in range(N) if numpy.max(resp1[:,i])>r_thresh and numpy.max(resp3[:,i])>r_thresh]

    return numpy.mean(ccs12), numpy.mean(ccs13)

def reduce_hsv_value(c, v):
    hsv = list(colorsys.rgb_to_hsv(*c[:3]))
    hsv[2] = v
    return colorsys.hsv_to_rgb(*hsv)

def getMeanCochleagram(spec1,spec2):
    N1, T1 = spec1.shape
    N2, T2 = spec2.shape
    spec1_new, spec2_new = subsample_equal(spec1.T, spec2.T)
    assert(spec1_new.shape == spec2_new.shape)
    spec_mean = numpy.mean(numpy.array([spec1_new,spec2_new]), axis=0)
    return spec_mean.T

def plotCochleagram(spec):
    pylab.imshow(spec[::-1,:], aspect='auto')

if __name__ == '__main__':
    #T = 0.5
    #f = 20
    #dt = 1e-3
    #nsteps = (int)(T/dt)
    #g_tau = 5e-3
    #st = generatePoissonSpikeTrain(T, f)
    #ts = numpy.arange(0,T,dt)
    #g = convolve_spike_train(st, T, dt, g_tau)


    #st = [generatePoissonSpikeTrain(T, f) for i in range(50)]
    #pylab.figure()
    #plot_spike_trains2(st,T,pylab.gca())
    #pylab.show()

    #pylab.figure()
    #pylab.hold(True)
    #pylab.stem(st,numpy.ones(st.shape))
    #pylab.plot(ts,g)
    #pylab.show()

    x = numpy.arange(-1,1.05,0.1)
    y1 = numpy.exp(x)
    y2 = numpy.exp(-x)
    y = numpy.array([y1,y2]).T
    x2 = numpy.linspace(-1,1,7)
    z1 = numpy.sin(x2)
    z2 = numpy.cos(x2)
    z = numpy.array([z1,z2]).T
    yr, zr = subsample_equal(y, z)

    pylab.figure()
    pylab.plot(x,y,'x')
    pylab.plot(x2,z,'o')
    pylab.figure()
    pylab.plot(x,yr,'x')
    pylab.plot(x,zr,'o')
    pylab.show()

    