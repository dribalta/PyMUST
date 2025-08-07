import scipy, scipy.interpolate, multiprocessing, multiprocessing.pool
from abc import ABC
import inspect, matplotlib, pickle, os, matplotlib.pyplot as plt, copy
from collections import deque
from .backend import get_backend


class dotdict(dict, ABC):
    """Copied from https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary"""
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def ignoreCaseInFieldNames(self):
        """Convert all field names to lower case"""
        names = self.names
        todelete =[]
        for k, v in self.items():
            if k.lower() in names and k in names:
                if k.lower() == k:
                    continue
                elif names[k] in self:
                    raise ValueError(f'Repeated key {k}')
                else:
                    self[names[k]] = v
                    todelete.append(k)
        for k in todelete:
            del self[k]
        return self
    def copy(self):
        return copy.deepcopy(self)
    def __getstate__(self):
        d = {k : v for k,v in self.items()}
        return d
    def __setstate__(self, d):
        for k, v in self.items():
            self[k] = v
        
class Options(dotdict):
    default_Number_Workers = multiprocessing.cpu_count()
    @property 
    def names(self):
        names = {'dBThresh','ElementSplitting',
                'FullFrequencyDirectivity','FrequencyStep','ParPool',
                'WaitBar'}
        return {n.lower(): n for n in names}
    
    def setParPool(self, workers, mode = 'process'):
        if mode not in ['process', 'thread']:
            raise ValueError('ParPoolMode must be either "process" or "thread"')
        self.ParPool_NumWorkers = workers
        self.ParPoolMode = mode
    
    def getParallelPool(self):
        workers = self.get('ParPool_NumWorkers', self.default_Number_Workers)
        mode = self.get('ParPoolMode', 'thread')
        if mode == 'process':
            pool = multiprocessing.Pool(workers)
        elif mode == 'thread':
            pool = multiprocessing.pool.ThreadPool(workers)
        else:
             raise ValueError('ParPoolMode must be either "process" or "thread"')
        return pool
    
    def getParallelSplitIndices(self, N,n_threads = None):
        if hasattr(N, '__len__'):
            N = len(N)
        assert isinstance(N, int), 'N must be an integer'

        n_threads = self.get('ParPool_NumWorkers', self.default_Number_Workers) if n_threads is None else n_threads
        #Create indices for parallel processing, split in workers
        backend = get_backend()
        idx = backend.arange(0, N, N//n_threads)

        #Repeat along new axis
        idx = backend.stack([idx, backend.roll(idx, -1)], axis = 1)
        idx[-1, 1] = N
        return idx

class Param(dotdict):
    @property 
    def names(self):
        names = {'attenuation','baffle','bandwidth','c','fc',
            'fnumber','focus','fs','height','kerf','movie','Nelements',
            'passive','pitch','radius','RXangle','RXdelay'
            'TXapodization','TXfreqsweep','TXnow','t0','width'}
        return {n.lower(): n for n in names}
    
    def getElementPositions(self):
        """
        Returns the position of each piezoelectrical element in the probe.
        """
        RadiusOfCurvature = self.radius
        NumberOfElements = self.Nelements

        backend = get_backend()
        if backend.isinf(RadiusOfCurvature):
            #% Linear array
            xe =  (backend.arange(NumberOfElements)-(NumberOfElements-1)/2)*self.pitch
            ze = backend.zeros((1,NumberOfElements))
            THe = backend.zeros_like(ze)
            h = backend.zeros_like(ze)
        else:
            #% Convex array
            chord = 2*RadiusOfCurvature*backend.sin(backend.arcsin(self.pitch/2/RadiusOfCurvature)*(NumberOfElements-1))
            h = backend.sqrt(RadiusOfCurvature**2-chord**2/4); #% apothem
            #% https://en.wikipedia.org/wiki/Circular_segment
            #% THe = angle of the normal to element #e with respect to the z-axis
            THe = backend.linspace(backend.arctan2(-chord/2,h),backend.arctan2(chord/2,h),NumberOfElements)
            ze = RadiusOfCurvature*backend.cos(THe)
            xe = RadiusOfCurvature*backend.sin(THe)
            ze = ze-h
        return xe.reshape((1,-1)), ze.reshape((1,-1)), THe.reshape((1,-1)), h.reshape((1,-1))
    
    def getPulseSpectrumFunction(self, FreqSweep = None):
        if 'TXnow' not in self:
            self.TXnow = 1

        #-- FREQUENCY SPECTRUM of the transmitted pulse
        if FreqSweep is None:
            # We want a windowed sine of width PARAM.TXnow
            T = self.TXnow /self.fc
            backend = get_backend()
            wc = 2 * backend.pi * self.fc
            pulseSpectrum = lambda w = None: 1j * (mysinc(T * (w - wc) / 2) - mysinc(T * (w + wc) / 2))
        else:
            # We want a linear chirp of width PARAM.TXnow
            # (https://en.wikipedia.org/wiki/Chirp_spectrum#Linear_chirp)
            T = self.TXnow / self.fc
            backend = get_backend()
            wc = 2 * backend.pi * self.fc
            dw = 2 * backend.pi * FreqSweep
            s2 = lambda w = None: backend.multiply(backend.sqrt(backend.pi * T / dw) * backend.exp(- 1j * (w - wc) ** 2 * T / 2 / dw),(fresnelint((dw / 2 + w - wc) / backend.sqrt(backend.pi * dw / T)) + fresnelint((dw / 2 - w + wc) / backend.sqrt(backend.pi * dw / T))))
            pulseSpectrum = lambda w = None: (1j * s2(w) - 1j * s2(- w)) / T
        return pulseSpectrum

    def getProbeFunction(self):
        #%-- FREQUENCY RESPONSE of the ensemble PZT + probe
        #% We want a generalized normal window (6dB-bandwidth = PARAM.bandwidth)
        #% (https://en.wikipedia.org/wiki/Window_function#Generalized_normal_window)
        #-- FREQUENCY RESPONSE of the ensemble PZT + probe
        # We want a generalized normal window (6dB-bandwidth = PARAM.bandwidth)
        # (https://en.wikipedia.org/wiki/Window_function#Generalized_normal_window)
        backend = get_backend()
        wc = 2 * backend.pi * self.fc
        wB = self.bandwidth * wc / 100
        p = backend.log(126) / backend.log(2 * wc / wB)
        probeSpectrum_sqr = lambda w: backend.exp(- backend.power(backend.abs(w - wc) / (wB / 2 / backend.power(backend.log(2), 1 / p)), p))
        # The frequency response is a pulse-echo (transmit + receive) response. A
        # square root is thus required when calculating the pressure field:
        probeSpectrum = lambda w: backend.sqrt(probeSpectrum_sqr(w))
        return probeSpectrum
    
# To maintain same notation as matlab
def interp1(y, xNew, kind):
    if kind == 'spline':
        kind = 'cubic' #3rd order spline
    backend = get_backend()
    interpolator = scipy.interpolate.interp1d(backend.arange(len(y)), y, kind = kind) 
    return interpolator(xNew)    

def isnumeric(x):
    backend = get_backend()
    return backend.is_array(x) or isinstance(x, int) or isinstance(x, float) or hasattr(x, '__array__')

def iscomplex(x):
    backend = get_backend()
    if backend.is_array(x):
        return backend.iscomplexobj(x)
    return isinstance(x, complex)

def islogical(v):
    return isinstance(v, bool)

def isfield(d, k ):
    return k in d

def mysinc(x=None):
    """MATLAB-compatible sinc function."""
    backend = get_backend()
    return backend.sinc(x / backend.pi)  # [note: In MATLAB/numpy, sinc is sin(pi*x)/(pi*x)]


def shiftdim(array, n=None):
    """
    From stack overflow https://stackoverflow.com/questions/67584148/python-equivalent-of-matlab-shiftdim
    """
    if n is not None:
        if n >= 0:
            axes = tuple(range(len(array.shape)))
            new_axes = deque(axes)
            new_axes.rotate(n)
            backend = get_backend()
            return backend.moveaxis(array, axes, tuple(new_axes))
        backend = get_backend()
        return backend.expand_dims(array, axis=tuple(range(-n)))
    else:
        idx = 0
        for dim in array.shape:
            if dim == 1:
                idx += 1
            else:
                break
        axes = tuple(range(idx))
        # Note that this returns a tuple of 2 results
        backend = get_backend()
        return backend.squeeze(array, axis=axes), len(axes)

def isEmpty(x):
    backend = get_backend()
    return  x is None or (isinstance(x, list) and len(x) == 0) or (backend.is_array(x) and len(x) == 0)

def emptyArrayIfNone(x):
    if isEmpty(x):
        backend = get_backend()
        x = backend.array([])
    return x

def eps(s = 'single'):
    if s == 'single':
        return 1.1921e-07 
    else:
        raise ValueError()

def nextpow2(n):
    i = 1
    while (1 << i) < n:
        i += 1
    return i

def fresnelint(x): 
    # FRESNELINT Fresnel integral.
    
    # J = FRESNELINT(X) returns the Fresnel integral J = C + 1i*S.
    
    # We use the approximation introduced by Mielenz in
#       Klaus D. Mielenz, Computation of Fresnel Integrals. II
#       J. Res. Natl. Inst. Stand. Technol. 105, 589 (2000), pp 589-590
    
    backend = get_backend()
    siz0 = x.shape
    x = x.flatten()

    issmall = backend.abs(x) <= 1.6
    c = backend.zeros(x.shape)
    s = backend.zeros(x.shape)
    # When |x| < 1.6, a Taylor series is used (see Mielenz's paper)
    if backend.any(issmall):
        n = backend.arange(0,11)
        cn = backend.concatenate([[1], backend.cumprod(- backend.pi ** 2 * (4 * n + 1) / (4 * (2 * n + 1) *(2 * n + 2)*(4 * n + 5)))])
        sn = backend.concatenate([[1],backend.cumprod(- backend.pi ** 2 * (4 * n + 3) / (4 * (2 * n + 2)*(2 * n + 3)*(4 * n + 7)))]) * backend.pi / 6
        n = backend.concatenate([n,[11]]).reshape((1,-1))
        c[issmall] = backend.sum(cn.reshape((1,-1))*x[issmall].reshape((-1, 1))  ** (4 * n + 1), 1)
        s[issmall] = backend.sum(sn.reshape((1,-1))*x[issmall].reshape((-1, 1)) ** (4 * n + 3), 1)
    
    # When |x| > 1.6, we use the following:
    if not backend.all(issmall ):
        n = backend.arange(0,11+1)
        fn = backend.array([0.318309844,9.34626e-08,- 0.09676631,0.000606222,0.325539361,0.325206461,- 7.450551455,32.20380908,- 78.8035274,118.5343352,- 102.4339798,39.06207702])
        fn = fn.reshape((1, fn.shape[0]))
        gn = backend.array([0,0.101321519,- 4.07292e-05,- 0.152068115,- 0.046292605,1.622793598,- 5.199186089,7.477942354,- 0.695291507,- 15.10996796,22.28401942,- 10.89968491])
        gn = gn.reshape((1, gn.shape[0]))

        fx = backend.sum(backend.multiply(fn,x[not issmall ] ** (- 2 * n - 1)), 1)
        gx = backend.sum(backend.multiply(gn,x[not issmall ] ** (- 2 * n - 1)), 1)
        c[not issmall ] = 0.5 * backend.sign(x[not issmall ]) + backend.multiply(fx,backend.sin(backend.pi / 2 * x[not issmall ] ** 2)) - backend.multiply(gx,backend.cos(backend.pi / 2 * x[not issmall ] ** 2))
        s[not issmall ] = 0.5 * backend.sign(x[not issmall ]) - backend.multiply(fx,backend.cos(backend.pi / 2 * x[not issmall ] ** 2)) - backend.multiply(gx,backend.sin(backend.pi / 2 * x[not issmall ] ** 2))
    
    f = backend.reshape(c, siz0) + 1j * backend.reshape(s, siz0)
    return f


# Plotting
def polarplot(x, z, v, cmap = 'gray',background = 'black', probeUpward = True, **kwargs):
    plt.pcolormesh(x, z, v, cmap = cmap, shading='gouraud', **kwargs)
    plt.axis('equal')
    ax = plt.gca()
    ax.set_facecolor(background)
    if probeUpward:
        ax.invert_yaxis()


def getDopplerColorMap():
    source_file_path = inspect.getfile(inspect.currentframe())
    with open( os.path.join(os.path.dirname(source_file_path), 'Data', 'colorMap.pkl'), 'rb') as f:
        dMap = pickle.load(f)
    new_cmap = matplotlib.colors.LinearSegmentedColormap('doppler', dMap)
    dopplerCM = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(),cmap=new_cmap)
    return dopplerCM

def applyDasMTX(M, IQ, imageShape):
    return (M @ IQ.flatten(order = 'F')).reshape(imageShape, order = 'F')
