from __future__ import annotations
import scipy, scipy.signal
import itertools
from .backend import get_backend
from . import utils
def tgc(S):
    """
    %TGC Time-gain compensation for RF or IQ signals
    %   TGC(RF) or TGC(IQ) performs a time-gain compensation of the RF or IQ
    %   signals using a decreasing exponential law. Each column of the RF/IQ
    %   array must correspond to a single RF/IQ signal over (fast-) time.
    %
    %   [~,C] = TGC(RF) or [~,C] = TGC(IQ) also returns the coefficients used
    %   for time-gain compensation (i.e. new_SIGNAL = C.*old_SIGNAL)
    %
    %
    %   This function is part of MUST (Matlab UltraSound Toolbox).
    %   MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later
    %
    %   See also RF2IQ, DAS.
    %
    %   -- Damien Garcia -- 2012/10, last update 2020/05
    %   website: <a
    %   href="matlab:web('https://www.biomecardio.com')">www.BiomeCardio.com</a>
    """

    backend = get_backend()
    siz0 = S.shape

    if not utils.iscomplex(S):  # we have RF signals
        C = backend.mean(backend.abs(scipy.signal.hilbert(S, axis = 0)),1)
        # C = median(abs(hilbert(S)),2);
    else:  # we have IQ signals
        C = backend.mean(backend.abs(S),1)
        # C = median(abs(S),2);
    n = len(C)
    n1 = int(backend.ceil(n/10))
    n2 = int(backend.floor(n*9/10))
    """
    % -- Robust linear fitting of log(C)
    % The intensity is assumed to decrease exponentially as distance increases.
    % A robust linear fitting is performed on log(C) to seek the TGC
    % exponential law.
    % --
    % See RLINFIT for details
    """
    N = 200  # a maximum of N points is used for the fitting
    p = min(N/(n2-n1)*100,100)
    slope,intercept = rlinfit(backend.arange(n1,n2), backend.log(C[n1:n2]),p)

    C = backend.exp(intercept+slope*backend.arange(n).reshape((-1,1)))
    C = C[0]/C
    S = S*C

    S = backend.reshape(S, siz0)
    return S, C

def rlinfit(x,y,p):
    """
    %RLINFIT   Robust linear regression
    %   See the original RLINFIT function for details
    """
    backend = get_backend()
    N = len(x)
    I = backend.random_permutation(N)
    n = int(backend.round(N*p/100))
    I = I[:n]
    x = x[I]
    y = y[I]

    # Not sure it is the best option, what about some regression with regularisation?
    if True:
        C = backend.array([ (i,j) for i,j in  itertools.combinations(backend.arange(n), 2)])
    else:
        pass
    slope = backend.median( (y[C[:,1]]-y[C[:,0]])  / (x[C[:,1]]-x[C[:,0]]) )
    intercept = backend.median(y-slope*x)
    return slope, intercept
