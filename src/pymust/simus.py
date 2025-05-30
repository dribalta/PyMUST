from . import utils, pfield, getpulse
import logging, copy, multiprocessing, functools
import numpy as np 

# pfield wrapper so it is compatible with multiprocessing. Needs to be defined in a global scope
def pfieldParallel(x: np.ndarray, y: np.ndarray, z: np.ndarray, RC: np.ndarray, delaysTX: np.ndarray, param: utils.Param, options: utils.Options):
    options = options.copy()
    options.ParPool = False # No parallel within the parallel
    options.RC = RC
    _, RFsp, idx =  pfield(x, y, z, delaysTX, param, options)
    return RFsp, idx


def simus(*varargin):
    """
    %SIMUS   Simulation of ultrasound RF signals for a linear or convex array
    %   RF = SIMUS(X,Y,Z,RC,DELAYS,PARAM) simulates ultrasound RF radio-
    %   frequency signals generated by an ultrasound uniform LINEAR or CONVEX
    %   array insonifying a medium of scatterers.
    %   The scatterers are characterized by their coordinates (X,Y,Z) and
    %   reflection coefficients RC.
    %
    %   X, Y, Z and RC must be of same size. The elements of the ULA are
    %   excited at different time delays, given by the vector DELAYS. The
    %   transmission and reception characteristics must be given in the
    %   structure PARAM (see below for details).
    %
    %   RF = SIMUS(X,Z,RC,DELAYS,PARAM) or RF = SIMUS(X,[],Z,RC,DELAYS,PARAM)
    %   disregards elevation focusing (PARAM.focus is ignored) and assumes that
    %   Y=0 (2-D space). The computation is faster in 2-D.
    %
    %   >--- Try it: enter "simus" in the command window for an example ---< 
    %
    %   The RF output matrix contains Number_of_Elements columns. Each column
    %   therefore represents an RF signal. The number of rows depends on the
    %   depth (estimated from max(Z)) and the sampling frequency PARAM.fs (see
    %   below). By default, the sampling frequency is four times the center
    %   frequency.
    %
    %   Units: X,Y,Z must be in m; DELAYS must be in s; RC has no unit.
    %
    %   DELAYS can also be a matrix. This alternative can be used to simulate
    %   MLT (multi-line transmit) sequences. In this case, each ROW represents
    %   a series of delays. For example, to create a 4-MLT sequence with a
    %   64-element phased array, DELAYS matrix must have 4 rows and 64 columns
    %   (size = [4 64]).
    %
    %   SIMUS uses PFIELD during transmission and reception. The parameters
    %   that must be included in the structure PARAM are similar as those in
    %   PFIELD. Additional parameters are also required (see below).
    %
    %   ---
    %   NOTE #1: X-, Y-, and Z-axes
    %   Conventional axes are used:
    %   i)  For a LINEAR array, the X-axis is PARALLEL to the transducer and
    %       points from the first (leftmost) element to the last (rightmost)
    %       element (X = 0 at the CENTER of the transducer). The Z-axis is
    %       PERPENDICULAR to the transducer and points downward (Z = 0 at the
    %       level of the transducer, Z increases as depth increases). The
    %       Y-axis is such that the coordinates are right-handed.
    %   ii) For a CONVEX array, the X-axis is parallel to the chord and Z = 0
    %       at the level of the chord.
    %   ---
    %   NOTE #2: Simplified method: Directivity
    %   By default, the calculation is made faster by assuming that the
    %   directivity of the elements is dependent only on the central frequency.
    %   This simplification very little affects the pressure field in most
    %   situations (except near the array). To turn off this option, use
    %   OPTIONS.FullFrequencyDirectivity = true.
    %   (see ADVANCED OPTIONS below).
    %   ---
    %
    %   PARAM is a structure that contains the following fields:
    %   -------------------------------------------------------
    %       *** TRANSDUCER PROPERTIES ***
    %   1)  PARAM.fc: central frequency (in Hz, REQUIRED)
    %   2)  PARAM.pitch: pitch of the array (in m, REQUIRED)
    %   3)  PARAM.width: element width (in m, REQUIRED)
    %        or PARAM.kerf: kerf width (in m, REQUIRED)
    %        note: width = pitch-kerf
    %   4)  PARAM.focus: elevation focus (in m, ignored if Y is not given)
    %            The default is Inf (no elevation focusing)
    %   5)  PARAM.height: element height (in m, ignored if Y is not given)
    %            The default is Inf (no elevation focusing)
    %   6)  PARAM.radius: radius of curvature (in m)
    %            The default is Inf (rectilinear array)
    %   7)  PARAM.bandwidth: pulse-echo 6dB fractional bandwidth (in %)
    %            The default is 75%.
    %   8)  PARAM.baffle: property of the baffle:
    %            'soft' (default), 'rigid' or a scalar > 0.
    %            See "Note on BAFFLE properties" in "help pfield"
    %
    %       *** MEDIUM PARAMETERS ***
    %   9)  PARAM.c: longitudinal velocity (in m/s, default = 1540 m/s)
    %   10) PARAM.attenuation: attenuation coefficient (dB/cm/MHz, default: 0)
    %            Notes: A linear frequency-dependence is assumed.
    %                   A typical value for soft tissues is ~0.5 dB/cm/MHz.
    %
    %       *** TRANSMIT PARAMETERS ***
    %   11) PARAM.TXapodization: transmit apodization (default: no apodization)
    %   12) PARAM.TXnow: number of wavelengths of the TX pulse (default: 1)
    %   13) PARAM.TXfreqsweep: frequency sweep for a linear chirp (default: [])
    %                          To be used to simulate a linear TX chirp.
    %
    %       *** RECEIVE PARAMETERS *** (not in PFIELD)
    %   14) PARAM.fs: sampling frequency (in Hz, default = 4*param.fc)
    %   15) PARAM.RXdelay: reception law delays (in s, default = 0)
    %
    %   Other syntaxes:
    %   --------------
    %   i}  [RF,PARAM] = SIMUS(...) updates the fields of the PARAM structure.
    %   ii} [...] = SIMUS without any input argument provides an interactive
    %        example designed to produce RF signals from a focused ultrasound
    %        beam using a 2.7 MHz phased-array transducer (without elevation
    %        focusing).
    %
    %   PARALLEL COMPUTING:
    %   ------------------
    %   SIMUS calls the function PFIELD. If you have the Parallel Computing
    %   Toolbox, SIMUS can execute several PFIELDs in parallel. If this option
    %   is activated, a parallel pool is created on the default cluster. All
    %   workers in the pool are used. The X and Z are splitted into NW chunks,
    %   NW being the number of workers. To execute parallel computing, use: 
    %       [...] = SIMUS(...,OPTIONS),
    %   with OPTIONS.ParPool = true (default = false).
    %
    %
    %   OTHER OPTIONS:
    %   -------------
    %      %-- FREQUENCY STEP & FREQUENCY SAMPLES --%
    %   1a) Only frequency components of the transmitted signal in the range
    %       [0,2fc] with significant amplitude are considered. The default
    %       relative amplitude is -100 dB. You can change this value by using
    %       the following:
    %           [...] = SIMUS(...,OPTIONS),
    %       where OPTIONS.dBThresh is the threshold in dB (default = -100).
    %   1b) The frequency step is determined automatically to avoid aliasing in
    %       the time domain. This step can be adjusted with a scaling factor
    %       OPTIONS.FrequencyStep (default = 1). It is not recommended to
    %       modify this scaling factor in SIMUS.
    %   ---
    %      %-- FULL-FREQUENCY DIRECTIVITY --%   
    %   2)  By default, the directivity of the elements depends only on the
    %       center frequency. This makes the calculation faster. To make the
    %       directivities fully frequency-dependent, use: 
    %           [...] = SIMUS(...,OPTIONS),
    %       with OPTIONS.FullFrequencyDirectivity = true (default = false).
    %   ---
    %       %-- ELEMENT SPLITTING --%   
    %   3)  Each transducer element of the array is split into small segments.
    %       The length of these small segments must be small enough to ensure
    %       that the far-field model is accurate. By default, the elements are
    %       split into M segments, with M being defined by:
    %           M = ceil(element_width/smallest_wavelength);
    %       To modify the number M of subelements by splitting, you may adjust
    %       OPTIONS.ElementSplitting. For example, OPTIONS.ElementSplitting = 1
    %   ---
    %       %-- WAIT BAR --%   
    %   4)  If OPTIONS.WaitBar is true, a wait bar appears (only if the number
    %       of frequency samples >10). Default is true.
    %   ---
    %
    %
    %   Notes regarding the model & REFERENCES:
    %   --------------------------------------
    %   1) SIMUS calls the function PFIELD. It works for uniform linear or
    %      convex arrays. A uniform array has identical elements along a
    %      rectilinear or curved line in space with uniform spacing. Each
    %      element is split into small segments (if required). The radiation
    %      patterns in the x-z plane are derived by using a Fraunhofer
    %      (far-field) approximation. Those in the x-y elevational plane are
    %      derived by using a Fresnel (paraxial) approximation.
    %   2) The paper that describes the first 2-D version of SIMUS is:
    %      SHAHRIARI S, GARCIA D. Meshfree simulations of ultrasound vector
    %      flow imaging using smoothed particle hydrodynamics. Phys Med Biol,
    %      2018;63:205011. <a
    %      href="matlab:web('https://www.biomecardio.com/publis/physmedbio18.pdf')">PDF here</a>
    %   3) The paper that describes the theory of the full (2-D + 3-D) version
    %      of SIMUS will be submitted by February 2021.
    %
    %
    %   A simple EXAMPLE:
    %   ----------------
    %   %-- Generate RF signals using a phased-array transducer
    %   % Phased-array @ 2.7 MHz:
    %   param = getparam('P4-2v');
    %   % TX time delays:
    %   x0 = 0; z0 = 3e-2; % focus location 
    %   dels = txdelay(x0,z0,param);
    %   % Six scatterers:
    %   x = zeros(1,6); y = zeros(1,6);
    %   z = linspace(1,10,6)*1e-2;
    %   % Reflectivity coefficients:
    %   RC = ones(1,6);
    %   % RF signals:
    %   param.fs = 20*param.fc; % sampling frequency
    %   RF = simus(x,y,z,RC,dels,param);
    %   % Plot the RF signals
    %   plot(bsxfun(@plus,RF(:,1:7:64)/max(RF(:)),(1:10))',...
    %      (0:size(RF,1)-1)/param.fs*1e6,'k')
    %   set(gca,'XTick',1:10,'XTickLabel',int2str((1:7:64)'))
    %   title('RF signals')
    %   xlabel('Element number'), ylabel('time (\mus)')
    %   xlim([0 11]), axis ij 
    %   
    %
    %   This function is part of <a
    %   href="matlab:web('https://www.biomecardio.com/MUST')">MUST</a> (Matlab UltraSound Toolbox).
    %   MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later
    %
    %   See also PFIELD, TXDELAY, MKMOVIE, GETPARAM, GETPULSE.
    %
    %   -- Damien Garcia -- 2017/10, last update 2022/05/14
    %   website: <a
    %   href="matlab:web('https://www.biomecardio.com')">www.BiomeCardio.com</a>
    """

    returnTime = False #NoteGB: Set to True if you want to return the time, but quite a mess right now with the matlab style arguments

    nargin = len(varargin)
    if nargin<= 3 or nargin > 7:
        raise ValueError("Wrong number of input arguments.")
    #%-- Input variables: X,Y,Z,DELAYS,PARAM,OPTIONS
    x = varargin[0]

    if nargin ==5: # simus(X,Z,RC,DELAYS,PARAM)
            y = None
            z = varargin[1]
            RC = varargin[2]
            delaysTX = varargin[3]
            param = varargin[4]
            options = utils.Options()
    elif nargin == 6: # simus(X,Z,RC,DELAYS,PARAM,OPTIONS)
            if isinstance(varargin[4], utils.Param): #% simus(X,Z,RC,DELAYS,PARAM,OPTIONS)
                y = None
                z = varargin[1]
                RC = varargin[2]
                delaysTX = varargin[3]
                param = varargin[4]
                options = copy.deepcopy(varargin[5])
            else: # % simus(X,Y,Z,RC,DELAYS,PARAM)
                y = varargin[1]
                z = varargin[2]
                RC = varargin[3]
                delaysTX = varargin[4]
                param = varargin[5]
                options = utils.Options()
    else: # simus(X,Y,Z,RC,DELAYS,PARAM,OPTIONS)
                y = varargin[1]
                z = varargin[2]
                RC = varargin[3]
                delaysTX = varargin[4]
                param = varargin[5]
                options = copy.deepcopy(varargin[6])
    assert isinstance(param, utils.Param),'PARAM must be a structure.'

    #%-- Elevation focusing and X,Y,Z size
    if utils.isEmpty(y):
        ElevationFocusing = False
        assert x.shape == z.shape and x.shape == RC.shape, 'X, Z, and RC must be of same size.'
    else:
        ElevationFocusing = True
        assert x.shape == z.shape and x.shape == RC.shape and y.shape == x.shape,  'X, Y, Z, and RC must be of same size.'

    if len(x.shape) ==0:
         return np.array([]), np.array([])


    #%------------------------%
    #% CHECK THE INPUT SYNTAX % 
    #%------------------------%


    param = param.ignoreCaseInFieldNames()
    options = options.ignoreCaseInFieldNames()
    options.CallFun = 'simus'

    # GB TODO: wait bar + parallelisation
    #%-- Wait bar
    #if ~isfield(options,'WaitBar')
    #    options.WaitBar = true;
    #end
    #assert(isscalar(options.WaitBar) && islogical(options.WaitBar),...
    #    'OPTIONS.WaitBar must be a logical scalar (true or false).')

    #%-- Parallel pool
    #if ~isfield(options,'ParPool')
    #    options.ParPool = False
    #end

    #%-- Check if syntax errors may appear when using PFIELD
    #try:
    #    opt = options
    #    opt.ParPool = false;
    #    opt.WaitBar = false;
    #    [~,param] = pfield([],[],delaysTX,param,opt);
    #catch ME
    #    throw(ME)
    #end

    #%-- Sampling frequency (in Hz)
    if not utils.isfield(param,'fs'):
        param.fs = 4*param.fc; #% default

    assert param.fs>=4*param.fc,'PARAM.fs must be >= 4*PARAM.fc.'

    NumberOfElements = param.Nelements # % number of array elements

    #%-- Receive delays (in s)
    if not utils.isfield(param,'RXdelay'):
        param.RXdelay = np.zeros((1,NumberOfElements), dtype = np.float32)
    else:
        assert  isinstance(param.RXdelay, np.ndarray) and utils.isnumeric(param.RXdelay), 'PARAM.RXdelay must be a vector'
        assert param.RXdelay.shape[1] ==NumberOfElements, 'PARAM.RXdelay must be of length = (number of elements)'
        param.RXdelay = param.RXdelay.reshape((1,NumberOfElements))

    #%-- dB threshold (in dB: faster computation if lower value)
    if not utils.isfield(options,'dBThresh'):
        options.dBThresh = -100; # % default is -100dB in SIMUS

    assert np.isscalar(options.dBThresh) and utils.isnumeric(options.dBThresh) and options.dBThresh<0,'OPTIONS.dBThresh must be a negative scalar.'

    #%-- Frequency step (scaling factor)
    #% The frequency step is determined automatically. It is tuned to avoid
    #% aliasing in the temporal domain. The frequency step can be adjusted by
    #% using a scaling factor. For a smoother result, you may use a scaling
    #% factor<1.
    if not utils.isfield(options,'FrequencyStep'):
        options.FrequencyStep = 1

    assert np.isscalar(options.FrequencyStep) and utils.isnumeric(options.FrequencyStep) and  options.FrequencyStep>0, 'OPTIONS.FrequencyStep must be a positive scalar.'
    
    if options.FrequencyStep>1:
       logging.warning('MUST:FrequencyStep', 'OPTIONS.FrequencyStep is >1: aliasing may be present!')
    
    if not utils.isfield(param, 'c'):
         param.c = 1540 #default sound speed in soft tissue



    #%-------------------------------%
   # % end of CHECK THE INPUT SYNTAX %
   # %-------------------------------%
    
    #GB NOTE: same as in pfield, put in param ?
    #%-- Centers of the tranducer elements (x- and z-coordinates)
    xe, ze, THe, h= param.getElementPositions()

    #%-- Maximum distance
    d2 = (x.reshape((-1,1))-xe)**2+(z.reshape((-1,1))-ze)**2
    maxD = np.sqrt(np.max(d2)) #% maximum element-scatterer distance
    _, tp = getpulse.getpulse(param, 2)
    maxD = maxD + tp[-1] * param.c #add pulse length

    #%-- FREQUENCY SAMPLES
    valid_tx_delays = np.array([e for e in delaysTX.flatten() if not np.isnan(e)])
    df = 1/2/(2*maxD/param.c + np.max(np.concat((valid_tx_delays,param.RXdelay.flatten())))) # % to avoid aliasing in the time domain
    # df = 1/2/(2*maxD/param.c + np.max(delaysTX.flatten() + param.RXdelay.flatten())) # % to avoid aliasing in the time domain
    df = df*options.FrequencyStep
    Nf = 2*int(np.ceil(param.fc/df))+1 # % number of frequency samples
    #%-- Run PFIELD to calculate the RF spectra
    RFspectrum = np.zeros((Nf,NumberOfElements), dtype = np.complex64)# % will contain the RF spectra
    options.FrequencyStep = df

    #%- run PFIELD in a parallel pool (NW workers)
    if options.get('ParPool', False):
        
        with options.getParallelPool() as pool:
            idx = options.getParallelSplitIndices(x.shape[1])

            RS = pool.starmap(functools.partial(pfieldParallel, delaysTX = delaysTX, param = param, options = options),
                            [ ( x[:,i:j],
                                y[:,i:j] if not utils.isEmpty(y) else None, 
                                z[:,i:j], 
                                RC[:,i:j]) for i,j in idx ])
            

            for (RFsp, idx_spectrum) in RS: 
                RFspectrum[idx_spectrum, :] += RFsp

    #    end
    else:
        #%- no parallel pool 
        options.RC =  RC
        _, RFsp,idx = pfield(x,y,z,delaysTX,param,options)
        RFspectrum[idx,:]  = RFsp

    #%-- RF signals (in the time domain)
    nf = int(np.ceil(param.fs/2/param.fc*(Nf-1)))
    RF = np.fft.irfft(np.conj(RFspectrum),nf, axis = 0)
    RF = RF[:(nf + 1)//2] #*param.fs/4/param.fc

    #%-- Zeroing the very small values
    RelThresh = 1e-5#; % -100 dB
    tmp2= lambda RelRF: 0.5*(1+np.tanh((RelRF-RelThresh)/(RelThresh/10)))
    tmp = lambda RelRF: np.round(tmp2(RelRF)/(RelThresh/10))*(RelThresh/10)
    RF = RF*tmp(np.abs(RF)/np.max(np.abs(RF)))
    if returnTime: 
        return RF,RFspectrum, np.arange(RF.shape[0])/param.fs
    else:
         return RF,RFspectrum

