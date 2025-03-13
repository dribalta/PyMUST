import pytest
import time

@pytest.mark.long
def test_long():
    time.sleep(5)
    assert 1 == 1
    
@pytest.mark.long
def test_pymust():
    import pymust
    import numpy as np
    param = pymust.getparam('P4-2v')
    xf = 2e-2
    zf = 5e-2
    txdel = pymust.txdelay(xf,zf,param)
    x = np.linspace(-4e-2,4e-2,200) # in m
    z = np.linspace(0,10e-2,200) # in m
    x,z = np.meshgrid(x,z)
    y = np.zeros_like(x)
    
    P, _, _  = pymust.pfield(x,y, z,txdel,param)
    assert P.shape == x.shape