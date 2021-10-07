# EIS parameter regression & i0 regression from Rct values
# required packages: numpy
# install via cmd : python -m pip install numpy

import numpy as np

# supply frequency numpy array and associated array of complex impedances
def regress_simplified_randles_cell(f,Z,fmin=0,scalechange=0.1,maxiter=10000,Rct=1000,Rsol=10,Cdl=1e-7,n=0.8,tryparams=False,attempts=1000):
    Z    = Z[f>fmin]
    f    = f[f>fmin]
    Zm   = lambda Rs,Rc,C,n : Rs + (Rc+(f*np.pi*2)**n*C*Rc**2*np.cos(n*np.pi/2)-1j*(np.pi*2*f)**n*C*Rc**2*np.sin(n*np.pi/2))/(1+2*(f*np.pi*2)**n*C*Rc*np.cos(n*np.pi/2)+(np.pi*2*f)**(2*n)*C**2*Rc**2)
    err  = lambda Rs,Rc,C,n : (z:=Zm(Rs,Rc,C,n),) and np.sum(((z.real-Z.real)*np.log10(f))**2+((z.imag-Z.imag)*np.log10(f))**2)
    err0 = err(Rsol,Rct,Cdl,n)

    if tryparams:
        for i in range(attempts):
            Rs = Rsol*2**np.random.normal(loc=0,scale=2)
            Rc = Rct*2**np.random.normal(loc=0,scale=2)
            Cd = Cdl*2**np.random.normal(loc=0,scale=2)
            n_ = 1/(5**np.random.random())
            e = err(Rs,Rc,Cd,n_)
            if err0>e:
                Rsol = Rs
                Rct  = Rc
                Cdl  = Cd
                n    = n_
                err0 = e
    
    err0 = err(Rsol,Rct,Cdl,n)
    errinit = err0
    
    param = np.array([Rsol,Rct,Cdl,n],dtype='float64')
    scale = [2,2,2,1.1]
    
    for i in range(maxiter*len(param)):
        p = np.random.randint(len(param))
        scaletmp   = [scale[k] if k==p else 1 for k in range(len(param))]
        paramplus  = param*scaletmp
        paramminus = param/scaletmp

        errplus  = err(*paramplus)
        errminus = err(*paramminus)
        if errplus<errminus:
            errnew = errplus
            paramnew = paramplus
        else:
            errnew = errminus
            paramnew = paramminus
        
        if err0>errnew:
            param = paramnew
            err0 = errnew
            scale[p] **= 1+np.random.random()*scalechange
        else:
            scale[p] **= 1/(1+np.random.random()*scalechange/len(scale))
    return param

def i0_regression(Rct,Cred,Cox,T=298.15,z=1,accuracy=3):
    F   = 96850
    R   = 8.314
    i0  = R*T/z/F/Rct
    f   = lambda beta,k0 : F * k0 * Cred**beta * Cox**(1-beta)

    # first step: fit beta parameter;
    # -> find beta which yields values that are proportional to exp data
    # -> minimize standard deviation of factor
    
    err = lambda beta : (factor:=f(beta,0.01)/i0,
                         avfact:=sum(factor)/len(factor)) and \
                        np.sum((factor-avfact)**2)/(len(Rct)-1)

    values = np.linspace(0,1,int(10**accuracy+1))
    errors = []
    for v in values: errors.append(err(v))
    errors = np.array(errors)
    beta = values[errors.argmin()]
    
    # second step: fit k0 scalar

    sum1 = i0.sum()
    sum2 = f(beta,1).sum()

    k0 = sum1/sum2

    return beta,k0
