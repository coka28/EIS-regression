# EIS data visualisation

from regressions import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings('ignore')
plt.ion()
plt.rcParams.update({'mathtext.default': 'regular'})

def i0_plot(Rct,Cred,Cox,beta,k0):
    R = 8.314
    F = 96850
    z = 1
    T = 298.15
    val = np.linspace(1/32,31/32,1001)
    f_Rct = lambda i0 : R*T/z/F/i0
    i0_ = F*k0*((Cred[0]+Cox[0])*val)**beta*((Cred[0]+Cox[0])*(1-val))**(1-beta)

    fig,ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('R$_{ct}$',rotation=0,fontsize=11)
    ax.yaxis.set_label_coords(-0.05,0.95)
    ax.set_xlabel(r'$\frac{\underset{\,}{C_{red}}}{\overset{\,}{C_{ges}}}$',fontsize=14)
    ax.xaxis.set_label_coords(1.05,-0.025)
    ax.plot(Cred/(Cred+Cox),Rct,'x',c='#f00',label='Modelldaten')
    ax.plot(val,f_Rct(i0_),label='extrapolierter Fit')
    ax.set_ylim(0,None)
    ax.set_xlim(0,1)
    fig.legend()
    fig.tight_layout()

class marker_handler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        scale = fontsize / 12
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch_sq = mpatches.Rectangle([x0, y0 + height/2 * (1 - scale) ], height * scale, height * scale, facecolor='none',
                edgecolor='#4d0', transform=handlebox.get_transform())
        patch_circ = mpatches.Circle([x0 + width - height/2, y0 + height/2], height/2 * scale, facecolor='none',
                edgecolor='#c80', transform=handlebox.get_transform())

        handlebox.add_artist(patch_sq)
        handlebox.add_artist(patch_circ)
        return patch_sq

class line_handler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        scale = fontsize / 12
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height

        mod_line1 = mlines.Line2D([x0, x0+width ], [height*scale-4,height*scale-4],color='#4d0', transform=handlebox.get_transform())
        mod_line2 = mlines.Line2D([x0, x0+width ], [height*scale,height*scale],color='#c80', transform=handlebox.get_transform())

        handlebox.add_artist(mod_line1)
        handlebox.add_artist(mod_line2)
        return mod_line1

class markerdummy(object):
    pass

class linedummy(object):
    pass

def IS_plot(filename,fmin=20,**model):
    with open(filename,'r') as file:
        txt = file.read()
        lines = txt.splitlines()

    fm = np.logspace(0,10,1001,base=np.e)

    potential = lines[3][14:]
    current   = lines[4][14:]
    data = [ln.split(' ') for ln in lines[19:]]
    data = [[d for d in dat if d] for dat in data]
    data = [dat for dat in data if len(dat)==6]
    f = np.array([float(dat[1]) for dat in data])
    Z = np.array([float(dat[2])+float(dat[3])*1j for dat in data])

    fig,ax = plt.subplots(1,2,figsize=(12,5.5))
    nyq = ax[0]
    nyq.invert_yaxis()
    bodZ = ax[1]
    bodP = bodZ.twinx()

    if model:
        Rct = model['Rct']
        Rsol = model['Rsol']
        Cdl = model['Cdl']
        n = model['n']
        Zm = Rsol + (Rct+(fm*np.pi*2)**n*Cdl*Rct**2*np.cos(n*np.pi/2)-1j*(np.pi*2*fm)**n*Cdl*Rct**2*np.sin(n*np.pi/2))/(1+2*(fm*np.pi*2)**n*Cdl*Rct*np.cos(n*np.pi/2)+(np.pi*2*fm)**(2*n)*Cdl**2*Rct**2)
    else:
        mparam = regress_simplified_randles_cell(f,Z,fmin=fmin,maxiter=10000,tryparams=True,attempts=1000)
        z_src = lambda Rs,Rc,C,n : Rs + (Rc+(fm*np.pi*2)**n*C*Rc**2*np.cos(n*np.pi/2)-1j*(np.pi*2*fm)**n*C*Rc**2*np.sin(n*np.pi/2))/(1+2*(fm*np.pi*2)**n*C*Rc*np.cos(n*np.pi/2)+(np.pi*2*fm)**(2*n)*C**2*Rc**2)
        Zm = z_src(*mparam)

    if model: nqm = nyq.plot(Zm.real,Zm.imag,c='#8af',label='Modell') # lab regression
    else: nqM = nyq.plot(Zm.real,Zm.imag,c='#8af',label='Modell') # home regression
    nqd = nyq.plot(Z.real,Z.imag,'x',c='#f00',label='Messpunkte')
    if model: bZm = bodZ.plot(np.log(fm),np.log(np.abs(Zm)),'-',c='#b60',label='Modell (k=1)')
    else: bZm = bodZ.plot(np.log(fm),np.log(np.abs(Zm)),c='#b60',label='Modell')
    bZd = bodZ.plot(np.log(f),np.log(np.abs(Z)),'o',markerfacecolor='None',c='#fa0',label='Messpunkte (k=1)')
    if model: bPm = bodP.plot(np.log(fm),-np.arctan(Zm.imag/Zm.real),'-',c='#2a0',label='Modell (k=2)')
    else: bPm = bodP.plot(np.log(fm),-np.arctan(Zm.imag/Zm.real),c='#2a0',label='Modell')
    bPd = bodP.plot(np.log(f),-np.arctan(Z.imag/Z.real),'s',markerfacecolor='None',c='#7f0',label='Messpunkte (k=2)')
    
    ax   = [nyq,bodZ,bodP]
    ylim = [(None,None),(0,None),(-0.2,np.pi/2+0.2)]
    ytix = [False,False,[0,np.pi/4,np.pi/2]]
    ytxl = [None, None, ['0°','-45°','-90°']]
    sphd = ['right','right','left']
    spcx = ['left','left','right']
    spcl = ['#000','#c80','#4d0']
    ylbl = ['Z\u2097\u2098 [k\u03a9]','log |Z|','\u03d5']
    ypos = [(-0.025,1.025),(-0.025,1.025),(1.05,1.05)]
    xlbl = ['Z\u1d63\u2091 [k\u03a9]','log f',False]
    xpos = [(0.5,-0.08),(0.5,-0.08),False]
    grid = [False,False,True]

    ax[0].set_title('Nyquist Plot')
    ax[1].set_title('Bode Plot')

    for i in (0,1,2):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines[sphd[i]].set_visible(False)
        ax[i].spines[spcx[i]].set_color(spcl[i])
        ax[i].set_ylabel(ylbl[i],rotation=0,fontsize=11)
        if ytix[i]:
            ax[i].set_yticks(ytix[i])
            ax[i].set_yticklabels(ytxl[i])
        ax[i].set_ylim(*ylim[i])
        ax[i].yaxis.set_label_coords(*ypos[i])
        ax[i].yaxis.label.set_color(spcl[i])
        ax[i].tick_params(axis='y',colors=spcl[i])
        if xlbl[i]:
            ax[i].set_xlabel(xlbl[i],fontsize=11)
            ax[i].xaxis.set_label_coords(*xpos[i])
        ax[i].grid(grid[i])

    # scale x:y 1 to 1 ?
    nyqbox = nyq.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    w,h = nyqbox.width,nyqbox.height
    lim = np.min([Z.real.max(),abs(Z.imag).max()])*1.05
    ax[0].set_xlim(None,lim)
    ax[0].set_ylim(None,np.sign(Z.imag[abs(Z.imag).argmax()])*lim)

    nyq.legend()
    ax[2].legend([linedummy(),markerdummy()],['Modell','Messpunkte'],handler_map={linedummy: line_handler(), markerdummy: marker_handler()})

    nyq.set_yticklabels([str(y) for y in nyq.get_yticks()/1000])
    nyq.set_xticklabels([str(x) for x in nyq.get_xticks()/1000])
    fig.tight_layout()
    res = {'Rsol':mparam[0], 'Rct':mparam[1], 'Cdl':mparam[2], 'n':mparam[3]}
    print(filename)
    print(*['  %s \t: %s'%(k,v) for k,v in res.items()],sep='\n')
    return res

Rct  = np.array([123.24,61.86,205.04])
Cred = np.array([0.012,0.0064,0.0008])
Cox  = np.array([0.0008,0.0064,0.012])
beta,k0 = i0_regression(Rct,Cred,Cox)
i0_plot(Rct,Cred,Cox,beta,k0)
