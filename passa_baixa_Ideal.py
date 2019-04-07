#Thales de Castro Soares 86958

#Passa baixas ideal
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as sci

f = np.zeros((256,256)).astype(float)
f[117:137,117:137] = 1
F = np.fft.fft2(f)

fig1, (ax1,ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
ax1.imshow(f, cmap='gray',vmin=0,vmax=1)
ax1.set_title('Dom. Espaço')
ax1.axis('off')
ax2.imshow(np.abs(np.fft.fftshift(F)), cmap='gray')
ax2.set_title('Dom. Frequencia')
ax2.axis('off')

size = f.shape
raio = 20

x = np.linspace(0, size[0]-1, size[0])-127
y = np.linspace(0, size[1]-1, size[1])-127
xv, yv = np.meshgrid(x, y)

gauss = np.random.normal(2,1,(size[0]*size[1]))
while(sci.stats.kstest(gauss,'norm') == 'false'):
    gauss = np.random.normal(2,1,(size[0]*size[1]))

z = np.zeros(size)
z = np.sqrt( np.multiply(xv,xv) + np.multiply(yv,yv) )

mask = gauss[0:]
mask = np.fft.fftshift(mask)
G = np.multiply(F,mask.shape[0])

g = np.fft.ifft2(G)

fig2, (ax3,ax4) = plt.subplots(ncols=2, sharex=True, sharey=True)
ax3.imshow(np.real(g), cmap='gray',vmin=0,vmax=1)
ax3.set_title('Dom. Espaço')
ax3.axis('off')
ax4.imshow(np.abs(np.fft.fftshift(G)), cmap='gray')
ax4.set_title('Dom. Frequencia')
ax4.axis('off')