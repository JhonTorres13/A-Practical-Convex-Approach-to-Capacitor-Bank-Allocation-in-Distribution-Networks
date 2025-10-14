import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
import pandas as pd

# === Lectura de datos ===
datos_l=pd.read_excel('Parac_electricos_IEEE33n.xlsx',sheet_name='Lineas')
datos_n=pd.read_excel('Parac_electricos_IEEE33n.xlsx',sheet_name='Nodos')
general=pd.read_excel('Parac_electricos_IEEE33n.xlsx',sheet_name='General')

# === Parámetros base ===
p_base=general.iloc[0, 0]   #potencia base
v_base=general.iloc[0, 1]   #voltaje base
z_base=(v_base**2)/p_base

# === Datos de red ===
l=datos_l.shape[0]     # nodos
n=datos_n.shape[0]     # lineas

# #pasar datos a p.u.
nodos_l=np.array(datos_l[['Nodo i', 'Nodo j']]-1).reshape(l,2)
rx=np.array(datos_l[['resistencia [ohmio]', 'reactancia [ohmio]']]/z_base).reshape(l,2)
y = 1/(rx[:,0]+1j*rx[:,1])  # vector de admitancias
nodos_n=np.array(datos_n['Nodo']-1).reshape(n,1)
pq_activa=np.array((datos_n[['Pload [Kw]', 'Qload  [Kvar]']]*1000)/p_base).reshape(n,2)


# === Matrices de incidencia ===
Apositiva=np.zeros((n,l))             #flujos envio
Anegativa=np.zeros((n,l))             #flujos recibo
for i in range(l):
    ne=int(nodos_l[i,0])
    nr=int(nodos_l[i,1])
    Apositiva[ne,i]=1
    Anegativa[nr,i]=1

# === Variables ===
sk = cvx.Variable((n),complex=True)  #variable compleja potencia generada en el nodo k
Skm= cvx.Variable((l),complex=True)  #variable compleja flujo de potencia de k a m
Smk= cvx.Variable((l),complex=True)  #variable compleja flujo de potencia de m a k
u = cvx.Variable((n))                #variable real tension
wl= cvx.Variable((l),complex=True)   #variable compleja

# === variables capacitores Kvar===
capacitores=np.linspace(150,2100,14)/(10**3)
z = cvx.Variable((n,len(capacitores)), boolean=True) # variable binaria ubicar en el nodo k
skc=cvx.Variable((n))                                      # variable la potencia reactiva capacitores



# === Restricciones ===
Funcion_objetivo= cvx.sum(sk-(pq_activa[:,0]+ 1j*pq_activa[:,1]))
res=[]
res+=[u[0]==1]
res+=[sk-(pq_activa[:,0]+ 1j*pq_activa[:,1]) + 1j*skc  ==Apositiva@Skm+Anegativa@Smk]   #balance de potencia
res+=[u>=0.8]                                                                  #limite inferior de tension
res+=[u<=1.1]                                                                  #limite superior de tension

res += [cvx.abs(sk[1:n]) <= 0]                                               # solo un nodo de holgura

res += [Skm == cvx.multiply(np.conj(y),((Apositiva.T @ u) - wl))]            #flujo de potencia km
res += [Smk == cvx.multiply(np.conj(y),((Anegativa.T @ u) - cvx.conj(wl)))]  #flujo de potencia mk

for i in range(l):
    res+=[cvx.SOC(u[int(nodos_l[i,0])]+u[int(nodos_l[i,1])],cvx.vstack([2 * wl[i],u[int(nodos_l[i,0])]-u[int(nodos_l[i,1])]]))]


# === Restricciones de los capacitores ===
res += [skc == z @ capacitores]
res += [cvx.sum(z,axis=0)  <= 1]
numero_capacitores_disponibles=3      #Numero de capacitores disponibles
res += [cvx.sum(z) <= numero_capacitores_disponibles]


# === Función objetivo y resolución ===
obj = cvx.Minimize(cvx.real(Funcion_objetivo))
Z= cvx.Problem(obj,res)
Z.solve("MOSEK")
print("Valor de las perdidas", obj.value*p_base, "KW", Z.status)


# === mostrar capacitores y nodos ===
nodos_capacitores=[]
for i in range(n):
    if sum(z.value[i,:])==1:
        nodos_capacitores.append(i+1)

valor_capacitor=[]
for k in range(n):
    for i in range(capacitores.shape[0]):
        if z.value[k,i] == 1:
            valor_capacitor.append(capacitores[i]*(10**3))


print("nodos donde se instalaron capacitores",  nodos_capacitores)
print("valor del capacitor instalado",  valor_capacitor)


voltajes_base=[1.,0.99403784,0.9660573,0.95135156,0.93691982,0.90148963,
 0.89480871,0.86915976,0.85739213,0.84657937,0.84498145,0.84219874,
 0.83090185,0.82673222,0.82413957,0.82163227,0.81792348,0.81681449,
 0.99298453,0.98586715,0.9844686,0.98320409,0.95902106,0.94599873,
 0.93954148,0.89782902,0.89297606,0.87148215,0.85620271,0.84963052,
 0.84197504,0.84029555,0.8397755]

ancho=10
alto=5
x=np.linspace(0,32,33)
voltajes_base1=np.sqrt(voltajes_base)
voltajes=np.sqrt(u.value)
plt.figure(1,figsize=(ancho, alto))

plt.plot(voltajes_base1,"purple",label="Voltajes caso base")
plt.plot(x,voltajes_base1,"o",mfc="black",mec="purple")

plt.plot(voltajes,"orange",label="Voltajes con capacitores")
plt.plot(x,voltajes,"o",mfc="black",mec="orange")

plt.xlabel('Nodo', fontsize=13)    #definir el eje x
plt.ylabel('Voltaje', fontsize=13)  # definir el eje y
plt.xlim(0,33)
plt.ylim(0.9,1.02)
plt.legend()
plt.grid()
plt.show()



