import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
import pandas as pd

rut_arc='Parac_electricos_IEEE33n.xlsx'
# === Lectura de datos ===
datos_l=pd.read_excel(rut_arc,sheet_name='Lineas')
datos_n=pd.read_excel(rut_arc,sheet_name='Nodos')
general=pd.read_excel(rut_arc,sheet_name='General')

# === Parámetros base ===
p_base=general.iloc[0, 0]   #potencia base
v_base=general.iloc[0, 1]   #voltaje base
z_base=(v_base**2)/p_base   #impedancia base



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

# === funcion objetivo ===
Funcion_objetivo= cvx.sum(sk-(pq_activa[:,0]+ 1j*pq_activa[:,1]))

# === Restricciones ===
res=[]
res+=[u[0]==1]
res+=[sk-(pq_activa[:,0]+ 1j*pq_activa[:,1]) + 1j*skc  ==Apositiva@Skm+Anegativa@Smk]   #balance de potencia
res+=[cvx.abs(sk[0]) <= 100]                                                      #nodo slack
res+=[cvx.real(sk[0]) >= 0]
res+=[u>=0.9**2]                                                                  #limite inferior de tension
res+=[u<=1.05**2]                                                                  #limite superior de tension
res += [cvx.abs(sk[1:n]) <= 0]                                                    #solo un nodo de holgura

res += [Skm == cvx.multiply(np.conj(y),((Apositiva.T @ u) - wl))]                 #flujo de potencia km
res += [Smk == cvx.multiply(np.conj(y),((Anegativa.T @ u) - cvx.conj(wl)))]       #flujo de potencia mk

for i in range(l):
    res+=[cvx.SOC(u[int(nodos_l[i,0])]+u[int(nodos_l[i,1])],cvx.vstack([2 * wl[i],u[int(nodos_l[i,0])]-u[int(nodos_l[i,1])]]))]



# === Función objetivo y solución ===
obj = cvx.Minimize(cvx.real(Funcion_objetivo))
Z= cvx.Problem(obj,res)
Z.solve('MOSEK')
print("Valor de las perdidas", obj.value*p_base, "KW", Z.status)




