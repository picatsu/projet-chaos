import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


#source : https://python.developpez.com/tutoriels/graphique-2d/matplotlib/
#https://matplotlib.org/gallery/api/legend.html#sphx-glr-gallery-api-legend-py

#############################
#                           #     
#      EXO 2                #
#                           #   
#############################
#Définition de la fonction f_mu:
#Q1:
def exo2_1(x,mu):
    """
    pour u E [0,4], et x E [0,1] , retourne (x*1.0)*mu*(1-x) , cad f(x) """
    return (x*1.0)*mu*(1-x)

#Q2:
def plot_f(mu):
    x = np.linspace(0,1,100)
    y = exo2_1(x,mu)
    plt.plot(x,y)
    
    plt.title("fonction plot_f")
    plt.axis('equal')
    plt.xlabel("fonction Fu")
    plt.ylabel("Valeur u")
    plt.show()
    plt.close()
plot_f(3.2)
#Q3: car le graphe dépasse l'intervalle de f(x,mu) 

#############################
#                           #     
#      EXO 3                #
#                           #   
#############################   #Simulation:
#Q1:
def exo3_1(mu,x0,n):
    """
        retourne [X0, X1, X2 ... ,Xn]  """
    S = []
    valeur = x0
    S.append(valeur)
    for i in range(0,n):
        valeur = exo2_1(valeur,mu)
        S.append(valeur)
    return S

#Q2:
def exo3_2(mu,x0,n,m):
    """
        retourne [Xm, Xm+1, Xm+2 ... ,Xn]  """
    S = []
    valeur = x0
    for i in range(0,n+1):
        valeur = exo2_1(valeur,mu)
    S.append(valeur)
    for i in range(m,n):
        valeur = exo2_1(valeur,mu)
        S.append(valeur)
    return S 
#Q3:

def test_exo3_2():
    
    plt.plot(exo3_2(3,0.5,7,3))
    plt.title("fonction S[u,n]")
    plt.xlabel("Valeur u")
    #plt.ylabel("")
    plt.show()
    plt.close()
test_exo3_2()

#############################
#                           #     
#      EXO 4                #
#                           #   
#############################
def exo4_q1(mu,n,x0,y0):
    """
     f( mu, n, X0, Y0) = [X0,...,Xn] , [Y0,...,Yn]
     """
    #liste.append([x0,y0])
    absc = x0
    ordn = y0
    L = np.array([absc])
    M = np.array([ordn])
    for i in range(0,n):
        absc = exo2_1(absc,mu)
        ordn = exo2_1(ordn,mu)
        L = np.append(L,[absc])
        M = np.append(M,[ordn])
    return L,M
   
def test_exo4_q1():
    liste=exo4_q1(3,17,0.5,0.4)
    plt.title("representation graphique de EXO 4.1")
    
    plt.scatter(*zip(liste))
    plt.show()  
#test_exo4_q1()
 
#Q2:
def graphe(mu,n,x0,y0):
    
    # plt.plot([x1,x2] , [y1, y2], style)  NORME
    
    plt.plot([0,1],[0,1],'k-')      # IDENTITE
    plt.title("graphe de EXO 4.2")

    a = np.linspace(0,1,100)
    fu = exo2_1(a,mu)
    plt.plot(a,fu)                   #fpnction Fu
    
    liste=exo4_q1(mu,n,x0,y0)
    #plt.plot( [liste[0][0],0] , [liste[0][0], liste[1][1]],'r-')# (x0,0) => (x0, y1)
    xx=liste[0]
    yy=liste[1]
    for i in range(0, len(liste[0]) -2):
        """
        (x0,0)    => (x0,x1) 
        (x0, x1)  => (x1 x1)
        (x1,x2)   => (x1, x2) 
        """
        #plt.plot( [   x1       ,x2         ] , [      y1  ,      y2      ], style)  NORME
        plt.plot( [xx[i],xx[i+1]] , [yy[i], yy[i+1]],'r-')#b = (x0,y0) -> (x1,y1)
        
        #plt.plot( [liste[0][i],liste[0][i+1]] , [liste[1][i], liste[1][i]],'r-')#decalage horizontale
    
        plt.plot( [xx[i],xx[i]] , [yy[i+1], yy[i+2]],'r-')#c = (x1,y1) -> (x1,y2) 
    plt.show()     
graphe(3.2,100,0.1,0.2)   
 

#############################
#                           #     
#      EXO 5                #
#                           #   
#############################

def listeu(n,m):#remplir une liste de u , de taille n-m
    
    liste=[]
    for i in range(1,5):
        a=i
        for j in range(0,(n-m)):
            #0.00955 
            a=a + 0.009513999 
            liste.append(a) # retourne une liste de flottant entre [1,4]
    return liste

def exo5_q1(mu, x0, n, m):
    """
        => [ (U0,Xm), (U1,Xm+1), ... (Un, Xn)]
        """
    listex=[]
    listeuu=[]
    listex.append(x0)
    listeuu.append(mu[0])    #recopie liste mu
    a=x0
    for i in range(1,len(mu) -1):
        a=exo2_1(a,mu[i])     # a = Fu(xi, Ui)
        listex.append(a)      # liste.append(a)
        listeuu.append(mu[i])
        
    return listeuu,listex

 #○print('EXO 5 \n : ',exo5_q1(listeu(200,100),0.1,200,100))

def test_exo5(mu,x0,n,m):
    
    #plt.axis([x0,xmax, y0, ymax])
    plt.axis([0, 5.0, 0, 1])
    liste = exo5_q1(mu,x0,n,m)
    plt.title("graphe de EXO 5.2")
    
    plt.scatter(liste[0], liste[1], s=1)
    plt.xlabel("U")
    plt.ylabel("Xi")
    plt.show()
     
listeu = listeu(200,100)

test_exo5(listeu,0.001,200,100)   
#a proche de 1 
#b  pour des valeurs de u entre 1 et 3 
#c  pour des valeur de u entre 3.0 et 3.5 
#d entre 3.5 et 3.6 
#e       
    
#############################
#                           #     
#      EXO 6                #
#                           #   
#############################
def exo6(mu,x0, n, m):
    """
        => [ (Xm, Xm+1), (Xm+1, Xm+2) ... (Xn, Xn+1)]
        => [Xm, Xm+1 ... Xn], [Xm+1,Xm+2 .. Xn+1]
        """
    liste=exo3_2(mu,x0,n,m)
    listem=[]
    listem1=[]
    
    for i in range(0,len(liste)-1):
        listem.append(liste[i])
        listem1.append(liste[i+1])
        
    listem.append(exo2_1(n,mu))
    listem1.append(exo2_1(n+1,mu))
    
    return listem,listem1            #
   
def test_exo6(mu,x0,n,m):
    
    #plt.axis([x0,xmax, y0, ymax])
    plt.axis([0, 4.0, 0, 1])
    
    liste =exo6(mu,x0,n,m)
    plt.title("graphe de EXO 6.2")
    
    plt.scatter(liste[0], liste[1], s=1)
    plt.xlabel("U")
    plt.ylabel("Xi")
    plt.show()   
test_exo6(3.9,0.1,5000,100)








    
        
        
        
    

    
