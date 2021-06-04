# -*- coding: utf-8 -*-


import networkx as nx
G=nx.Graph()

    
def matrix_link(m,k,a,b):
    m.add_edge(a,b)

matrix_link(G,matrix_key,'MV101','LIT101')
matrix_link(G,matrix_key,'FIT101','LIT101')
matrix_link(G,matrix_key,'LIT101','P101')
matrix_link(G,matrix_key,'P101','FIT201')
matrix_link(G,matrix_key,'P101','AIT201')
matrix_link(G,matrix_key,'FIT201','P201')
matrix_link(G,matrix_key,'AIT201','P201')
matrix_link(G,matrix_key,'P201','P203')
matrix_link(G,matrix_key,'P203','P205')
matrix_link(G,matrix_key,'P205','MV201')
matrix_link(G,matrix_key,'MV201','AIT202')
matrix_link(G,matrix_key,'MV201','AIT203')
matrix_link(G,matrix_key,'AIT202','LIT301')
matrix_link(G,matrix_key,'AIT203','LIT301')
matrix_link(G,matrix_key,'LIT301','P301')
matrix_link(G,matrix_key,'P301','DPIT301')
matrix_link(G,matrix_key,'DPIT301','LIT401')
matrix_link(G,matrix_key,'LIT401','P401')
matrix_link(G,matrix_key,'P401','FIT401')
matrix_link(G,matrix_key,'FIT401','AIT402')
matrix_link(G,matrix_key,'AIT402','AIT503')
matrix_link(G,matrix_key,'AIT503','P501')
matrix_link(G,matrix_key,'P501','AIT504')
matrix_link(G,matrix_key,'P501','P602')
matrix_link(G,matrix_key,'P602','DPIT301')

import matplotlib.pyplot as plt
nx.draw(G,pos = nx.spring_layout(G),node_color = 'b',edge_color = 'r',with_labels = True,font_size =20,node_size =50)
plt.show()

















