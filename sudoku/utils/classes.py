# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:50:42 2020

@author: alspe
"""


import os
import datetime
import pandas as pd
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt

from configs.environment import Config



j = complex(0, 1)

class VFA_Ilyes:
        
    def __init__(self, n_modes, w_values, mat, alpha_r, alpha_i):
        self.n            = n_modes
        self.w_values     = w_values # 100000 valeurs
        self.mat          = mat
        self.sub_w_values = np.array([w for i, w in enumerate(self.w_values) if i%1000==0]) # 1000 valeurs
        self.b_           = mat.rho_0*mat.k_0*mat.alpha_inf/(mat.eta*mat.phi)
        self.alpha_inf    = mat.alpha_inf
        self.alpha_r      = alpha_r
        self.alpha_i      = alpha_i
        
    
    def update(self, a=None, b=None, c=None, d=None, e=None, f=None, g=None, r_k=None, s_k=None, mu_k=None, xi_k=None):
        if a is not None:
            self.a    = a
        if b is not None:
            self.b    = b
        if c is not None:
            self.c    = c
        if d is not None:
            self.d    = d
        if e is not None:
            self.e    = e
        if f is not None:
            self.f    = f
        if g is not None:
            self.g    = g
        if r_k is not None:
            self.r_k  = r_k
        if s_k is not None:
            self.s_k  = s_k
        if mu_k is not None:
            self.mu_k = mu_k
        if xi_k is not None:
            self.xi_k = xi_k
        
    
    def evaluate(self, w):
        # s = jw avec w € R*+
        
        if Config.CASE_STUDY == "alpha":
            f = self.alpha_inf + self.alpha_inf/(self.b_*j*w) + \
                np.sum(self.mu_k / (j*w + np.abs(self.xi_k)))
            return f
        
        elif Config.CASE_STUDY == "f(w)":
            f = self.a*j*w + self.b + self.c/(j*w) + \
                (self.d*j*w + self.e)*np.sum(self.r_k  / (j*w - self.s_k)) + \
                (self.f*j*w + self.g)*np.sum(self.mu_k  / (j*w - self.xi_k))
            return f
    
            
    def compute(self, all_values=True):
        # On ne prend qu'un sous ensemble, car avec les 100000 de self.w_values,
        # cette fonction prend ~1.3s
        # Avec seulement 1000 la fonction dure maintenant ~0.016s
        if all_values:
            w_list = self.w_values
        else:
            w_list = self.sub_w_values
        
        f_values = np.empty(shape=w_list.shape, dtype=complex)
        for k in range(w_list.shape[0]):
            f_values[k] = self.evaluate(w_list[k])
        
        df = pd.DataFrame({"w": w_list,
                           "alpha": f_values,
                           "alpha_r": np.real(f_values),
                           "alpha_i": np.imag(f_values)})
        return df
    
            
    def save(self, path_name):
        # Enregistre les paramètres déterminés sous format JSON
        if not os.path.isdir('results'):
            os.mkdir('results')
            
        df = self.compute()
        df['true_r'] = self.alpha_r
        df['true_i'] = self.alpha_i
        df.to_csv("results/{0}/tortuosite.csv".format(path_name), index=False)




class Summary:
        
    def __init__(self):
        # On vérifie qu'il existe un dossier "results"
        if not os.path.isdir('results'):
            os.mkdir('results')
        
        # On crée le dossier de résultats du nom de la date
        date = datetime.datetime.today()
        self.path_name = date.isoformat().replace(":","-").split(".")[0]
        os.mkdir(os.path.join('results', self.path_name))
        os.mkdir(os.path.join(os.path.join('results', self.path_name), 'fig'))
        
        self.file_name = 'results/{0}/params.csv'.format(self.path_name)
        
        # On copie les fichiers configs/environment.py et configs/dqn.py
        if not os.path.isdir(os.path.join('results', 'configs')):
            os.mkdir(os.path.join('results', 'configs'))
            for _file in ['dqn.py', 'environment.py']:
                copyfile(os.path.join('configs', _file),
                         os.path.join(os.path.join('results', 'configs'), _file))
        
        
    def create(self):
        # On initialise le fichier csv par l'entête
        with open(self.file_name, "w") as file:
            file.write("current_step,r2_coef,reward,Re(mu[0]),Im(mu[0]),done,max_r2_coef\n")
            
        
    def update(self, current_step, r2_coef, reward, mu, done, max_r2_coef):
        # On ajoute une ligne au fichier
        with open(self.file_name, "a") as file:
            file.write("{0},{1},{2},{3},{4},{5},{6}\n".format(current_step, 
                                                      r2_coef, 
                                                      reward, 
                                                      mu.real, 
                                                      mu.imag,
                                                      done,
                                                      max_r2_coef))
            
    def plot(self, mode="humain"):    
        # On charge les résultats
        df = pd.read_csv(self.file_name)
        
        if Config.REWARD_TYPE=="re_coef":
            label        = "R² coef"
            label_extrem = "max(R²)"
        elif Config.REWARD_TYPE=="mse":
            label        = "MSE"
            label_extrem = "min(MSE)"
        
        # coefficient R² et reward selon les étapes d'entraînement
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 2, 1)
        plt.xlabel("step")
        plt.grid(True, linestyle='--')
        plt.scatter(df["current_step"], df["r2_coef"], label=label)
        plt.scatter(df["current_step"], df["max_r2_coef"], label=label_extrem)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.xlabel("step")
        plt.grid(True, linestyle='--')
        plt.scatter(df["current_step"], df["reward"], label="reward")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/{0}/fig/reward.png'.format(self.path_name))
        if mode=="humain":
            plt.show()
        else:
            plt.close()
        
        # Affichage des valeurs du complexe "a"
        plt.figure(figsize=(4+1, 4))
        
        plt.xlabel("Re")
        plt.ylabel("Im")
        plt.grid(True, linestyle='--')
        plt.scatter(df["Re(mu[0])"], df["Im(mu[0])"], c=df["current_step"], label=None)
        plt.colorbar(label='step')
        
        plt.tight_layout()
        plt.savefig('results/{0}/fig/param_mu_0.png'.format(self.path_name))
        if mode=="humain":
            plt.show()
        else:
            plt.close()
        
        # Affichage des étape "terminées"
        plt.figure(figsize=(4, 4))
        
        plt.yticks([0, 1], ("continue", "done"))
        plt.bar(df["current_step"], df["done"], width=1.)
        
        plt.tight_layout()
        plt.savefig('results/{0}/fig/done.png'.format(self.path_name))
        if mode=="humain":
            plt.show()
        else:
            plt.close()