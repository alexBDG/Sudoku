# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 09:28:44 2020

@author: alspe
"""


import os
import glob
import cmath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from configs.environment import Config



def correlation(df, pred_df, split=False):
    """
    Calcule le coefficient r2 pour l'ensemble [[Re(z) for z in df] [Im(z) for z in df]]

    Parameters
    ----------
    df : pandas DataFrame
        Les vraies valeurs de la fonction.
    pred_df : pandas DataFrame
        Les valeurs explorées.

    Returns
    -------
    r2coef : float
        Le coefficient r2 calculé.

    """
    
    r2coef_r = r2_score(df.alpha_r.values,
                        pred_df.alpha_r.values)
    r2coef_i = r2_score(df.alpha_i.values,
                        pred_df.alpha_i.values)
    
    if split:
        return r2coef_r, r2coef_i

    else:
        return (r2coef_r + r2coef_i)/2.




def mse(df, pred_df, split=False):
    """
    Calcule l'erreur quadratique moyenne (MSE) pour l'ensemble [[Re(z) for z in df] [Im(z) for z in df]]

    Parameters
    ----------
    df : pandas DataFrame
        Les vraies valeurs de la fonction.
    pred_df : pandas DataFrame
        Les valeurs explorées.

    Returns
    -------
    r2coef : float
        L'erreur quadratique moyenne calculée.

    """
    
    mse_coef_r = mean_squared_error(df.alpha_r.values,
                                  pred_df.alpha_r.values)
    mse_coef_i = mean_squared_error(df.alpha_i.values,
                                  pred_df.alpha_i.values)
    
    if split:
        return mse_coef_r, mse_coef_i

    else:
        return (mse_coef_r + mse_coef_i)/2.




def plot(df, pred_df, mode="humain", path_name=None):
    # Réels et Imaginaires selon la pulsation
    plt.figure(figsize=(2*4, 4))
    
    plt.subplot(1, 2, 1)
    plt.xlabel("$\omega$")
    plt.xscale('log')
    plt.ylabel("Re")
    plt.grid(True, linestyle='--')
    plt.scatter(df["w"],      df["alpha_r"],      s=50, label="JCAPL")
    plt.scatter(pred_df["w"], pred_df["alpha_r"], s=25, label="pred")
    plt.title("$R^2$=%.2f, MSE=%.2f" % (r2_score(df["alpha_r"], pred_df["alpha_r"]), 
                                        mean_squared_error(df["alpha_r"], pred_df["alpha_r"])))
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.xlabel("$\omega$")
    plt.xscale('log')
    plt.ylabel("Im")
    plt.grid(True, linestyle='--')
    plt.scatter(df["w"],      df["alpha_i"],      s=50, label="JCAPL")
    plt.scatter(pred_df["w"], pred_df["alpha_i"], s=25, label="pred")
    plt.title("$R^2$=%.2f, MSE=%.2f" % (r2_score(df["alpha_i"], pred_df["alpha_i"]), 
                                        mean_squared_error(df["alpha_i"], pred_df["alpha_i"])))
    plt.legend()
    
    plt.tight_layout()
    if path_name is not None:
        plt.savefig('results/{0}/fig/reel_w_imag_w.png'.format(path_name))
    if mode=="humain":
        plt.show()
    else:
        plt.close()
    
    
    # Réel suivis d'Imaginaire selon pulsation
    plt.figure(figsize=(8, 8))    
    
    plt.xlabel("$\omega$")
    plt.xscale('log')
    plt.ylabel("Im")
    plt.grid(True, linestyle='--')
    plt.scatter(df["w"].values + df["w"].values[-1]*pred_df["w"].values, 
                df["alpha_r"].values + df["alpha_i"].values,
                s=50, label="JCAPL")
    plt.scatter(df["w"].values + df["w"].values[-1]*pred_df["w"].values, 
                pred_df["alpha_r"].values + pred_df["alpha_i"].values, 
                s=25, label="pred")
    plt.title("$R^2$=%.2f, MSE=%.2f" % (r2_score(df["alpha_r"].values      + df["alpha_i"].values, 
                                                 pred_df["alpha_r"].values + pred_df["alpha_i"].values), 
                                        mean_squared_error(df["alpha_r"].values      + df["alpha_i"].values,
                                                           pred_df["alpha_r"].values + pred_df["alpha_i"].values)))
    plt.legend()
    
    plt.tight_layout()
    if mode=="humain":
        plt.show()
    else:
        plt.close()
    
    
    # Imaginaire selon Réel
    plt.figure(figsize=(8, 8))
    
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.grid(True, linestyle='--')
    plt.scatter(df["alpha_r"],      df["alpha_i"],      s=50, label="JCAPL")
    plt.scatter(pred_df["alpha_r"], pred_df["alpha_i"], s=25, label="pred")
    plt.title("$R^2$=%.2f, MSE=%.2f" % (r2_score(df["alpha_r"].values      + df["alpha_i"].values, 
                                                 pred_df["alpha_r"].values + pred_df["alpha_i"].values), 
                                        mean_squared_error(df["alpha_r"].values      + df["alpha_i"].values,
                                                           pred_df["alpha_r"].values + pred_df["alpha_i"].values)))
    plt.legend()
    
    plt.tight_layout()
    if path_name is not None:
        plt.savefig('results/{0}/fig/reel_imag.png'.format(path_name))
    if mode=="humain":
        plt.show()
    else:
        plt.close()
    

    # Gain selon pulsation et Phase selon pulsation
    y_true = list(cmath.polar(complex(z)) for z in df["alpha"].values)
    y_pred = list(cmath.polar(complex(z)) for z in pred_df["alpha"].values)
    
    plt.figure(figsize=(8, 8))
        
    plt.subplot(2, 1, 1)
    plt.semilogx(df["w"],      [r for (r, phi) in y_true], linewidth=4, label="JCAPL")
    plt.semilogx(pred_df["w"], [r for (r, phi) in y_pred], linewidth=2, label="pred")
    plt.legend()
    plt.grid(True,linestyle='--')
    plt.xlabel("$\omega$")
    plt.xscale('log')
    plt.title("Magnetude")
    
    plt.subplot(2, 1, 2)
    plt.semilogx(df["w"],      [phi*180/np.pi for (r, phi) in y_true], linewidth=4, label="JCAPL")
    plt.semilogx(pred_df["w"], [phi*180/np.pi for (r, phi) in y_pred], linewidth=2, label="pred")
    plt.legend()
    plt.grid(True,linestyle='--')
    plt.xlabel("$\omega$")
    plt.xscale('log')
    plt.title("Phase")
        
    plt.tight_layout()
    if path_name is not None:
        plt.savefig('results/{0}/fig/gain_phase.png'.format(path_name))
    if mode=="humain":
        plt.show()
    else:
        plt.close()




def analyse(path_name=None):
    if path_name is not None:
        directory = 'results/{0}/params.csv'.format(path_name)
    else:
        directory = min(glob.glob(os.path.join("results", '*/')), key=os.path.getmtime)
        directory = os.path.join(directory, "params.csv")
        
    if Config.REWARD_TYPE=="re_coef":
        label="R² coef"
    elif Config.REWARD_TYPE=="mse":
        label="MSE"
    
    # coefficient R² et reward selon les étapes d'entraînement
    df = pd.read_csv(directory)
    
    print(df)
    
    plt.figure(figsize=(8, 4))
    
    plt.xlabel("step")
    plt.grid(True, linestyle='--')
    plt.scatter(df["current_step"], df["r2_coef"], label=label)
    plt.scatter(df["current_step"], df["reward"], label="reward")
    plt.legend()
    
    plt.figure(figsize=(2*4, 4))
    
    plt.subplot(1, 2, 1)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.grid(True, linestyle='--')
    plt.scatter(df["Re(mu[0])"], df["Im(mu[0])"])
    
    plt.subplot(1, 2, 2)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.grid(True, linestyle='--')
    plt.bar(df["current_step"], df["done"], width=1.)
    
    plt.tight_layout()
    plt.show()