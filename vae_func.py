#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:39:38 2020

@author: sam
"""

class trainedVAE():
    def __init__(self,encoder,decoder, history=None):
        self.encoder = encoder
        self.decoder = decoder
        self.history = history
    
    def embed(self,data):
        #takes img data and puts in latent space
        return self.encoder.predict(data)
    
    def reconstruct(self, data):
        #takes latent data and put in img space
        return self.decoder.predict(data)
        