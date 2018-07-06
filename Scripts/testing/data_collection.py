#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 14:52:55 2018

@author: asewnath
"""

#Script to gather data for sequential model experimentation


"""
Data Required:
    Both models require an input vector of month number, regional sea ice
    concentrations, and average ice thickness.

"""

startYr    = 1980
forecastYr = 2015

for year in range(1980, 2015):
    for month in range(1, 13):
        sample = []
        #append month to list
        sample.append(month)
        
