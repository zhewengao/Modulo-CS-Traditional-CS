# Image Compression With Modulo Compressed Sensing

The compressed sensing theory generally assumes that the measurements have infinite precision which is not practical. The problem is typically tackled using quantization, but it suffers from clipping or saturation effect when the measurement exceeds the dynamic range of the analog-to-digital converter (ADC). Recently, a new modulo-based architecture was introduced for ADC. This architecture counters the clipping effect by folding signals that extend beyond the range back into the dynamic range of ADCs through modulo arithmetic. The goal of this project is to solve the compressed sensing problem when the measurements are obtained by modulo operation and evaluate the performance. 

# Objective of this project
* implement the Modulo CS algorithm, and compare the recovering performance with traditional CS algorithms (e.g., MP, OMP, CoSaMP, IHT).
* The "Data Compression" file contains four ".ipynb" files for traditional CS algorithms (MP, OMP, CoSaMP, IHT) and one ".m" file for the modulo CS.
* To run the "ModuloCS.m" file, make sure the add the _intlinprog_ library in MATLAB.
