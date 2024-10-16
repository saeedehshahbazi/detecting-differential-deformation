# Detecting Differential Deformation

This is a Python software that can detect at-risk buildings and classify them based on gradient intensity. It helps in identifying structures that are susceptible to differential deformation.

## Features

- Detects at-risk buildings from PSI (Persistent Scatterer Interferometry) data and building polygons.
- Classifies buildings based on deformation gradient intensity.
- Supports automated execution for multiple datasets.

## Requirements

- **Input Files:**
  - PSI Data
  - Building Polygons
- Both input files must have the same **CRS (Coordinate Reference System): `EPSG:3035`.**

## How to Use

To run the program with specific datasets, use the following command:

```bash
python gradientCalculationInsidePoints_sepPoly.py PS_point buildingsPolygonData
