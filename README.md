# Fractal Image Compression with Parallel Processing

## Overview
This repository contains a Python implementation of fractal image compression, a technique that leverages self-similarity within an image to achieve efficient compression. One notable feature is the integration of parallel processing to enhance the compression performance significantly.

## Fractal Image Compression
Fractal image compression is a method that exploits the recursive nature of fractals found within an image. The basic idea is to represent an image using a set of transformations that can be applied to smaller blocks within the image. These transformations, when iteratively applied, approximate the original image. The compression process involves finding the best set of transformations for each block in the image.

## Parallel Processing
The implementation includes parallel processing using the multiprocessing module. The parallelization is applied to the two main computational tasks: generating transformed blocks and compressing the image. This results in a significant reduction in processing time, making the compression process more efficient.
