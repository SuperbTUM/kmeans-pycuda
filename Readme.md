**Archive Only At Present**

## Update Log

Oct 11, 2021 I found out I made a huge mistake! I didn't use synchronized method in each kernel, and that led to a tremendous collapse in evaluation indexes! Though there are a lot to do next, I decide to archive this repository for a while because I have to complete my master's degree.

Oct 7, 2021 Code review, v1.2 released, with more than 100% speed up (16s -> less than 6s). I managed to reduce memeory transfer frequency. The square of datapoints should be calculated for only once! 

Feb 13, 2021 Confused by implementation in sklearn, since it's incredibly fast while maintaining high accuracy.

Sep 7, 2020  Minor optimization on dis_computation function

Sep 6, 2020  Optimization on reduction algorithm

## Introduction

The goal of this individual work is to design and implement a general acceleration algorithm for k-means clustering with the assistance of GPU. The work is an algorithm that integrates global discriminator, cuBLAS matrix multiplication, ThrsutRTC, and basic feature engineering (PCA). To test the algorithm, I set traditional k-means algorithm (Lloyd, 1982) as well as [triangle inequality algorithm](https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf) (Elkan, 2006) as references. I leveraged Adjust Rand Index and Adjust Normalization Index as accuracy evaluation functions and linear speedup as speed evaluation function.

Once we have the methodology, we need to write a demo to test it quantitatively. The following table shows basic information of the dataset for experiment. The URL of the dataset is: https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

| name                      | parameter                |
| ------------------------- | ------------------------ |
| dataset                   | kdd-cup 1999 (corrected) |
| feature number            | 41                       |
| original cluster number   | 23                       |
| simplified cluster number | 5                        |

## Quick start

Before the quick start, make sure you do have a CUDA in your computer. Use the command `cat /usr/local/cuda/version.txt` to check CUDA version. Read the Docs of your GPU to determine the maximum threads in your SP. The number of maximum threads per block is 1024. For a quick start, you can merely alter the file path in `kmeans_v1_2rc.ipynb` whenever you want to test the dataset (commonly in csv or txt format) and set your desired cluster number, then click 'run'. 

Moreover, I design a simple GUI. However, it is a naïve one. For those who want to embedd the algorithm into a software, you can try to put more effort in it.

## Next Step

I intend to employ a global discriminator to reduce calculation amount, but this has been stopped in the aspect of CPU. The reason why I couldn't integrate this into GPU acceleration is that I should use sparse matrix for distance calculation. Then cuBLAS needs to be replaced by cuSPARSE. Unluckily, I couldn't install pyculib successfully though I had the idea of how to use cuSPARSE to make the acceleration algorithm into the next level. I am faced with the following attribute error: `undefined symbol: cusparseCaxpyi_v2`. I had a demo about level-3 function usage uploaded and you could use that to test whether cuSPARSE is functioning normally.

