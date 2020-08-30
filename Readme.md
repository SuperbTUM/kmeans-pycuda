## Introduction

This is my graduation project. The goal of this individual work is to design and implement a general acceleration algorithm for k-means clustering with the assistance of GPU. I'd like to name it as hybrid method. Frankly speaking, it is a kind of algorithm that integrates all possible acceleration modules that I can think of, including PCA as dataset pretreatment. To test the algorithm, I set traditional k-means algorithm (Lloyd, 1982) as well as triangle inequality algorithm (Elkan, 2006) as references. I leveraged Adjust Rand Index and Adjust Normalization Index as accuracy functions and linear speedup as algorithm speed evaluation function.

Once we have the methodology, we need to write a demo to test it quantitatively. The following table shows basic information of the dataset for experiment. The URL for dataset download is: https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

| name                      | parameter                |
| ------------------------- | ------------------------ |
| dataset                   | kdd-cup 1999 (corrected) |
| feature number            | 41                       |
| original cluster number   | 23                       |
| simplified cluster number | 5                        |

## Quick start

Before the quick start, make sure you do have CUDA in your computer. Use the command `cat /usr/local/cuda/version.txt` to check CUDA version. Read the Docs of your GPU to determine the maximum threads in your SP. In my case, the number of maximum threads per block is 1024. But generally this number would be 512. For a quick start, you can merely alter the file path in final_v1.0.ipynb whenever you want to test the dataset (commonly in csv or txt format) and set your desired cluster number, then click 'run'. 

Moreover, I design a simple GUI. However, it is a naïve one. For those who want to embedd the algorithm into a software, you can try to put more effort on it.

## Next Step

I intend to employ a global discriminator to reduce calculation amount, but this has been stopped in the aspect of CPU. The reason why I couldn't integrate this into GPU acceleration is that I should use sparse matrix for distance calculation. Then cuBLAS needs to be replaced by cuSPARSE. Unluckily, I couldn't install pyculib successfully though I had the idea of how to use cuSPARSE to make the acceleration algorithm into the next level. I had a demo about level-3 function usage uploaded and you could use that to test whether cuSPARSE is functioning normally.

## Something else I want to say

For us Chinese learners, if we are asked to write some code, we are likely to turn to the Internet (mostly CSDN for startup learners) for help as our first step. I won't contend this is inappropriate because this can help to complete our tasks quicker, but learning is not always copying from others without understanding what the code writer is thinking at that moment and assessing the correctness of the referenced code. This is not to demand everyone to become an expert, this is to encourage everyone to become critical, to be vigilant towards latent bugs, since certain bugs may lead to fatal errors under massive and exhaustive tests.

## Further information

If you would like to know more detail about the structure of the algorithm or if you have any other question, feel free to leave a comment or contact me via e-mail: hmz.superb@gmail.com
