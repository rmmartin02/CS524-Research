#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
using namespace cv;
using namespace std;


// Kernel function to add the elements of two arrays
__global__
void filterMean(int n, int S, int C, int tol, float *imgR, float *meanImg, int loop)
{

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n/3; i += stride){
    int x = (i%N)%S;
    int y = (i%N)/S;
    int r = imgR[i];
    int g = imgR[i+A];
    int b = imgR[i+2*A];
    int a = y*S+x;
    //coord=i
    if(x<(S/2)){
      //search right
      if(y<(S/2)){
      //search down
        while(r==0 || (b-r)>tol || (r-b)<-tol){
          x++;
          y++;
          a = y*S+x;
          r = imgR[a];
          g = imgR[a+A];
          b = imgR[a+2*A];
        }
      }
      else{
        //search up
        while(r==0 || (b-r)>tol || (r-b)<-tol){
          x++;
          y--;
          a = y*S+x;
          r = imgR[a];
          g = imgR[a+A];
          b = imgR[a+2*A];
        }
      }
    }
    else{
      //search left
      if(y<(S/2)){
      //search down
        while(r==0 || (b-r)>tol || (r-b)<-tol){
          x--;
          y++;
          a = y*S+x;
          r = imgR[a];
          g = imgR[a+A];
          b = imgR[a+2*A];
        }
      }
      else{
        while(r==0 || (b-r)>tol || (r-b)<-tol){
          x--;
          y--;
          a = y*S+x;
          r = imgR[a];
          g = imgR[a+A];
          b = imgR[a+2*A];
        }
      }
    }
    meanImg[i] += r;
  }
}

int main(void)
{
  const int NUM_COLORS = 3; //number of colors
  const int IMG_WIDTH = 1024;
  const int IMG_DIM = IMG_WIDTH*IMG_WIDTH;
  const int IMG_DIM_COLORS = IMG_DIM*NUM_COLORS;
  const int TOL = 30;

  int NUM_FILES = 0;
  string line;
  ifstream myfile ("images.txt");
  if (myfile.is_open()){
    while ( getline (myfile,line) ){
      NUM_FILES++;
    }
  }

  auto begin = std::chrono::high_resolution_clock::now();
  float *meanImg;
  Mat imgMat;
  float avgDur = 0.0f;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&meanImg, IMG_DIM*sizeof(float));

  //open images
  string line;
  ifstream myfile ("images.txt");
  if (myfile.is_open()){
    int num_batchs_proc = 0;
    while ( getline (myfile,line) ){
      int *batchMean;
      cudaMallocManaged(&batchMean, IMG_DIM*sizeof(int));

      char *batch;
      size_t curAvailMem;
      size_t totalMem;

      cudaMemGetInfo(&curAvailMem, &totalMem);
      int batchSize = (curAvailMem/IMG_DIM)-1;
      if(batchSize+num_imgs_proc>NUM_FILES){
        batchSize = NUM_FILES-num_imgs_proc;
      }
      cudaMallocManaged(&batch, batchSize*IMG_DIM_COLORS*sizeof(char));

      //initilize batch
      int i = 0;
      for(int b =0; b<batchSize; b++){
        for(int y = 0; y<imgMat.rows; y++){
          for(int x = 0; x<imgMat.cols; x++){
            Vec3b color = imgMat.at<Vec3b>(Point(x,y));
            batch[i*b] = color[0];
            batch[i*B+IMG_DIM] = color[1];
            batch[i*B+2*IMG_DIM] = color[2];
            if(b==0){
              batchMean[i] = 0;
            }
            i++;
          }
        }
      }

      int blockSize = 512;
      int numBlocks = (N + blockSize - 1) / blockSize;
      filterMean<<<numBlocks, blockSize>>>(IMG_DIM_COLORS, IMG_DIM, NUM_COLORS, TOL, img, meanImg, num_imgs_proc);

      // Wait for GPU to finish before accessing on host
      cudaDeviceSynchronize();
      
    }
  }

  return 0;
}
//export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
//nvcc -o FilterAndMean FilterAndMean.cu `pkg-config opencv --cflags --libs` -std=c++11