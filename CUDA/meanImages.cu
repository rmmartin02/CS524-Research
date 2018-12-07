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
void mean(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = (x[i] + y[i])/2.0f;
}

int main(void)
{
  int N = 1024*1024;
  float *img, *meanImg;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&img, N*sizeof(float));
  cudaMallocManaged(&meanImg, N*sizeof(float));

  Mat imgMat;
  Mat meanImgMat;

  //open images
  std::string line;
  std::ifstream myfile ("images.txt");
  if (myfile.is_open()){
    int i = 0;
    while ( getline (myfile,line) && i<2 ){
      std::cout << line << '\n';
      std::cout << i << '\n';
      if (i==1){
        imgMat = imread( line, IMREAD_COLOR );
      }
      else{
        meanImgMat = imread( line, IMREAD_COLOR );
      }
      i++;
    }
  }
  printf("%d %d\n",meanImgMat.rows,meanImgMat.cols);
  //initialize arrays to be passed to be passed
  int i = 0;
  for(int y = 0; y<meanImgMat.rows; y++){
    for(int x = 0; x<meanImgMat.cols; x++){
      img[i] = imgMat.at<Vec3b>(Point(x,y))[0];
      meanImg[i] = meanImgMat.at<Vec3b>(Point(x,y))[0];
      i++;
    }
  }
  auto start = std::chrono::high_resolution_clock::now();
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	mean<<<numBlocks, blockSize>>>(N, img, meanImg);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
  std::cout << "Time taken by function: " << duration.count() << " microseconds\n";

  //Reconstruct Mat
  i = 0;
  for(int y = 0; y<meanImgMat.rows; y++){
    for(int x = 0; x<meanImgMat.cols; x++){
      int c = meanImg[i];
      meanImgMat.at<Vec3b>(Point(x,y)) = Vec3b(c,c,c);
      i++;
    }
  }
  namedWindow( line, WINDOW_AUTOSIZE );
  imwrite("./mean.jpg",meanImgMat);
  imshow( line, meanImgMat );
  waitKey(0);

  // Free memory
  cudaFree(img);
  cudaFree(meanImg);
  
  return 0;
}
//nvcc -o meanImages meanImages.cpp `pkg-config opencv --cflags --libs` -std=c++11