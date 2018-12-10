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
void mean(int n, float *x, float *y, int loop)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride){
    float a = x[i];
    float b = y[i];
    y[i] = b + (a - b)/((float)loop);
  }
}

int main(void)
{
  auto begin = std::chrono::high_resolution_clock::now();

  int N = 1024*1024;
  float *img, *meanImg;
  Mat imgMat;
  int avgDur = 0;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&img, N*sizeof(float));
  cudaMallocManaged(&meanImg, N*sizeof(float));

  //open images
  std::string line;
  std::ifstream myfile ("images.txt");
  if (myfile.is_open()){
    int loops = 1;
    while ( getline (myfile,line) ){
      std::cout << line << '\n';
      std::cout << loops << '\n';
      imgMat = imread( line, IMREAD_COLOR );

      //initialize arrays to be passed to be passed
      int i = 0;
      for(int y = 0; y<imgMat.rows; y++){
        for(int x = 0; x<imgMat.cols; x++){
          img[i] = imgMat.at<Vec3b>(Point(x,y))[0];
          if(loops==1){
           meanImg[i] = 0.0f;
          }
          i++;
        }
      }

      auto start = std::chrono::high_resolution_clock::now();

    	int blockSize = 512;
    	int numBlocks = (N + blockSize - 1) / blockSize;
    	mean<<<numBlocks, blockSize>>>(N, img, meanImg,loops);

      // Wait for GPU to finish before accessing on host
      cudaDeviceSynchronize();

      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
      std::cout << "Time taken by function: " << duration.count() << " microseconds\n";
      avgDur = avgDur + (duration.count()-avgDur)/loops;
      std::cout << "Avg Duration: " << avgDur << " microseconds\n";

      loops++;
    }

    //Reconstruct Mat
    int i = 0;
    for(int y = 0; y<imgMat.rows; y++){
      for(int x = 0; x<imgMat.cols; x++){
      int c = meanImg[i];
      imgMat.at<Vec3b>(Point(x,y)) = Vec3b(c,c,c);
      i++;
      }
    }
    // Free memory
    cudaFree(img);
    cudaFree(meanImg);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin); 
    std::cout << "Total Time: " << duration.count() << " microseconds\n";

    namedWindow( line, WINDOW_AUTOSIZE );
    imwrite("./mean.jpg",imgMat);
    imshow( line, imgMat );
    waitKey(0);
  }
  
  return 0;
}
//nvcc -o multiMean multiMean.cu `pkg-config opencv --cflags --libs` -std=c++11