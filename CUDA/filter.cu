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
void filter(int n, int S, int C, int tol, float *imgW, float *imgR)
{
  int A = S*S;
  int N = A*C;
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
    imgW[i] = r;
    imgW[i+A] = g;
    imgW[i+2*A] = b;
  }
}

int main(void)
{
  int C = 3; //number of colors
  int S = 1024;
  int A = S*S;
  int N = A*C;
  int tol = 60;


  auto begin = std::chrono::high_resolution_clock::now();
  float *img, *readImg;
  Mat imgMat;
  int avgDur = 0;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&img, N*sizeof(float));
  cudaMallocManaged(&readImg, N*sizeof(float));

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
          img[i+A] = imgMat.at<Vec3b>(Point(x,y))[1];
          img[i+2*A] = imgMat.at<Vec3b>(Point(x,y))[2];
          readImg[i] = imgMat.at<Vec3b>(Point(x,y))[0];
          readImg[i+A] = imgMat.at<Vec3b>(Point(x,y))[1];
          readImg[i+2*A] = imgMat.at<Vec3b>(Point(x,y))[2];
          i++;
        }
      }

      auto start = std::chrono::high_resolution_clock::now();

    	int blockSize = 512;
    	int numBlocks = (N + blockSize - 1) / blockSize;
    	mean<<<numBlocks, blockSize>>>(N, S, C, tol, img, readImg);

      // Wait for GPU to finish before accessing on host
      cudaDeviceSynchronize();

      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
      std::cout << "Time taken by function: " << duration.count() << " microseconds\n";
      avgDur = avgDur + (duration.count()-avgDur)/loops;
      std::cout << "Avg Duration: " << avgDur << " microseconds\n";

      //Reconstruct Mat
      int i = 0;
      for(int y = 0; y<imgMat.rows; y++){
        for(int x = 0; x<imgMat.cols; x++){
        int c = readImg[i];
        imgMat.at<Vec3b>(Point(x,y)) = Vec3b(c,c,c);
        i++;
      }

         namedWindow( line, WINDOW_AUTOSIZE );
      imwrite("./mean.jpg",imgMat);
      imshow( line, imgMat );
      waitKey(0);

      loops++;
    }

    }
    // Free memory
    cudaFree(img);
    cudaFree(readImg);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin); 
    std::cout << "Total Time: " << duration.count() << " microseconds\n";

  }
  
  return 0;
}
//nvcc -o multiMean multiMean.cu `pkg-config opencv --cflags --libs` -std=c++11