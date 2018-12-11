#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
using namespace cv;
using namespace std;

__global__
void combineMeans(int *batchMean, float *meanImg, int batchSize, int dim, int num){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < dim; i += stride){
    int a = batchMean[i]/batchSize;
    int z = meanImg[i];
    meanImg[i] = z + (a - z)/((float)num);
  }
}

// Kernel function to add the elements of two arrays
__global__
void filterMean(int *imgR, int *meanImg, int batchSize, int width, int dim, int num_colors, int tol)
{
  int A = batchSize*dim;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < batchSize*dim; i += stride){
    int bn = i/(dim*num_colors);//batch number
    int x = i%width;
    int y = (i%dim)/width;
    int r = imgR[i];
    int g = imgR[i+A];
    int b = imgR[i+2*A];
    int a = (y*width+x)+(dim*bn);
    //coord=i
    //printf("before %d\n",r);
    if(x<(width/2)){
      //search right
      if(y<(width/2)){
      //search down
        while(r==0 || (b-r)>tol || (r-b)<-tol){
          x++;
          y++;
          a = (y*width+x)+(dim*bn);
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
          a = (y*width+x)+(dim*bn);
          r = imgR[a];
          g = imgR[a+A];
          b = imgR[a+2*A];
        }
      }
    }
    else{
      //search left
      if(y<(width/2)){
      //search down
        while(r==0 || (b-r)>tol || (r-b)<-tol){
          x--;
          y++;
          a = (y*width+x)+(dim*bn);
          r = imgR[a];
          g = imgR[a+A];
          b = imgR[a+2*A];
        }
      }
      else{
        while(r==0 || (b-r)>tol || (r-b)<-tol){
          x--;
          y--;
          a = (y*width+x)+(dim*bn);
          r = imgR[a];
          g = imgR[a+A];
          b = imgR[a+2*A];
        }
      }
    }
    //printf("after %d\n",r);
    //printf("mean before %d\n",meanImg[i]);
    meanImg[i] += (int) r;
    //printf("mean after %d\n",meanImg[i]);
  }
}

int main(void)
{
  const int NUM_COLORS = 3; //number of colors
  const int IMG_WIDTH = 1024;
  const int IMG_DIM = IMG_WIDTH*IMG_WIDTH;
  const int IMG_DIM_COLORS = IMG_DIM*NUM_COLORS;
  const int TOL = 30;


  int device_count;
  size_t max_mem = 0;
  int best_device = 0;
  cudaGetDeviceCount(&device_count);
  for(int i = 0; i<device_count; i++){
    size_t curAvailMem, totalMem;
    cudaSetDevice(i);
    cudaMemGetInfo(&curAvailMem, &totalMem);
    printf("%zd %zd %zd\n",i,curAvailMem, totalMem);
    if(curAvailMem>max_mem){
      max_mem = curAvailMem;
      best_device = i;
    }
  }

  printf("Best device is %d with %zd free memory\n",best_device,max_mem);
  cudaSetDevice(best_device);

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
  int *batchMean;
  cudaMallocManaged(&batchMean, IMG_DIM*sizeof(int));
  cudaMallocManaged(&meanImg, IMG_DIM*sizeof(float));

  //open images
  ifstream myfile2 ("images.txt");
  if (myfile2.is_open()){
    int num_batchs_proc = 0;
    int num_imgs_proc = 0;
    while ( num_imgs_proc < NUM_FILES ){

      int *batch;
      size_t curAvailMem;
      size_t totalMem;

      cudaMemGetInfo(&curAvailMem, &totalMem);

      int batchSize = (curAvailMem-IMG_DIM*sizeof(int)+IMG_DIM*sizeof(float))/(8*IMG_DIM_COLORS*sizeof(int))-1;

      printf("%d %zd %zd\n",batchSize, curAvailMem, totalMem);
      if(batchSize+num_imgs_proc>NUM_FILES){
        batchSize = NUM_FILES-num_imgs_proc;
      }

      batchSize = 10;

      printf("%d %zd %zd\n",batchSize, curAvailMem, totalMem);
      cudaMallocManaged(&batch, batchSize*IMG_DIM_COLORS*sizeof(int));

      cudaMemGetInfo(&curAvailMem, &totalMem);
      printf("%d %zd %zd\n",batchSize, curAvailMem, totalMem);

      printf("initilize batch\n");
      //initilize batch
      int i = 0;
      for(int b =0; b<batchSize; b++){
        getline(myfile2, line);
        imgMat =  imread( line, IMREAD_COLOR );
        printf("%d %d\n",b,batchSize);
        //printf("%zd %zd\n",i,i+2*IMG_DIM*b);
        for(int y = 0; y<imgMat.rows; y++){
          for(int x = 0; x<imgMat.cols; x++){
            //segfaulting somwhere between batch[1796210688] and batch[1799356416]
            Vec3b color = imgMat.at<Vec3b>(Point(x,y));
            batch[i] = (int) color[0];
            batch[i+IMG_DIM*b] = (int) color[1];
            batch[i+2*IMG_DIM*b] = (int) color[2];
            if(batch[i]<0 ||  batch[i+IMG_DIM*b]<0 || batch[i+2*IMG_DIM*b]<0){
              printf("%d %d %d %d %d",i,b,batch[i],batch[i+IMG_DIM*b],batch[i+2*IMG_DIM*b]);
            }
            if(b==0){
              batchMean[i] = 0;
            }
            i++;
          }
        }
      }

      printf("Batch Sample %d\n", batch[(batchSize/2)*(IMG_DIM_COLORS/2)]);

      int blockSize = 512;
      int numBlocks = (IMG_DIM + blockSize - 1) / blockSize;
      printf("%d\n", batchSize);
      filterMean<<<numBlocks, blockSize>>>(batch, batchMean, batchSize, IMG_WIDTH, IMG_DIM, NUM_COLORS, TOL);
      cudaDeviceSynchronize();

      cudaFree(batch);
      printf("Mean Sample %d\n", batchMean[IMG_DIM/2]);
      num_batchs_proc++;

      printf("%d\n", num_batchs_proc);
      combineMeans<<<numBlocks,blockSize>>>(batchMean,meanImg,batchSize, IMG_DIM, num_batchs_proc);
      cudaDeviceSynchronize();

      num_imgs_proc+= batchSize;
      printf("%d\n",num_imgs_proc);
      break;
    }
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

  namedWindow( line, WINDOW_AUTOSIZE );
  imwrite("./mean.jpg",imgMat);
  imshow( line, imgMat );
  waitKey(0);

  return 0;
}
//export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
//nvcc -o batches batches.cu `pkg-config opencv --cflags --libs` -std=c++11