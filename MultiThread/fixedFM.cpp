#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
using namespace cv;
using namespace std;

const int C = 3; //number of colors
const int S = 1024;
const int A = S*S;
const int N = A*C;
const int tol = 30;
int img [N];
unsigned long meanImg [N];


void filterMean(int i, int loop)
{
  int x = (i%N)%S;
  int y = (i%N)/S;
  int r = img[i];
  int g = img[i+A];
  int b = img[i+2*A];
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
        r = img[a];
        g = img[a+A];
        b = img[a+2*A];
      }
    }
    else{
      //search up
      while(r==0 || (b-r)>tol || (r-b)<-tol){
        x++;
        y--;
        a = y*S+x;
        r = img[a];
        g = img[a+A];
        b = img[a+2*A];
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
        r = img[a];
        g = img[a+A];
        b = img[a+2*A];
      }
    }
    else{
      while(r==0 || (b-r)>tol || (r-b)<-tol){
        x--;
        y--;
        a = y*S+x;
        r = img[a];
        g = img[a+A];
        b = img[a+2*A];
      }
    }
  }
  meanImg[i] += r;
}


int main( int argc, char** argv )
{

  int numT = atoi(argv[1]); //number of threads
  std::thread threads[numT];

  auto begin = std::chrono::high_resolution_clock::now();
  Mat imgMat;
  float avgDur = 0.0f;

  //open images
  std::string line;
  std::ifstream myfile ("images.txt");
  if (myfile.is_open()){
    int loops = 1;
    while ( getline (myfile,line) ){
      std::cout << line << '\n';
      //std::cout << loops << '\n';
      imgMat = imread( line, IMREAD_COLOR );

      //initialize arrays to be passed to be passed
      int i = 0;
      for(int y = 0; y<imgMat.rows; y++){
        for(int x = 0; x<imgMat.cols; x++){
          img[i] = imgMat.at<Vec3b>(Point(x,y))[0];
          img[i+A] = imgMat.at<Vec3b>(Point(x,y))[1];
          img[i+2*A] = imgMat.at<Vec3b>(Point(x,y))[2];
          if(loops==1){
           meanImg[i] = 0;
          }
          i++;
        }
      }
      //printf("%f %f %f",img[N/2],img[N/2+A],img[N/2+A*2]);

      auto start = std::chrono::high_resolution_clock::now();

      //maybe use cache blocking?
      for(int i=0; i<A/numT; i++){
        for(int t = 0; t<numT; t++){
          threads[t]=std::thread(filterMean, i+(t*A/numT), loops);
        }
        for(int t = 0; t<numT; t++){
          threads[t].join();
        }
      }

      
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
      std::cout << "Time taken by function: " << duration.count() << " milliseconds\n";
      avgDur = avgDur + (duration.count()-avgDur)/loops;
      std::cout << "Avg Duration: " << avgDur << " microseconds\n";
      
      /*
      i = 0;
      for(int y = 0; y<imgMat.rows; y++){
        for(int x = 0; x<imgMat.cols; x++){
          int c = meanImg[i];
          //printf("Before: %d %d %d\n",imgMat.at<Vec3b>(Point(x,y))[0],imgMat.at<Vec3b>(Point(x,y))[1],imgMat.at<Vec3b>(Point(x,y))[2]);
          imgMat.at<Vec3b>(Point(x,y)) = Vec3b(c,c,c);
          //printf("After:  %d %d %d\n",imgMat.at<Vec3b>(Point(x,y))[0],imgMat.at<Vec3b>(Point(x,y))[1],imgMat.at<Vec3b>(Point(x,y))[2]);
          i++;
        }
      }
      if (loops>2){
        namedWindow( line, WINDOW_AUTOSIZE );
        imshow( line, imgMat );
        waitKey(0);
      }
      */

      loops++;
    }

    for(int i=0; i<N; i++){
      meanImg[i]/=loops;
    }

    //Reconstruct Mat
    int i = 0;
    for(int y = 0; y<imgMat.rows; y++){
      for(int x = 0; x<imgMat.cols; x++){
        int c = meanImg[i];
        //printf("Before: %d %d %d\n",imgMat.at<Vec3b>(Point(x,y))[0],imgMat.at<Vec3b>(Point(x,y))[1],imgMat.at<Vec3b>(Point(x,y))[2]);
        imgMat.at<Vec3b>(Point(x,y)) = Vec3b(c,c,c);
        //printf("After:  %d %d %d\n",imgMat.at<Vec3b>(Point(x,y))[0],imgMat.at<Vec3b>(Point(x,y))[1],imgMat.at<Vec3b>(Point(x,y))[2]);
        i++;
      }
    }

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
//g++ -o fixedFM fixedFM.cpp `pkg-config opencv --cflags --libs` -std=c++11 -pthread