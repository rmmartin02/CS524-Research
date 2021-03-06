#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <future>
#include <chrono>
using namespace cv;

int tol = 60;
Mat meanImage;
Mat image;

Vec3b interPolat(int x, int y){
	//printf("Called interPolat %d %d\n", x, y);
	Vec3b color = image.at<Vec3b>(Point(x,y));
	if(x<(image.cols/2)){
		//search right
		if(y<(image.rows/2)){
		//search down
			while(color[0]==0 || (color[2]-color[0])>tol || (color[2]-color[0])<-tol){
				x++;
				y++;
				color = image.at<Vec3b>(Point(x,y));
			}
		}
		else{
			//search up
			while(color[0]==0 || (color[2]-color[0])>tol || (color[2]-color[0])<-tol){
				x++;
				y--;
				color = image.at<Vec3b>(Point(x,y));
			}
		}
	}
	else{
		//search left
		if(y<(image.rows/2)){
		//search down
			while(color[0]==0 || (color[2]-color[0])>tol || (color[2]-color[0])<-tol){
				x--;
				y++;
				color = image.at<Vec3b>(Point(x,y));
			}
		}
		else{
			//search up
			while(color[0]==0 || (color[2]-color[0])>tol || (color[2]-color[0])<-tol){
				x--;
				y--;
				color = image.at<Vec3b>(Point(x,y));
			}
		}
	}
	return color;
}

void filterMean(int y, int num){
	for(int x=0; x<image.cols; x++){
		// get pixel
		Vec3b color = image.at<Vec3b>(Point(x,y));
		
		if ((color[2]-color[0])>tol || (color[2]-color[0])<-tol){
			color = interPolat(x,y);
		}

		// set pixel
		//fixed.at<Vec3b>(Point(x,y)) = color;
		Vec3b meanColor = meanImage.at<Vec3b>(Point(x,y));
		for(int i=0;i<3;i++){
			meanColor[i] = meanColor[i] + (color[i]-meanColor[i])/num;
		}
		meanImage.at<Vec3b>(Point(x,y)) = meanColor;
		//meanImage = meanImage+(1/i)*(fixed[:,:]-meanImage)
	}
}

int main( int argc, char** argv )
{
	std::string line;
	std::ifstream myfile ("images.txt");
	int N = 8; //number of threads
	std::thread threads[N];
	int num = 1;
	int avgDur = 0;
	//Mat meanImage;
	if (myfile.is_open()){
		while ( getline (myfile,line) ){
			std::cout << line << '\n';
			std::cout << num << '\n';
			//Mat image;
			//Mat fixed;
			image = imread( line, IMREAD_COLOR );
			image = image(Rect(256,256,512,512));
			//fixed = imread(	line, IMREAD_COLOR );
			if(!image.data ){
				//printf( " No image data \n " );
				return -1;
			}
			//left, up, width, height
			//1024x1024
			auto start = std::chrono::high_resolution_clock::now();
			if (num>1){
				//fixed = fixed(Rect(256,256,512,512));
				for(int y=0;y<image.rows/N; y++){
					for(int i=0; i<N; i++){
						int val = y+(i*(image.rows/N));
						threads[i]=std::thread(filterMean,val,num);
					}
					for(int i = 0; i<N; i++){
						threads[i].join();
					}
				}
			}
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
		    std::cout << "Time taken by function: " << duration.count() << " microseconds\n";
		    avgDur = avgDur + (duration.count()-avgDur)/num;
		    std::cout << "Avg Duration: " << avgDur << " microseconds\n";
			if(num == 1){
				meanImage = image;
			}
			num++;
		}
		myfile.close();
		namedWindow( line, WINDOW_AUTOSIZE );
		imshow( line, meanImage );
		waitKey(0);
	}

	else std::cout << "Unable to open file";
	/*
	cvtColor( fixed, fixed, COLOR_BGR2GRAY );
	namedWindow( imageName, WINDOW_AUTOSIZE );
	imshow( imageName, fixed );
	*/
	return 0;
}
//g++ -o FilterAndMean FilterAndMean.cpp `pkg-config opencv --cflags --libs` -std=c++11 -pthread