#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
using namespace cv;

int tol = 60;

Vec3b interPolat(Mat image, int x, int y){
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

int main( int argc, char** argv )
{
	std::string line;
	std::ifstream myfile ("images.txt");
	int num = 1;
	int avgDur = 0;
	Mat meanImage;
	if (myfile.is_open()){
		while ( getline (myfile,line) ){
			std::cout << line << '\n';
			std::cout << num << '\n';
			Mat image;
			Mat fixed;
			image = imread( line, IMREAD_COLOR );
			fixed = imread(	line, IMREAD_COLOR );
			if(!image.data ){
				printf( " No image data \n " );
				return -1;
			}
			//left, up, width, height
			//1024x1024
			//image = image(Rect(256,256,512,512));
			//fixed = fixed(Rect(256,256,512,512));
			auto start = std::chrono::high_resolution_clock::now();
			for(int y=1;y<image.rows-1;y++){
				for(int x=1;x<image.cols-1;x++){
					// get pixel
					Vec3b color = image.at<Vec3b>(Point(x,y));
					
					//if ((color[2]-color[0])>tol || (color[2]-color[0])<-tol){
					//	color = interPolat(image,x,y);
					//}

					// set pixel
					//fixed.at<Vec3b>(Point(x,y)) = color;
					if (num>1){
						Vec3b meanColor = meanImage.at<Vec3b>(Point(x,y));
						for(int i=0;i<3;i++){
							meanColor[i] = meanColor[i] + (color[i]-meanColor[i])/num;
						}
						meanImage.at<Vec3b>(Point(x,y)) = meanColor;
						//meanImage = meanImage+(1/i)*(fixed[:,:]-meanImage)
					}
				}
			}
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
		    std::cout << "Time taken by function: " << duration.count() << " microseconds\n";
		    avgDur = avgDur + (duration.count()-avgDur)/num;
		    std::cout << "Avg Duration: " << avgDur << " microseconds\n";
			if(num==1){
				meanImage = fixed;
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
//g++ -o FilterAndMean FilterAndMean.cpp `pkg-config opencv --cflags --libs` -std=c++11