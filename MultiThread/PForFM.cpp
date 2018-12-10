#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <future>
#include <chrono>
using namespace cv;

int tol = 30;
Mat meanImage;
Mat image;
int num = 1;

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

class Body : public cv::ParallelLoopBody
{
private:
	int y;

public:
	Body(int val)
	: y(val){}

    void operator ()(const cv::Range& range) const
    {
        for (int x = range.start; x < range.end; ++x){
        	//meanColor = meanColor[i] + (color[i]-meanColor[i])/num;
        	Vec3b color = image.at<Vec3b>(Point(x,y));
        	//printf("before %d\n",color[0]);
        	if ((color[2]-color[0])>tol || (color[2]-color[0])<-tol){
				color = interPolat(x,y);
			}
			//printf("after %d\n",color[0]);
        	int meanColor = meanImage.at<uchar>(y,x);
        	meanImage.at<uchar>(y,x) = meanColor+(color[0]-meanColor)/num;
        }
    }
};


int main( int argc, char** argv )
{
	auto begin = std::chrono::high_resolution_clock::now();
	std::string line;
	std::ifstream myfile ("images.txt");
	setNumThreads(atoi(argv[1]));
	printf("Num Threads: %d\n",getNumThreads());
	int avgDur = 0;
	//Mat meanImage;
	if (myfile.is_open()){

		while ( getline (myfile,line) ){
			//std::cout << line << '\n';
			//std::cout << num << '\n';
			//Mat image;
			//Mat fixed;
			image = imread( line, IMREAD_COLOR );
			if(!image.data ){
				//printf( " No image data \n " );
				return -1;
			}
			//image = image(Rect(256,256,512,512));
			if(num == 1){
				//create empty 8-bit matrix
				meanImage = Mat(image.rows,image.cols,CV_8U);
			}
			//fixed = imread(	line, IMREAD_COLOR );
			//left, up, width, height
			//1024x1024
			//auto start = std::chrono::high_resolution_clock::now();
		    for (int y=0; y<image.rows; y++) { 
		     	parallel_for_(cv::Range(0, image.cols), Body(y));
		    }
		    //namedWindow( line, WINDOW_AUTOSIZE );
			//imshow( line, meanImage );
			//waitKey(0);
			/*
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
		    std::cout << "Time taken by function: " << duration.count() << " microseconds\n";
		    avgDur = avgDur + (duration.count()-avgDur)/num;
		    std::cout << "Avg Duration: " << avgDur << " microseconds\n";
		    */
			num++;
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin); 
		std::cout << "Total Time: " << duration.count() << " microseconds\n";

		myfile.close();
		namedWindow( line, WINDOW_AUTOSIZE );
		imwrite("./mean.jpg",meanImage);
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
//g++ -o PForFM PForFM.cpp `pkg-config opencv --cflags --libs` -std=c++11 -pthread