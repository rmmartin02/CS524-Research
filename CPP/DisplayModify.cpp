#include <opencv2/opencv.hpp>
using namespace cv;

int tol = 60;
/*
Vec3b interPolat(Mat image, int x, int y){
	int c = 0;
	int total = 0;
	Vec3b colors [8] = {
		image.at<Vec3b>(Point(x-1,y-1)),
		image.at<Vec3b>(Point(x-1,y)),
		image.at<Vec3b>(Point(x-1,y+1)),
		image.at<Vec3b>(Point(x,y-1)),
		image.at<Vec3b>(Point(x,y+1)),
		image.at<Vec3b>(Point(x+1,y-1)),
		image.at<Vec3b>(Point(x+1,y)),
		image.at<Vec3b>(Point(x+1,y+1))
	};
	for (int i = 0; i<9; i++){
		if (colors[i][0]!=0 && (colors[i][2]-colors[i][0])<tol && (colors[i][2]-colors[i][0])>-tol){
			//printf("%d, %d : %d %d %d\n",x,y,colors[i][0],colors[i][1],colors[i][2]);
			total += colors[i][0];
			c++;
		}
	}
	if(c!=0){
		total = total/c;
	}
	return Vec3b(total,total,total);
}*/

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
	char* imageName = argv[1];
	Mat image;
	Mat fixed;
	image = imread( imageName, IMREAD_COLOR );
	fixed = imread(	imageName, IMREAD_COLOR );
	if( argc != 2 || !image.data ){
		printf( " No image data \n " );
		return -1;
	}
	//left, up, width, height
	//1024x1024
	image = image(Rect(200,200,600,600));
	fixed = fixed(Rect(200,200,600,600));

	//535 473
	for(int y=1;y<image.rows-1;y++)
	{
		for(int x=1;x<image.cols-1;x++)
		{
			// get pixel
			Vec3b color = image.at<Vec3b>(Point(x,y));
			
			if ((color[2]-color[0])>tol || (color[2]-color[0])<-tol){
				color = interPolat(image,x,y);
			}

			// set pixel
			fixed.at<Vec3b>(Point(x,y)) = color;
		}
	}

	cvtColor( fixed, fixed, COLOR_BGR2GRAY );
	namedWindow( imageName, WINDOW_AUTOSIZE );
	imshow( imageName, fixed );
	waitKey(0);
	return 0;
}
