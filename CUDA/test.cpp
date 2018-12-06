#include <iostream>

#include "opencv2/core/cuda.hpp"

using namespace std;
using namespace cv;

int main()
{
	cout << cuda::getDevice();
}