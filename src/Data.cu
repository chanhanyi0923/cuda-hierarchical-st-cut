#include "Data.cuh"
#include "Data_kernel.cuh"

#include <fstream>
#include <iostream>
#include <limits>

Data::Data():
	rowSize(0), columnSize(0)//, flow(0)
{
}


Data::~Data()
{
    delete[] this->weightLeft;
    delete[] this->weightRight;
    delete[] this->weightUp;
    delete[] this->weightDown;
    delete[] this->weightS;
    delete[] this->weightT;
    delete[] this->height;
    delete[] this->capacity;
//    delete[] this->bfsTag;
    cudaFree(device_active);
    cudaFree(device_weightLeft);
    cudaFree(device_weightRight);
    cudaFree(device_weightUp);
    cudaFree(device_weightDown);
    cudaFree(device_weightS);
    cudaFree(device_weightT);
    cudaFree(device_height);
    cudaFree(device_capacity);
}


#define DEBUG
#ifdef DEBUG
void Data::Print()
{
	std::cout << "Weight:" << std::endl << std::endl;

    std::cout << "Left:" << std::endl;
	for (int i = 0; i < this->rowSize; i++) {
		for (int j = 0; j < this->columnSize; j++) {
			size_t idx = (i * this->columnSize + j);
            std::cout << this->weightLeft[idx] << " ";
		}
        std::cout << std::endl;
	}
    std::cout << std::endl;
    
    std::cout << "Right:" << std::endl;
	for (int i = 0; i < this->rowSize; i++) {
		for (int j = 0; j < this->columnSize; j++) {
			size_t idx = (i * this->columnSize + j);
            std::cout << this->weightRight[idx] << " ";
		}
        std::cout << std::endl;
	}
    std::cout << std::endl;
    
    std::cout << "Up:" << std::endl;
	for (int i = 0; i < this->rowSize; i++) {
		for (int j = 0; j < this->columnSize; j++) {
			size_t idx = (i * this->columnSize + j);
            std::cout << this->weightUp[idx] << " ";
		}
        std::cout << std::endl;
	}
    std::cout << std::endl;
    
    std::cout << "Down:" << std::endl;
	for (int i = 0; i < this->rowSize; i++) {
		for (int j = 0; j < this->columnSize; j++) {
			size_t idx = (i * this->columnSize + j);
            std::cout << this->weightDown[idx] << " ";
		}
        std::cout << std::endl;
	}
    std::cout << std::endl;

	std::cout << "Capacity:" << std::endl;
	for (int i = 0; i < this->rowSize; i++) {
		for (int j = 0; j < this->columnSize; j++) {
			size_t idx = (i * this->columnSize + j);
			std::cout << this->capacity[idx] << " ";
        }
        std::cout << std::endl;
	}
    std::cout << std::endl;

	std::cout << "Height:" << std::endl;
	for (int i = 0; i < this->rowSize; i++) {
		for (int j = 0; j < this->columnSize; j++) {
			size_t idx = (i * this->columnSize + j);
			std::cout << this->height[idx] << " ";
		}
        std::cout << std::endl;
	}
    std::cout << std::endl;
    std::cout << std::endl;
}
#endif


Data::Data(size_t rowSize, size_t columnSize) :
	rowSize(rowSize), columnSize(columnSize)//, flow(0)
{
	//weight = new int[4 * rowSize * columnSize](); // (): set to zero
    weightLeft = new int[rowSize * columnSize]();
    weightRight = new int[rowSize * columnSize]();
    weightUp = new int[rowSize * columnSize]();
    weightDown = new int[rowSize * columnSize]();
	weightS = new int[rowSize * columnSize]();
	weightT = new int[rowSize * columnSize]();
	height = new int[rowSize * columnSize]();
	capacity = new int[rowSize * columnSize]();

	//int *bfsTag;

    cudaMalloc( &device_active, sizeof(bool) );
    cudaMalloc( &device_weightLeft, sizeof(int) * rowSize * columnSize );
    cudaMalloc( &device_weightRight, sizeof(int) * rowSize * columnSize );
    cudaMalloc( &device_weightUp, sizeof(int) * rowSize * columnSize );
    cudaMalloc( &device_weightDown, sizeof(int) * rowSize * columnSize );
    cudaMalloc( &device_weightS, sizeof(int) * rowSize * columnSize );
    cudaMalloc( &device_weightT, sizeof(int) * rowSize * columnSize );
    cudaMalloc( &device_height, sizeof(int) * rowSize * columnSize );
    cudaMalloc( &device_capacity, sizeof(int) * rowSize * columnSize );

    cudaMemset( device_active, false, sizeof(bool) );
    cudaMemset( device_height, 0, sizeof(int) * rowSize * columnSize );
    cudaMemset( device_capacity, 0, sizeof(int) * rowSize * columnSize );


}


void Data::Read(const char * filename)
{
	using std::fstream;
	fstream fin;
	fin.open(filename, fstream::in);
	fin >> this->rowSize >> this->columnSize;

    this->weightLeft = new int[this->rowSize * this->columnSize]();
    this->weightRight = new int[this->rowSize * this->columnSize]();
    this->weightUp = new int[this->rowSize * this->columnSize]();
    this->weightDown = new int[this->rowSize * this->columnSize]();
	this->weightS = new int[this->rowSize * this->columnSize]();
	this->weightT = new int[this->rowSize * this->columnSize]();
	this->height = new int[this->rowSize * this->columnSize]();
	this->capacity = new int[this->rowSize * this->columnSize]();
    //this->bfsTag = new int[this->rowSize * this->columnSize]();

	for (int i = 0; i < this->rowSize * this->columnSize; i++) {
		// order: s, t, left, right, up, down
		fin >> this->weightS[i] >> this->weightT[i];
        fin >> this->weightLeft[i] >> this->weightRight[i] >> this->weightUp[i] >> this->weightDown[i];
	}
	fin.close();
    
    
    cudaMalloc( &device_active, sizeof(bool) );
    cudaMalloc( &device_weightLeft, sizeof(int) * rowSize * columnSize );
    cudaMalloc( &device_weightRight, sizeof(int) * rowSize * columnSize );
    cudaMalloc( &device_weightUp, sizeof(int) * rowSize * columnSize );
    cudaMalloc( &device_weightDown, sizeof(int) * rowSize * columnSize );
    cudaMalloc( &device_weightS, sizeof(int) * rowSize * columnSize );
    cudaMalloc( &device_weightT, sizeof(int) * rowSize * columnSize );
    cudaMalloc( &device_height, sizeof(int) * rowSize * columnSize );
    cudaMalloc( &device_capacity, sizeof(int) * rowSize * columnSize );

    cudaMemset( device_active, false, sizeof(bool) );
    cudaMemset( device_height, 0, sizeof(int) * rowSize * columnSize );
    cudaMemset( device_capacity, 0, sizeof(int) * rowSize * columnSize );
    
    cudaMemcpy( device_weightS, weightS, sizeof(int) * rowSize * columnSize, cudaMemcpyHostToDevice);
    cudaMemcpy( device_weightT, weightT, sizeof(int) * rowSize * columnSize, cudaMemcpyHostToDevice);
    cudaMemcpy( device_weightLeft, weightLeft, sizeof(int) * rowSize * columnSize, cudaMemcpyHostToDevice);
    cudaMemcpy( device_weightRight, weightRight, sizeof(int) * rowSize * columnSize, cudaMemcpyHostToDevice);
    cudaMemcpy( device_weightUp, weightUp, sizeof(int) * rowSize * columnSize, cudaMemcpyHostToDevice);
    cudaMemcpy( device_weightDown, weightDown, sizeof(int) * rowSize * columnSize, cudaMemcpyHostToDevice);    

    //cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
}







int Data::GetFlow()
{
	int count = 0;// debug


    bool *active = new bool(true);
	//this->active = true;
	while (*active) {
    //for (int _ = 0; _ < 100; _ ++) {
        count ++;
        //this->Print();
    
		//this->active = false;
        cudaMemset( device_active, false, sizeof(bool) );

        Data_PushLeftForLine<<< dim3(1, 1, 1), dim3(this->columnSize, 1, 1) >>>(            
            this->device_active,
            this->device_weightLeft,
            this->device_weightRight,
            this->device_weightS,
            this->device_weightT,
            this->device_height,
            this->device_capacity,
            this->rowSize,
            this->columnSize
        );

        //cudaMemcpy(active, device_active, sizeof(bool), cudaMemcpyDeviceToHost);

        Data_PushUpForLine<<< dim3(1, 1, 1), dim3(this->rowSize, 1, 1) >>>(            
            this->device_active,
            this->device_weightUp,
            this->device_weightDown,
            this->device_weightS,
            this->device_weightT,
            this->device_height,
            this->device_capacity,
            this->rowSize,
            this->columnSize
        );

        //cudaMemcpy(active, device_active, sizeof(bool), cudaMemcpyDeviceToHost);

        Data_PushRightForLine<<< dim3(1, 1, 1), dim3(this->columnSize, 1, 1) >>>(            
            this->device_active,
            this->device_weightLeft,
            this->device_weightRight,
            this->device_weightS,
            this->device_weightT,
            this->device_height,
            this->device_capacity,
            this->rowSize,
            this->columnSize
        );

        //cudaMemcpy(active, device_active, sizeof(bool), cudaMemcpyDeviceToHost);

        Data_PushDownForLine<<< dim3(1, 1, 1), dim3(this->rowSize, 1, 1) >>>(            
            this->device_active,
            this->device_weightUp,
            this->device_weightDown,
            this->device_weightS,
            this->device_weightT,
            this->device_height,
            this->device_capacity,
            this->rowSize,
            this->columnSize
        );
        cudaMemcpy(active, device_active, sizeof(bool), cudaMemcpyDeviceToHost);
	}
    
    std::cout << count << std::endl;

//#ifdef DEBUG
    cudaMemcpy(weightS, device_weightS, sizeof(int) * rowSize * columnSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(weightT, device_weightT, sizeof(int) * rowSize * columnSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(weightLeft, device_weightLeft, sizeof(int) * rowSize * columnSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(weightRight, device_weightRight, sizeof(int) * rowSize * columnSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(weightUp, device_weightUp, sizeof(int) * rowSize * columnSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(weightDown, device_weightDown, sizeof(int) * rowSize * columnSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(height, device_height, sizeof(int) * rowSize * columnSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(capacity, device_capacity, sizeof(int) * rowSize * columnSize, cudaMemcpyDeviceToHost);
//#endif

    
    // this->Print();
    // std::cout << "-----" << std::endl;
    
    
    return 0;
	//return this->flow;
}


void Data::BfsFromT()
{
    dim3 grid((this->rowSize * this->columnSize + 32767) / 32768, 32, 1);
    dim3 block(32, 32, 1);

    Data_BfsFromT<<< grid, block >>>(
        this->device_weightT,
        this->device_height,
        this->rowSize,
        this->columnSize
    );

    // debug
    //cudaMemcpy(height, device_height, sizeof(int) * rowSize * columnSize, cudaMemcpyDeviceToHost); this->Print();
    // debug

    int maxHeight = 1;
    for (bool *active = new bool(true); *active; maxHeight ++) {
        cudaMemset( device_active, false, sizeof(bool) );
        Data_BfsLevelK<<< grid, block >>>(
            this->device_active,
            this->device_weightUp,
            this->device_weightDown,
            this->device_weightLeft,
            this->device_weightRight,
            this->device_height,
            this->rowSize,
            this->columnSize,
            // parameter
            maxHeight
        );
        cudaMemcpy(active, device_active, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(height, device_height, sizeof(int) * rowSize * columnSize, cudaMemcpyDeviceToHost);
    
    // debug
    //this->Print();
    // debug
    
    // html debug
	int result[100][100] = { 0 };

	for (int i = 0; i < this->rowSize; i++) {
		for (int j = 0; j < this->columnSize; j++) {
			const size_t index = i * this->columnSize + j;
			int v = this->height[index];
			if (v != -1) {


				if (i != 0 && weightLeft[index] > 0 &&
					this->height[(i - 1) * this->columnSize + j] == -1) {
					result[i][j] = 1;
					result[i - 1][j] = 2;
				}

				if (i != this->rowSize && weightRight[index] > 0 &&
					this->height[(i + 1) * this->columnSize + j] == -1) {
					result[i][j] = 1;
					result[i + 1][j] = 2;
				}

				if (j != this->columnSize - 1 && weightUp[index] > 0 &&
					this->height[i * this->columnSize + j + 1] == -1) {
					result[i][j] = 1;
					result[i][j + 1] = 2;
				}

				if (j != 0 && weightDown[index] > 0 &&
					this->height[i * this->columnSize + j - 1] == -1) {
					result[i][j] = 1;
					result[i][j - 1] = 2;
				}


			}
		}
	}


	std::cout << this->rowSize << " " << this->columnSize << std::endl;
	for (int i = 0; i < this->rowSize; i++) {
		std::cout << "<tr>";
		for (int j = 0; j < this->columnSize; j++) {
			int v = this->height[i * this->columnSize + j];
			//if (v == 6) {
			//	std::cout << "(" << i << " " << j << ")" << std::endl;
			//}
			//std::cout << (v > 10000 ? -1 : v) << " ";

			//if (v == max_k) {
			//	std::cout << "<td class=\"yellow\">&nbsp;</td>";
			if (result[i][j] == 1) {
				std::cout << "<td class=\"yellow\">&nbsp;</td>";
			} else if (result[i][j] == 2) {
				std::cout << "<td class=\"purple\">&nbsp;</td>";
			} else if (this->weightS[i * this->columnSize + j] > 0) {
				std::cout << "<td class=\"red\">&nbsp;</td>";
			} else if (this->weightT[i * this->columnSize + j] > 0) {
				std::cout << "<td class=\"green\">&nbsp;</td>";

			} else {
				if (v == -1) {
					std::cout << "<td>&nbsp;</td>";
				} else {
					std::cout << "<td style=\"background-color: rgb(0, 0," << (5 * v) << ")\">&nbsp;</td>";
				}
			}
		}
		std::cout << "</tr>";
		//std::cout << std::endl;
	}
    // html debug
}


/*
void Data::BfsFromT()
{
	for (int i = 0; i < this->rowSize; i++) {
		for (int j = 0; j < this->columnSize; j++) {
			if (this->weightT[i * this->columnSize + j] > 0) {
				this->height[i * this->columnSize + j] = 1;
			} else {
				this->height[i * this->columnSize + j] = std::numeric_limits<int>::max();
			}
		}
	}

	//int max_k = -1;
	int maxHeight = 0;

	bool check = true;
	for (int k = 1; check; k++) {
		check = false;
		for (int i = 0; i < this->rowSize; i++) {
			for (int j = 0; j < this->columnSize; j++) {
				if (this->height[i * this->columnSize + j] == k) {
					//
					if (i != 0 && weightRight[(i - 1) * this->columnSize + j] > 0 &&
						this->height[(i - 1) * this->columnSize + j] > k) {
						this->height[(i - 1) * this->columnSize + j] = k + 1;
						check = true;
					}

					if (i != this->rowSize && weightLeft[(i + 1) * this->columnSize + j] > 0 &&
						this->height[(i + 1) * this->columnSize + j] > k) {
						this->height[(i + 1) * this->columnSize + j] = k + 1;
						check = true;
					}

					if (j != this->columnSize - 1 && weightDown[i * this->columnSize + j + 1] > 0 &&
						this->height[i * this->columnSize + j + 1] > k) {
						this->height[i * this->columnSize + j + 1] = k + 1;
						check = true;
					}

					if (j != 0 && weightUp[i * this->columnSize + j - 1] > 0 &&
						this->height[i * this->columnSize + j - 1] > k) {
						this->height[i * this->columnSize + j - 1] = k + 1;
						check = true;
					}
					//
				}
			}
		}
		//std::cout << "k = " << k << std::endl;
		maxHeight = k;
	}

	//for (int i = 0; i < this->rowSize * this->columnSize; i++) {
	//	std::cout << this->weight[i * 4] << " ";
	//	std::cout << this->weight[i * 4 + 1] << " ";
	//	std::cout << this->weight[i * 4 + 2] << " ";
	//	std::cout << this->weight[i * 4 + 3] << std::endl;
	//}


	// return if there is a path from t to s
	// for (int i = 0; i < this->rowSize; i++) {
		// for (int j = 0; j < this->columnSize; j++) {
			// const size_t index = i * this->columnSize + j;
			// if (this->height[index] <= maxHeight && this->weightS[index] > 0) {
				// return true;
			// }
		// }
	// }
	// return false;

#define HTML_DEBUG
#ifdef HTML_DEBUG

	int result[100][100] = { 0 };

	for (int i = 0; i < this->rowSize; i++) {
		for (int j = 0; j < this->columnSize; j++) {
			const size_t index = i * this->columnSize + j;
			int v = this->height[index];
			if (v <= maxHeight) {


				if (i != 0 && weightLeft[index] > 0 &&
					this->height[(i - 1) * this->columnSize + j] > maxHeight) {
					result[i][j] = 1;
					result[i - 1][j] = 2;
				}

				if (i != this->rowSize && weightRight[index] > 0 &&
					this->height[(i + 1) * this->columnSize + j] > maxHeight) {
					result[i][j] = 1;
					result[i + 1][j] = 2;
				}

				if (j != this->columnSize - 1 && weightUp[index] > 0 &&
					this->height[i * this->columnSize + j + 1] > maxHeight) {
					result[i][j] = 1;
					result[i][j + 1] = 2;
				}

				if (j != 0 && weightDown[index] > 0 &&
					this->height[i * this->columnSize + j - 1] > maxHeight) {
					result[i][j] = 1;
					result[i][j - 1] = 2;
				}


			}
		}
	}


	std::cout << this->rowSize << " " << this->columnSize << std::endl;
	for (int i = 0; i < this->rowSize; i++) {
		std::cout << "<tr>";
		for (int j = 0; j < this->columnSize; j++) {
			int v = this->height[i * this->columnSize + j];
			//if (v == 6) {
			//	std::cout << "(" << i << " " << j << ")" << std::endl;
			//}
			//std::cout << (v > 10000 ? -1 : v) << " ";

			//if (v == max_k) {
			//	std::cout << "<td class=\"yellow\">&nbsp;</td>";
			if (result[i][j] == 1) {
				std::cout << "<td class=\"yellow\">&nbsp;</td>";
			} else if (result[i][j] == 2) {
				std::cout << "<td class=\"purple\">&nbsp;</td>";
			} else if (this->weightS[i * this->columnSize + j] > 0) {
				std::cout << "<td class=\"red\">&nbsp;</td>";
			} else if (this->weightT[i * this->columnSize + j] > 0) {
				std::cout << "<td class=\"green\">&nbsp;</td>";

			} else {
				if (v > maxHeight) {
					std::cout << "<td>&nbsp;</td>";
				} else {
					std::cout << "<td style=\"background-color: rgb(0, 0," << (5 * v) << ")\">&nbsp;</td>";
				}
			}
		}
		std::cout << "</tr>";
		//std::cout << std::endl;
	}
#endif

	//std::cout << this->rowSize << " " << this->columnSize << std::endl;
	//for (int i = 0; i < this->rowSize; i++) {
	//	for (int j = 0; j < this->columnSize; j++) {
	//		int v = this->height[i * this->columnSize + j];
	//		if (v == 6) {
	//			std::cout << "(" << i << ", " << j << ")" << std::endl;
	//		}
	//	}
	//	std::cout << std::endl;
	//}
}
*/

