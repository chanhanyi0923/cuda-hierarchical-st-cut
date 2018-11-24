#include "Data_kernel.cuh"

__device__
bool Data_PushDown(
    int *device_weightUp,
    int *device_weightDown,
    int *device_height,
    int *device_capacity,
    int columnSize,
    // parameters
    int x,
    int y
)
{
	if (y == 0) {
		return false;
	}

	const size_t indexFrom = x * columnSize + y;
	const size_t indexTo = x * columnSize + (y - 1);

    if (device_height[indexFrom] != device_height[indexTo] + 1) {
        return false;
    }

    int value = min(device_weightDown[indexFrom], device_capacity[indexFrom]);

    device_weightDown[indexFrom] -= value;
    device_capacity[indexFrom] -= value;

    device_weightUp[indexTo] += value;
    device_capacity[indexTo] += value;

    device_height[indexTo] = device_capacity[indexTo] > 0 ? 1 : 0;

    bool active = value > 0;
    return active;
}


__device__
bool Data_PushUp(
    int *device_weightUp,
    int *device_weightDown,
    int *device_height,
    int *device_capacity,
    int columnSize,
    // parameters
    int x,
    int y
)
{
	if (y == columnSize - 1) {
		return false;
	}

	const size_t indexFrom = x * columnSize + y;
	const size_t indexTo = x * columnSize + (y + 1);

    if (device_height[indexFrom] != device_height[indexTo] + 1) {
        return false;
    }

    int value = min(device_weightUp[indexFrom], device_capacity[indexFrom]);

    device_weightUp[indexFrom] -= value;
    device_capacity[indexFrom] -= value;

    device_weightDown[indexTo] += value;
    device_capacity[indexTo] += value;

    device_height[indexTo] = device_capacity[indexTo] > 0 ? 1 : 0;

    bool active = value > 0;
    return active;
}


__device__
bool Data_PushLeft(
    int *device_weightLeft,
    int *device_weightRight,
    int *device_height,
    int *device_capacity,
    int columnSize,
    // parameters
    int x,
    int y
)
{
    if (x == 0) {
        return false;
    }

    const size_t indexFrom = x * columnSize + y;
    const size_t indexTo = (x - 1) * columnSize + y;

    if (device_height[indexFrom] != device_height[indexTo] + 1) {
        return false;
    }

    int value = min(device_weightLeft[indexFrom], device_capacity[indexFrom]);

    device_weightLeft[indexFrom] -= value;
    device_capacity[indexFrom] -= value;

    device_weightRight[indexTo] += value;
    device_capacity[indexTo] += value;

    device_height[indexTo] = device_capacity[indexTo] > 0 ? 1 : 0;

    bool active = value > 0;
    return active;
}


__device__
bool Data_PushRight(
    int *device_weightLeft,
    int *device_weightRight,
    int *device_height,
    int *device_capacity,
    int rowSize,
    int columnSize,
    // parameters
    int x,
    int y
)
{
    if (x == rowSize - 1) {
        return false;
    }

	const size_t indexFrom = x * columnSize + y;
	const size_t indexTo = (x + 1) * columnSize + y;

    if (device_height[indexFrom] != device_height[indexTo] + 1) {
        return false;
    }

    int value = min(device_weightRight[indexFrom], device_capacity[indexFrom]);

    device_weightRight[indexFrom] -= value;
    device_capacity[indexFrom] -= value;

    device_weightLeft[indexTo] += value;
    device_capacity[indexTo] += value;

    device_height[indexTo] = device_capacity[indexTo] > 0 ? 1 : 0;

    bool active = value > 0;
    return active;
}


__device__
void Data_PushFromS(
    int *device_weightS,
    int *device_height,
    int *device_capacity,
    int columnSize,
    int x,
    int y
)
{
	const size_t index = x * columnSize + y;
	if (device_weightS[index] > 0) {
		device_height[index] = 1;
	}
	device_capacity[index] += device_weightS[index];
	//this->weightS[index] = 0;
}


__device__
void Data_PushToT(
    int *device_weightT,
    int *device_height,
    int *device_capacity,
    int columnSize,
    int x,
    int y
)
{
	const size_t index = x * columnSize + y;
	int value = min(device_capacity[index], device_weightT[index]);
	device_capacity[index] -= value;
	//this->flow += value;
}



__global__
void Data_PushLeftForLine(
    bool *device_active,
    int *device_weightLeft,
    int *device_weightRight,
    int *device_weightS,
    int *device_weightT,
    int *device_height,
    int *device_capacity,
    int rowSize,
    int columnSize
)
{
    bool local_active = false;
    //int tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.y + threadIdx.x;
    int tid = threadIdx.x;
    for (int i = rowSize - 1; i >= 0; i --) {
        int x = i, y = tid;
        Data_PushFromS(
            device_weightS,
            device_height,
            device_capacity,
            columnSize,
            x,
            y
        );
        bool active = Data_PushLeft(
            device_weightLeft,
            device_weightRight,
            device_height,
            device_capacity,
            columnSize,
            // parameters
            x,
            y
        );
        local_active = local_active || active;
        Data_PushToT(
            device_weightT,
            device_height,
            device_capacity,
            columnSize,
            x,
            y
        );
    }

    if (local_active) {
        *device_active = true;
    }
}


__global__
void Data_PushRightForLine(
    bool *device_active,
    int *device_weightLeft,
    int *device_weightRight,
    int *device_weightS,
    int *device_weightT,
    int *device_height,
    int *device_capacity,
    int rowSize,
    int columnSize
)
{
    bool local_active = false;
    //int tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.y + threadIdx.x;
    int tid = threadIdx.x;
    for (int i = 0; i < rowSize; i ++) {
        int x = i, y = tid;
        Data_PushFromS(
            device_weightS,
            device_height,
            device_capacity,
            columnSize,
            x,
            y
        );
        bool active = Data_PushRight(
            device_weightLeft,
            device_weightRight,
            device_height,
            device_capacity,
            rowSize,
            columnSize,
            // parameters
            x,
            y
        );
        local_active = local_active || active;
        Data_PushToT(
            device_weightT,
            device_height,
            device_capacity,
            columnSize,
            x,
            y
        );
    }

    if (local_active) {
        *device_active = true;
    }
}


__global__
void Data_PushUpForLine(
    bool *device_active,
    int *device_weightUp,
    int *device_weightDown,
    int *device_weightS,
    int *device_weightT,
    int *device_height,
    int *device_capacity,
    int rowSize,
    int columnSize
)
{
    bool local_active = false;
    //int tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.y + threadIdx.x;
    int tid = threadIdx.x;
    for (int i = 0; i < columnSize; i ++) {
        int x = tid, y = i;
        Data_PushFromS(
            device_weightS,
            device_height,
            device_capacity,
            columnSize,
            x,
            y
        );

        bool active = Data_PushUp(
            device_weightUp,
            device_weightDown,
            device_height,
            device_capacity,
            columnSize,
            // parameters
            x,
            y
        );
        local_active = local_active || active;

        Data_PushToT(
            device_weightT,
            device_height,
            device_capacity,
            columnSize,
            x,
            y
        );
    }

    if (local_active) {
        *device_active = true;
    }
}


__global__
void Data_PushDownForLine(
    bool *device_active,
    int *device_weightUp,
    int *device_weightDown,
    int *device_weightS,
    int *device_weightT,
    int *device_height,
    int *device_capacity,
    int rowSize,
    int columnSize
)
{
    bool local_active = false;
    //int tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.y + threadIdx.x;
    int tid = threadIdx.x;
    for (int i = columnSize - 1; i >= 0; i --) {
        int x = tid, y = i;
        Data_PushFromS(
            device_weightS,
            device_height,
            device_capacity,
            columnSize,
            x,
            y
        );

        bool active = Data_PushDown(
            device_weightUp,
            device_weightDown,
            device_height,
            device_capacity,
            columnSize,
            // parameters
            x,
            y
        );
        local_active = local_active || active;

        Data_PushToT(
            device_weightT,
            device_height,
            device_capacity,
            columnSize,
            x,
            y
        );
    }

    if (local_active) {
        *device_active = true;
    }
}


__global__
void Data_BfsFromT(
    int *device_weightT,
    int *device_height,
    int rowSize,
    int columnSize
)
{
    int tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.y + threadIdx.x;
    //int tid = threadIdx.y * blockDim.y + threadIdx.x;

    if (tid < rowSize * columnSize) {
        int x = tid / columnSize, y = tid % columnSize;
        if (device_weightT[x * columnSize + y] > 0) {
            device_height[x * columnSize + y] = 1;
        } else {
            device_height[x * columnSize + y] = -1;
        }
    }
}


__global__
void Data_BfsLevelK(
    bool *device_active,
    int *device_weightUp,
    int *device_weightDown,
    int *device_weightLeft,
    int *device_weightRight,
    int *device_height,
    int rowSize,
    int columnSize,
    // parameter
    int k
)
{
    int tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.y + threadIdx.x;

    if (tid < rowSize * columnSize) {
        int x = tid / columnSize, y = tid % columnSize;

        int centerIndex = x * columnSize + y;
        int leftIndex = (x - 1) * columnSize + y;
        int rightIndex = (x + 1) * columnSize + y;
        int upIndex = x * columnSize + (y + 1);
        int downIndex = x * columnSize + (y - 1);

        if ( device_height[centerIndex] == -1 && (
            ( x != 0 && device_height[leftIndex] == k && device_weightLeft[centerIndex] > 0 ) || // left
            ( x != rowSize - 1 && device_height[rightIndex] == k && device_weightRight[centerIndex] > 0 ) || // right
            ( y != columnSize - 1 && device_height[upIndex] == k && device_weightUp[centerIndex] > 0 ) || // up
            ( y != 0 && device_height[downIndex] == k && device_weightDown[centerIndex] > 0 ) // down
        ) ) {
            device_height[centerIndex] = k + 1;
            *device_active = true;
        }
    }
}

