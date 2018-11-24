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
);

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
);

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
);

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
);

__device__
void Data_PushFromS(
    int *device_weightS,
    int *device_height,
    int *device_capacity,
    int columnSize,
    int x,
    int y
);

__device__
void Data_PushToT(
    int *device_weightT,
    int *device_height,
    int *device_capacity,
    int columnSize,
    int x,
    int y
);

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
);

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
);

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
);

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
);

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
);


__global__
void Data_BfsFromT(
    int *device_weightT,
    int *device_height,
    int rowSize,
    int columnSize
);


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
);

