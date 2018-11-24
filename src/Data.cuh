class Data
{
private:
	//bool active;
	size_t rowSize, columnSize;
	int *weightLeft, *weightRight, *weightUp, *weightDown;
	int *weightS, *weightT;
	int *height, *capacity;

	//int flow;
	//int *bfsTag;

    // variables on device (GPU)
    bool *device_active;
	int *device_weightLeft, *device_weightRight, *device_weightUp, *device_weightDown;
	int *device_weightS, *device_weightT;
	int *device_height, *device_capacity;

public:
	Data();
	Data(size_t rowSize, size_t columnSize);
    ~Data();
	void Read(const char * filename);
	int GetFlow();
	void BfsFromT();
    void Print();
};
