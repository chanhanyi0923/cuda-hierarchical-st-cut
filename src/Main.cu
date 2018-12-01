#define DEBUG

#include "Data.cuh"

int main(int argc, char *argv[])
{
    Data * data = new Data;
	data->Read(argv[1]);
	int flow = data->GetFlow();
    data->BfsFromT();

    return 0;
}
