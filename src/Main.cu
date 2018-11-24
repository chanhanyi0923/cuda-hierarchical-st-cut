#define DEBUG

#include "Data.cuh"

int main()
{
    Data * data = new Data;
	data->Read("../../data/test3.txt");
	int flow = data->GetFlow();
    data->BfsFromT();

    return 0;
}
