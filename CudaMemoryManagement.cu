#include <iostream>

using namespace std;

__global__ void changeValue(int *val) {
    *val = 50;
}

__global__ void changeValueArr(int *val) {
    val[blockIdx.x] = 40;
}

template <typename DEFTYPE>
class CudaMemoryManagement {
private:
    DEFTYPE* arr_store[100];
    int arr_store_ctr;
public:
    CudaMemoryManagement() {
        this->arr_store_ctr = 0;
    }
    ~CudaMemoryManagement() {
        // Free memory here?
        for (int i = 0; i < arr_store_ctr; i++) {
            cudaFree(arr_store[i]);
        }
    }

    // Create memory area on GPU - side
    DEFTYPE* createMemory(DEFTYPE originalValue) {
        DEFTYPE* device_value;
        cudaMalloc((void **)&device_value, sizeof(DEFTYPE));
        cudaMemcpy(device_value, &originalValue, sizeof(DEFTYPE), cudaMemcpyHostToDevice);
        // Register to arr_store
        this->arr_store[arr_store_ctr++] = device_value;
        return device_value;
    }

    // Overload function for previous createMemory (For 1d array implementation)
    DEFTYPE* createMemory(DEFTYPE* originalValue, int length) {
        DEFTYPE* device_value;
        cudaMalloc((void **)&device_value, sizeof(DEFTYPE) * length);
        cudaMemcpy(device_value, &originalValue, sizeof(DEFTYPE) * length, cudaMemcpyHostToDevice);
        // Register to arr_store
        this->arr_store[arr_store_ctr++] = device_value;
        return device_value;
    }

    // Get the result from device_ptr(GPU's memory)
    void getResult(DEFTYPE *device_ptr, DEFTYPE *host_ptr) {
        cudaMemcpy(host_ptr, device_ptr, sizeof(DEFTYPE), cudaMemcpyDeviceToHost);
    }

    // Overload function for previous getResult (For 1d array implementation)
    void getResult(DEFTYPE *device_ptr, DEFTYPE *host_ptr, int length) {
        cudaMemcpy(host_ptr, device_ptr, sizeof(DEFTYPE) * length, cudaMemcpyDeviceToHost);
    }
};

int main(void) {
    CudaMemoryManagement<int> cmm;
    int arr_host[10];
    for (int i = 0; i < 10; i++) {
        arr_host[i] = rand()%10;
    }
    int *dev_arr = cmm.createMemory(arr_host, 10);
    changeValueArr<<<10, 1>>>(dev_arr);
    cmm.getResult(dev_arr, arr_host, 10);
    for (int i = 0; i < 10; i++) {
        cout << arr_host[i] << endl;
    }
    return 0;
}