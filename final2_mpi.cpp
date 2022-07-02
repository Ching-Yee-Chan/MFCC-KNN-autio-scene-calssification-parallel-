#include <iostream>
#include <cmath>
#include <algorithm>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <smmintrin.h> //SSE4.1
#include <pthread.h>
#include <omp.h>
#include <mpi.h>
#include <windows.h>
#define MAX 1e9
#define INTERVAL 1000
#define NUM_THREADS 6
using namespace std;
typedef long long ll;

const int k = 5;
const int dim = 128;

void plain(float (*train)[dim], float (*test)[dim], float* dist, int trainNum, int testNum)
{
	#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0;i < testNum;i++)
	{
		#pragma omp parallel for num_threads(NUM_THREADS)
		for (int j = 0;j < trainNum;j++)
		{
			float sum = 0.0;
			for (int k = 0;k < dim;k++)
			{
				float temp = test[i][k] - train[j][k];
				temp *= temp;
				sum += temp;
			}
			dist[i * trainNum + j] = sqrtf(sum);
		}
	}
}

void square_unwrapped(float(*train)[dim], float(*test)[dim], float* dist, int trainNum, int testNum)
{
	float* temp_test = new float[testNum];
	float* temp_train = new float[trainNum];
	#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0;i < testNum;i++)
	{
		float sum = 0.0;
		for (int j = 0;j < dim;j++)
			sum += test[i][j] * test[i][j];
		temp_test[i] = sum;
	}
	#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0;i < trainNum;i++)
	{
		float sum = 0.0;
		for (int j = 0;j < dim;j++)
			sum += train[i][j] * train[i][j];
		temp_train[i] = sum;
	}

	//pthread_t train_h, test_h;
	//pthread_create(&train_h, NULL, train_square, (void*)temp_train);
	//pthread_create(&test_h, NULL, test_square, (void*)temp_test);
	//pthread_join(train_h, NULL);
	//pthread_join(test_h, NULL);
	#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0;i < testNum;i++)
		#pragma omp parallel for num_threads(NUM_THREADS)
		for (int j = 0;j < trainNum;j++)
		{
			float sum = 0;
			for (int k = 0;k < dim;k++)
				sum += test[i][k] * train[j][k];
			dist[i * trainNum + j] = sqrtf(temp_test[i] + temp_train[j] - 2 * sum);
		}
    delete[] temp_test;
    delete[] temp_train;
}

 void sqrt_unwrapped(float(*train)[dim], float(*test)[dim], float* dist, int trainNum, int testNum)
 {
 	#pragma omp parallel for num_threads(NUM_THREADS)
 	for (int i = 0;i < testNum;i++)
 	{
        #pragma omp parallel for num_threads(NUM_THREADS)
 		for (int j = 0;j < trainNum;j++)
 		{
 			int k = 0;
 			int sumTemp = 0;
 			for (;(dim - k) & 3;k++)
 			{
 				float temp = test[i][k] - train[j][k];
 				temp *= temp;
 				sumTemp += temp;
 			}
 			__m128 sum = _mm_setzero_ps();
 			for (;k < dim;k += 4)
 			{
 				__m128 temp_test = _mm_load_ps(&test[i][k]);
 				__m128 temp_train = _mm_load_ps(&train[j][k]);
 				temp_test = _mm_sub_ps(temp_test, temp_train);
 				temp_test = _mm_mul_ps(temp_test, temp_test);
 				sum = _mm_add_ps(sum, temp_test);
 			}
 			sum = _mm_hadd_ps(sum, sum);
 			sum = _mm_hadd_ps(sum, sum);
 			_mm_store_ss(dist + i * trainNum + j, sum);
 			dist[i * trainNum + j] += sumTemp;
 		}
        int j = 0;
        for (;j < trainNum && (trainNum - j) & 3;j++)
            dist[i * trainNum + j] = sqrtf(dist[i * trainNum + j]);
 		for (;j < trainNum;j += 4)
 		{
 			__m128 temp_dist = _mm_load_ps(&dist[i * trainNum + j]);
 			temp_dist = _mm_sqrt_ps(temp_dist);
 			_mm_store_ps(&dist[i * trainNum + j], temp_dist);
 		}
 	}
 }

 void vertical_SIMD(float(*train)[dim], float(*test)[dim], float* dist, int trainNum, int testNum)
 {
 	#pragma omp parallel for num_threads(NUM_THREADS)
 	for (int i = 0;i < testNum;i++)
 	{
 		int j = 0;
 		for (;j < trainNum && (trainNum - j) & 3;j++)//串行处理剩余部分
 		{
 			float sum = 0.0;
 			for (int k = 0;k < dim;k++)
 			{
 				float temp = test[i][k] - train[j][k];
 				temp *= temp;
 				sum += temp;
 			}
            dist[i * trainNum + j] = sqrtf(sum);
 		}
 		for (;j < trainNum;j += 4)//并行处理4的倍数部分
 		{
 			__m128 sum = _mm_set1_ps(0);
 			for (int k = 0;k < dim;k++)
 			{
 				__m128 temp_train, temp_test;
 				temp_train = _mm_set_ps(train[j + 3][k], train[j + 2][k], train[j + 1][k], train[j][k]);
 				temp_test = _mm_load1_ps(&test[i][k]);
 				temp_test = _mm_sub_ps(temp_test, temp_train);
 				temp_test = _mm_add_ps(temp_test, temp_test);
 				sum = _mm_add_ps(temp_test, sum);
 			}
 			_mm_store_ss(&dist[i * trainNum + j], sum);
 		}
 	}
 }

 void square_unwrapped_SIMD(float(*train)[dim], float(*test)[dim], float* dist, int trainNum, int testNum)
 {
     float* temp_test = new float[testNum];
     float* temp_train = new float[trainNum];
 	#pragma omp parallel for num_threads(NUM_THREADS)
 	for (int i = 0;i < testNum;i++)
 	{
 		int j = 0;
 		int sumTemp = 0;
 		for (;j < dim && (dim - j) & 3;j++)
 			sumTemp += test[i][j] * test[i][j];
 		__m128 sum = _mm_set1_ps(0);
 		for (;j < dim;j += 4)
 		{
 			__m128 square = _mm_loadu_ps(&test[i][j]);
 			square = _mm_mul_ps(square, square);
 			sum = _mm_add_ps(sum, square);
 		}
 		sum = _mm_hadd_ps(sum, sum);
 		sum = _mm_hadd_ps(sum, sum);
 		_mm_store_ss(&temp_test[i], sum);
 		temp_test[i] += sumTemp;
 	}
 	#pragma omp parallel for num_threads(NUM_THREADS)
 	for (int i = 0;i < trainNum;i++)
 	{
 		int j = 0;
 		int sumTemp = 0;
 		for (;j < dim && (dim - j) & 3;j++)
 			sumTemp += train[i][j] * train[i][j];
 		__m128 sum = _mm_set1_ps(0);
 		for (;j < dim;j += 4)
 		{
 			__m128 square = _mm_loadu_ps(&train[i][j]);
 			square = _mm_mul_ps(square, square);
 			sum = _mm_add_ps(sum, square);
 		}
 		sum = _mm_hadd_ps(sum, sum);
 		sum = _mm_hadd_ps(sum, sum);
 		_mm_store_ss(&temp_train[i], sum);
 		temp_train[i] += sumTemp;
 	}
 	#pragma omp parallel for num_threads(NUM_THREADS)
 	for (int i = 0;i < testNum;i++)
 	{
        #pragma omp parallel for num_threads(NUM_THREADS)
 		for (int j = 0;j < trainNum;j++)
 		{
 			int k = 0;
 			int sumTemp = 0;
 			for (;k < dim && (dim - k) & 3;k++)
 				sumTemp += train[j][k] * test[i][k];
 			__m128 sum = _mm_set1_ps(0);
 			for (;k < dim;k += 4)
 			{
 				__m128 _train = _mm_loadu_ps(&train[j][k]);
 				__m128 _test = _mm_loadu_ps(&test[i][k]);
 				_train = _mm_mul_ps(_train, _test);
 				sum = _mm_add_ps(_train, sum);
 			}
 			sum = _mm_hadd_ps(sum, sum);
 			sum = _mm_hadd_ps(sum, sum);
 			_mm_store_ss(&dist[i * trainNum + j], sum);
            dist[i * trainNum + j] += sumTemp;
 		}
 		//dist[i][j] = sqrtf(temp_test[i] + temp_train[j] - 2 * sum);
 		int j = 0;
 		for (;j < trainNum && (trainNum - j) & 3;j++)
            dist[i * trainNum + j] = sqrtf(temp_test[i] + temp_train[j] - 2 * dist[i * trainNum + j]);
 		__m128 _test = _mm_load1_ps(&temp_test[i]);
 		for (;j < trainNum;j += 4)
 		{
 			__m128 _train = _mm_loadu_ps(&temp_train[j]);
 			__m128 res = _mm_loadu_ps(&dist[i * trainNum + j]);
 			_train = _mm_sub_ps(_train, res);
 			_train = _mm_sub_ps(_train, res);
 			_train = _mm_add_ps(_test, _train);
 			_mm_storeu_ps(&dist[i * trainNum + j], _train);
 		}
 	}
    delete[] temp_train;
    delete[] temp_test;
 }

void getClass(float* dist, bool* trainLabel, bool *predLabel, int trainNum, int testNum)
{
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0;i < testNum;i++)
    {
        double sum = 0;
        //#pragma omp parallel for num_threads(NUM_THREADS)
        for (int j = 0;j < k;j++)
        {
            int minLabel = 0;
            for (int k = 0;k < testNum;k++)
                if (dist[i * trainNum + k] < dist[i * trainNum + minLabel])
                    minLabel = k;
            sum += trainLabel[minLabel];
            dist[i * trainNum + minLabel] = MAX;
        }
        predLabel[i] = (sum / k) > 0.5;
    }
}

void init(float(*train)[dim], float(*test)[dim], bool* trainLabel, float trainNum, float testNum)
{
	for (int i = 0;i < testNum;i++)
		for (int k = 0;k < dim;k++)
			test[i][k] = rand() / double(RAND_MAX) * 1000;//0-100间随机浮点数
	for (int i = 0;i < trainNum;i++)
		for (int k = 0;k < dim;k++)
			train[i][k] = rand() / double(RAND_MAX) * 1000;//0-100间随机浮点数
    for (int i = 0;i < trainNum;i++)
        trainLabel[i] = (i & 1) ? 0 : 1;
}

void serial()
{
    ll head, tail, freq;
    double time = 0;
    int counter = 0;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    const int trainNum = 1024;
    const int testNum = 128;
    float (*test)[dim] = new float[testNum][dim];
    float (*train)[dim] = new float [trainNum] [dim] ;
    float (*dist)[trainNum] = new float[testNum][trainNum];
    bool trainLabel[trainNum];
    bool predLabel[testNum];
    while (INTERVAL > time)
    {
        init(train, test, trainLabel, trainNum, testNum);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        square_unwrapped_SIMD(train,test, (float*)dist, trainNum, testNum);
        getClass((float*)dist, trainLabel, predLabel, trainNum, testNum);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        time += (tail - head) * 1000.0 / freq;
        counter++;
    }
    std::cout << "串行：" << time / counter << '\n';
    delete[] test;
    delete[] train;
    delete[] dist;
}

void testDivMPI()
{
    double tot = 0;

    const int trainNum = 1024;
    const int testNum = 128;
    float (*test)[dim] = NULL;
    static float train[trainNum][dim];
    float (*dist)[trainNum] = NULL;
    bool trainLabel[trainNum];
    bool* predLabel = NULL;
    bool* result = NULL;

    double start, finish;//计时变量
    int comm_sz;
    int my_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    for(int i = 0;i < 1000;i++)
    {
        int* sendCounts = new int[comm_sz];
        int* recvCounts = new int[comm_sz];
        int* displs = new int[comm_sz + 1];
        int* recvdispls = new int[comm_sz + 1];
        int pos = 0;
        fill(sendCounts, sendCounts + testNum % comm_sz, (int)ceil((float)testNum / comm_sz) * dim);
        fill(sendCounts + testNum % comm_sz, sendCounts + comm_sz, testNum / comm_sz * dim);
        fill(recvCounts, recvCounts + testNum % comm_sz, (int)ceil((float)testNum / comm_sz));
        fill(recvCounts + testNum % comm_sz, recvCounts + comm_sz, testNum / comm_sz);
        for (int i = 0;i < comm_sz;i++)
        {
            displs[i] = pos;
            recvdispls[i] = pos / dim;
            pos += sendCounts[i];
        }
        displs[comm_sz] = pos;
        recvdispls[comm_sz] = pos / dim;
        if (my_rank == 0)
        {
            test = new float[testNum][dim];
            result = new bool[testNum];
            init(train, test, trainLabel, trainNum, testNum);
            start = MPI_Wtime();
        }
        dist = new float[recvCounts[my_rank]][trainNum];
        predLabel = new bool[recvCounts[my_rank]];
        float(*myTest)[dim] = new float[recvCounts[my_rank]][dim];
        MPI_Bcast(train, trainNum * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);//广播训练集
        MPI_Bcast(trainLabel, trainNum, MPI_C_BOOL, 0, MPI_COMM_WORLD);//广播训练集标签
        MPI_Scatterv(test, sendCounts, displs, MPI_FLOAT, myTest, sendCounts[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
        //cout << my_rank << "starts" << endl;
        square_unwrapped_SIMD(train, myTest, (float*)dist, trainNum, recvCounts[my_rank]);
        getClass((float*)dist, trainLabel, predLabel, trainNum, recvCounts[my_rank]);
        //cout << my_rank << "ends" << endl;
        MPI_Gatherv(predLabel, recvCounts[my_rank], MPI_C_BOOL, result, recvCounts, recvdispls, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        if (my_rank == 0)
        {
            finish = MPI_Wtime();
            //cout << "测试集划分：" << (finish - start) * 1000 << endl;
            tot += (finish - start)*1000;
            delete[] test;
            delete[] result;
        }
        delete[] myTest;
        delete[] dist;
        delete[] predLabel;
        delete[] sendCounts;
        delete[] recvCounts;
        delete[] displs;
        delete[] recvdispls;
    }
    MPI_Finalize();
    cout << "测试集划分：" << tot/1000<< endl;
}

int main()
{
    serial();
    testDivMPI();
	return 0;
}