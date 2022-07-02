#include<iostream>
#include<cmath>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <smmintrin.h> //SSE4.1
#include <pthread.h>
#include <omp.h>
#include <sys/time.h>
#define INTERVAL 10000
#define NUM_THREADS 6
using namespace std;
typedef long long ll;

const int dim = 128;
const int trainNum = 1024;
const int testNum = 128;
float test[testNum][dim];
float train[trainNum][dim];
float dist[testNum][trainNum];

//pthread function for square_unwrapped
void* train_square(void* temp_train)
{
	#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0;i < trainNum;i++)
	{
		float sum = 0.0;
		for (int j = 0;j < dim;j++)
			sum += train[i][j] * train[i][j];
		((float*)temp_train)[i] = sum;
	}
}

void* test_square(void* temp_test)
{
	#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0;i < testNum;i++)
	{
		float sum = 0.0;
		for (int j = 0;j < dim;j++)
			sum += test[i][j] * test[i][j];
		((float*)temp_test)[i] = sum;
	}
}

void plain()
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
			dist[i][j] = sqrtf(sum);
		}
	}
}

void square_unwrapped()
{
	float temp_test[testNum];
	float temp_train[trainNum];
	//#pragma omp parallel for num_threads(NUM_THREADS)
	//for (int i = 0;i < testNum;i++)
	//{
	//	float sum = 0.0;
	//	for (int j = 0;j < dim;j++)
	//		sum += test[i][j] * test[i][j];
	//	temp_test[i] = sum;
	//}
	//#pragma omp parallel for num_threads(NUM_THREADS)
	//for (int i = 0;i < trainNum;i++)
	//{
	//	float sum = 0.0;
	//	for (int j = 0;j < dim;j++)
	//		sum += train[i][j] * train[i][j];
	//	temp_train[i] = sum;
	//}

	pthread_t* train_h, *test_h;
	pthread_create(train_h, NULL, train_square, (void*)train);
	pthread_create(test_h, NULL, test_square, (void*)test);
	pthread_join(*train_h, NULL);
	pthread_join(*test_h, NULL);
	#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0;i < testNum;i++)
		#pragma omp parallel for num_threads(NUM_THREADS)
		for (int j = 0;j < trainNum;j++)
		{
			float sum = 0;
			for (int k = 0;k < dim;k++)
				sum += test[i][k] * train[j][k];
			dist[i][j] = sqrtf(temp_test[i] + temp_train[j] - 2 * sum);
		}
}

void sqrt_unwrapped()
{
	#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0;i < testNum;i++)
	{
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
			_mm_store_ss(dist[i] + j, sum);
			dist[i][j] += sumTemp;
		}
		for (int j = 0;j < trainNum;j += 4)
		{
			__m128 temp_dist = _mm_load_ps(&dist[i][j]);
			temp_dist = _mm_sqrt_ps(temp_dist);
			_mm_store_ps(&dist[i][j], temp_dist);
		}
	}
}

void vertical_SIMD()
{
	#pragma omp parallel for num_threads(8)
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
			dist[i][j] = sqrtf(sum);
		}
		for (int j = 0;j < trainNum;j += 4)//并行处理4的倍数部分
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
			_mm_store_ss(&dist[i][j], sum);
		}
	}
}

void square_unwrapped_SIMD()
{
	float temp_test[testNum];
	float temp_train[trainNum];
	#pragma omp parallel for num_threads(8)
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
	#pragma omp parallel for num_threads(8)
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
	#pragma omp parallel for num_threads(8)
	for (int i = 0;i < testNum;i++)
	{
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
			_mm_store_ss(&temp_train[i], sum);
			dist[i][j] += sumTemp;
		}
		//dist[i][j] = sqrtf(temp_test[i] + temp_train[j] - 2 * sum);
		int j = 0;
		for (;j < trainNum && (trainNum - j) & 3;j++)
			dist[i][j] = sqrtf(temp_test[i] + temp_train[j] - 2 * dist[i][j]);
		__m128 _test = _mm_load1_ps(&temp_test[i]);
		for (;j < trainNum;j += 4)
		{
			__m128 _train = _mm_loadu_ps(&temp_train[j]);
			__m128 res = _mm_loadu_ps(&dist[i][j]);
			_train = _mm_sub_ps(_train, res);
			_train = _mm_sub_ps(_train, res);
			_train = _mm_add_ps(_test, _train);
			_mm_storeu_ps(&dist[i][j], _train);
		}
	}
}

void timing(void (*func)())
{
    timeval tv_begin, tv_end;
    int counter(0);
    double time = 0;
    while(INTERVAL>time)
    {
    	gettimeofday(&tv_begin, 0);
        func();
        gettimeofday(&tv_end, 0);
        counter++;
        time += ((ll)tv_end.tv_sec - (ll)tv_begin.tv_sec)*1000.0 + ((ll)tv_end.tv_usec - (ll)tv_begin.tv_usec)/1000.0;
    }
    cout<<time/counter<<","<<counter<<'\t';
}

void init()
{
	for (int i = 0;i < testNum;i++)
		for (int k = 0;k < dim;k++)
			test[i][k] = rand() / double(RAND_MAX) * 1000;//0-100间随机浮点数
	for (int i = 0;i < trainNum;i++)
		for (int k = 0;k < dim;k++)
			train[i][k] = rand() / double(RAND_MAX) * 1000;//0-100间随机浮点数
}

int main()
{
	init();
	printf("%s", "朴素算法耗时：");
	timing(plain);
	printf("%s", "平方展开串行耗时：");
	timing(square_unwrapped);
	printf("%s", "横向展开算法耗时：");
	timing(sqrt_unwrapped);
	printf("%s", "纵向展开算法耗时：");
	timing(vertical_SIMD);
	printf("%s", "平方展开算法耗时：");
	timing(square_unwrapped_SIMD);
	return 0;
}