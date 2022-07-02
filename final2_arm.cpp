#include <arm_neon.h>
#include<assert.h>
#include <stdio.h>
#include <sys/time.h>
#include<cmath>
#include <pthread.h>
#include <iostream>
#define INTERVAL 1000
#define NUM_THREADS 6
using namespace std;
typedef long long ll;
const int dim = 128;
const int trainNum = 1024;
const int testNum = 128;
float train[trainNum][dim];
float test[testNum][dim];
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

void sqrt_unwrapped()
{
    #pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0;i < testNum;i++)
	{
        #pragma omp parallel for num_threads(NUM_THREADS)
		for (int j = 0;j < trainNum;j++)
		{
			assert(dim % 4 == 0);//首先假定维度为4的倍数
			float32x4_t sum = vmovq_n_f32(0);
			for (int k = 0;k < dim;k += 4)
			{
				float32x4_t temp_test = vld1q_f32(&test[i][k]);
				float32x4_t temp_train = vld1q_f32(&train[j][k]);
				temp_test = vsubq_f32(temp_test, temp_train);
				//temp_test = vmulq_f32(temp_test, temp_test);
				//sum = vaddq_f32(sum, temp_test);
				sum = vmlaq_f32(sum, temp_test, temp_test);
			}
			float32x2_t sumlow = vget_low_f32(sum);
			float32x2_t sumhigh = vget_high_f32(sum);
			sumlow = vpadd_f32(sumlow, sumhigh);
			float32_t sumlh = vpadds_f32(sumlow);
			dist[i][j] = (float)sumlh;
		}
		for (int j = 0;j < trainNum;j += 4)
		{
			float32x4_t temp_dist = vld1q_f32(&dist[i][j]);
			temp_dist = vsqrtq_f32(temp_dist);
			vst1q_f32(&dist[i][j], temp_dist);
		}
	}
}

void square_unwrapped()
{
	float temp_test[testNum];
	float temp_train[trainNum];
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

	// pthread_t train_h, test_h;
	// pthread_create(&train_h, NULL, train_square, (void*)temp_train);
	// pthread_create(&test_h, NULL, test_square, (void*)temp_test);
	// pthread_join(train_h, NULL);
	// pthread_join(test_h, NULL);

    #pragma omp parallel for num_threads(NUM_THREADS)
	for(int i = 0;i < testNum;i++)
        #pragma omp parallel for num_threads(NUM_THREADS)
		for (int j = 0;j < trainNum;j++)
		{
			float sum = 0;
			for (int k = 0;k < dim;k++)
				sum += test[i][k] * train[j][k];
			dist[i][j] = sqrtf(temp_test[i] + temp_train[j] - 2 * sum);
		}
}

void square_unwrapped_NEON()
{
	float temp_test[testNum];
	float temp_train[trainNum];
	assert(dim % 4 == 0);//假定维数为4的倍数
    #pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0;i < testNum;i++)
	{
		float32x4_t sum = vmovq_n_f32(0);
		for (int j = 0;j < dim;j += 4)
		{
			float32x4_t square = vld1q_f32(&test[i][j]);
			sum = vmlaq_f32(sum, square, square);
		}
		float32x2_t sumlow = vget_low_f32(sum);
		float32x2_t sumhigh = vget_high_f32(sum);
		sumlow = vpadd_f32(sumlow, sumhigh);
		float32_t sumlh = vpadds_f32(sumlow);
		temp_test[i] = (float)sumlh;
	}
    #pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0;i < trainNum;i++)
	{
		float32x4_t sum = vmovq_n_f32(0);
		for (int j = 0;j < dim;j += 4)
		{
			float32x4_t square = vld1q_f32(&train[i][j]);
			sum = vmlaq_f32(sum, square, square);
		}
		float32x2_t sumlow = vget_low_f32(sum);
		float32x2_t sumhigh = vget_high_f32(sum);
		sumlow = vpadd_f32(sumlow, sumhigh);
		float32_t sumlh = vpadds_f32(sumlow);
		temp_train[i] = (float)sumlh;
	}
    #pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0;i < testNum;i++)
	{
        #pragma omp parallel for num_threads(NUM_THREADS)
		for (int j = 0;j < trainNum;j++)
		{
			float32x4_t sum = vmovq_n_f32(0);
			for (int k = 0;k < dim;k += 4)
			{
				float32x4_t _train = vld1q_f32(&train[j][k]);
				float32x4_t _test = vld1q_f32(&test[i][k]);
				_train = vmulq_f32(_train, _test);
				sum = vaddq_f32(_train, sum);
			}
			float32x2_t sumlow = vget_low_f32(sum);
			float32x2_t sumhigh = vget_high_f32(sum);
			sumlow = vpadd_f32(sumlow, sumhigh);
			float32_t sumlh = vpadds_f32(sumlow);
			dist[i][j] = (float)sumlh;
		}
		//dist[i][j] = sqrtf(temp_test[i] + temp_train[j] - 2 * sum);
		float32x4_t _test = vld1q_dup_f32(&temp_test[i]);
		for (int j = 0;j < trainNum;j += 4)
		{
			float32x4_t _train = vld1q_f32(&temp_train[j]);
			float32x4_t res = vld1q_f32(&dist[i][j]);
			res = vmulq_n_f32(res, -2);
			res = vaddq_f32(_train, res);
			res = vaddq_f32(_test, res);
			res = vsqrtq_f32(res);
			vst1q_f32(&dist[i][j], res);
		}
	}
}

void timing(void (*func)())
{
    timeval tv_begin, tv_end;
    int counter(0);
    double time = 0;
    gettimeofday(&tv_begin, 0);
    while(INTERVAL>time)
    {
        func();
        gettimeofday(&tv_end, 0);
        counter++;
        time = ((ll)tv_end.tv_sec - (ll)tv_begin.tv_sec)*1000.0 + ((ll)tv_end.tv_usec - (ll)tv_begin.tv_usec)/1000.0;
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
	float distComp[testNum][trainNum];
	init();
	printf("%s", "朴素算法耗时：");
	timing(plain);
	float error = 0;
	for (int i = 0;i < testNum;i++)
		for (int j = 0;j < trainNum;j++)
			distComp[i][j] = dist[i][j];
	printf("%s", "开方SIMD算法耗时：");
	timing(sqrt_unwrapped);
	for (int i = 0;i < testNum;i++)
		for (int j = 0;j < trainNum;j++)
			error += (distComp[i][j] - dist[i][j]) * (distComp[i][j] - dist[i][j]);
	printf("误差%f\n", error);
	printf("%s", "串行平方展开算法耗时：");
	timing(square_unwrapped);
	error = 0;
	printf("%s", "SIMD平方展开算法耗时：");
	timing(square_unwrapped_NEON);
	for (int i = 0;i < testNum;i++)
		for (int j = 0;j < trainNum;j++)
			error += (distComp[i][j] - dist[i][j]) * (distComp[i][j] - dist[i][j]);
	printf("误差%f\n", error);
}