#include<string>
#include<cmath>
#include <sys/time.h>
#include <arm_neon.h>
#include <algorithm>
//#include<mpi.h>
#include"wav_arm.h"
#define TRAINNUM 1
#define INTERVAL 10000
using namespace std;
typedef long long ll;
const double pi = 3.14159265358979323846;
int stride = 256;  //步长
int length_frame = 512; //帧长，由于要做傅里叶变换，必须为2的整数次幂
int log_length = 9;
const int number_filterbanks = 26;//过滤器数量，最终得到3*number_filterbanks维的特征数据

timeval tv_begin, tv_end;
double _time = 0;
int counter = 0;

void FFT(int length, float* Xr, double* Xi)
{
	//int log_length = (int)(log((double)length) / log(2.0));
	//此处使用openMP进行并行化，记得加锁！
	//#pragma omp parallel for num_threads(4)
	for (int i = 0; i < length; i++)
	{
		int j = 0;
		for (int k = 0; k < log_length; k++)
		{
			j = (j << 1) | (1 & (i >> k));
		}
		if (j < i)
		{
			swap(Xr[i], Xr[j]);
			swap(Xi[i], Xi[j]);
		}
	}
	for (int i = 0; i < log_length; i++)
	{
		int L = (int)pow(2.0, i);
		for (int j = 0; j < length - 1; j += 2 * L)
		{
			for (int k = 0; k < L; k++)
			{
				double argument = -pi * k / L;
				double xr = Xr[j + k + L] * cos(argument) - Xi[j + k + L] * sin(argument);
				double xi = Xr[j + k + L] * sin(argument) + Xi[j + k + L] * cos(argument);

				Xr[j + k + L] = Xr[j + k] - xr;
				Xi[j + k + L] = Xi[j + k] - xi;
				Xr[j + k] = Xr[j + k] + xr;
				Xi[j + k] = Xi[j + k] + xi;
			}
		}
	}
}

void FFTSerial(int length, float* Xr, double* Xi)
{
	//int log_length = (int)(log((double)length) / log(2.0));
	int* rev = new int[length];
	rev[0] = 0;
	for (int i = 0; i < length; i++)
	{
		//此处去除一个循环，但会导致上面的循环无法展开
		rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (log_length - 1));
		if (i < rev[i]) 
		{
			swap(Xr[i], Xr[rev[i]]);
			swap(Xi[i], Xi[rev[i]]);
		}
	}
	for (int mid = 1; mid < length; mid <<= 1) 
	{
		double tmpR = cos(pi / mid);
		double tmpI = sin(pi / mid);
		for (int i = 0; i < length; i += mid * 2) 
		{
			double omegaR = 1;
			double omegaI = 0;
			for (int j = 0; j < mid; ++j) 
			{
				double yr = Xr[i + j + mid] * omegaR - Xi[i + j + mid] * omegaI;
				double yi = Xi[i + j + mid] * omegaR + Xr[i + j + mid] * omegaI;
				Xr[i + j + mid] = Xr[i + j] - yr;
				Xi[i + j + mid] = Xi[i + j] - yi;
				Xr[i + j] += yr;
				Xi[i + j] += yi;
				//omega *= tmp
				double omegaR_temp = omegaR * tmpR - omegaI * tmpI;
				omegaI = omegaR * tmpI + omegaI * tmpR;
				omegaR = omegaR_temp;
			}
		}
	}
}

//梅尔频率范围
//direction = 1将实际频率转换为梅尔频率，direction = -1将梅尔频率转换为实际频率
double Mel_Scale(int direction, double x)
{
	switch (direction)
	{
	case -1:
		return 700.0 * (exp(x / 1125.0) - 1);
	case 1:
		return 1125.0 * log(1 + x / 700.0);
	}
	return 0;
}

//离散余弦变换（正向）
void DCT(int length, float* X)
{
	float* temp = new float[length];
	//openMP
	//#pragma omp parallel for num_threads(6)
	for (int k = 0; k < length; k++)
	{
		double sum = 0;
		//SIMD
		for (int n = 0; n < length; n++)
		{
			sum += ((k == 0) ? (sqrt(0.5)) : (1)) * X[n] * cos(pi * (n + 0.5) * k / length);
		}
		temp[k] = sum * sqrt(2.0 / length);
	}
	temp[0] *= sqrt(0.5);
	memcpy(X, temp, length * sizeof(float));
	delete[] temp;
}

void processClip(
	float* data, 
	float(*feature_vector)[number_filterbanks * 3], 
	float* hammingWindow, 
	int length_buffer, 
	int nSamplesPerSec, 
	int number_feature_vectors)
{
	//第0步：准备梅尔滤波器组
	double max_Mels_frequency = Mel_Scale(1, nSamplesPerSec / 2);//频率上限
	double min_Mels_frequency = Mel_Scale(1, 300);//频率下限
	double interval = (max_Mels_frequency - min_Mels_frequency) / (number_filterbanks + 1);//间隔数量
	//串行使用
	//float* frequency_boundary = new float[number_filterbanks + 2];//滤波器边界值
	float* actual_boundary = new float[number_filterbanks + 2];//实际滤波器边界频率
	//这里可以添加SIMD，但意义似乎不大？
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < number_filterbanks + 2; i++)
	{
		actual_boundary[i] = min_Mels_frequency + interval * i;
		actual_boundary[i] = Mel_Scale(-1, actual_boundary[i]);
	}

	//SSE
	// int i = 0;
	// float32x4_t intv_vector = vmovq_n_f32(interval);
	// float32x4_t minMel_vector = vmovq_n_f32(min_Mels_frequency);
	// for (;i < number_filterbanks + 2 && ((number_filterbanks + 2 - i) & 3);++i)
	// 	actual_boundary[i] = min_Mels_frequency + interval * i;
	// for (;i < number_filterbanks + 2;i += 4)
	// {
	// 	float32x4_t i_vector;
	// 	vsetq_lane_f32(i + 3, i_vector, 0);
	// 	vsetq_lane_f32(i + 2, i_vector, 1);
	// 	vsetq_lane_f32(i + 1, i_vector, 2);
	// 	vsetq_lane_f32(i, i_vector, 3);
	// 	float32x4_t adder2 = vmulq_f32(intv_vector, i_vector);
	// 	adder2 = vaddq_f32(minMel_vector, adder2);
	// 	vst1q_f32(&actual_boundary[i], adder2);
	// }

	// for(int m = 0;m<number_filterbanks + 2;m++)
	// 	actual_boundary[m] = Mel_Scale(-1, actual_boundary[m]);

	//在此处添加openMP指令
	//#pragma omp parallel num_threads(8)
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < length_buffer - length_frame; i += stride)
	{
		float* frame = new float[length_frame];
		float* filterbank = feature_vector[i / stride];//滤波结果
		memset(filterbank, 0, number_filterbanks * 3 * sizeof(float));
		//第一步：预加重、加汉明窗、补零
		//#pragma omp parallel for num_threads(8)
		//for (int j = 0; j < length_frame; j++)
		//{
		//	//SIMD
		//	if (i + j < length_buffer && i + j > 0)
		//	{
		//		frame[j] = data[i + j] - 0.95 * data[i + j - 1];//预加重因子γ=0.95
		//		frame[j] *= hammingWindow[j];
		//	}
		//	else if (i + j == 0)
		//	{
		//		frame[j] = data[i + j] * hammingWindow[j];
		//	}
		//	else
		//	{
		//		frame[j] = 0;
		//	}
		//}
		
		for (int j = 0; j < length_frame; j += 4)
		{
			float32x4_t front = vld1q_f32(&data[i + j]);
			float32x4_t back;
			if (!i && !j)
			{
				vsetq_lane_f32(data[2], back, 3);
				vsetq_lane_f32(data[1], back, 2);
				vsetq_lane_f32(data[0], back, 1);
				vsetq_lane_f32(0, back, 0);
			}
			else back = vld1q_f32(&data[i + j - 1]);
			float32x4_t mult = vmovq_n_f32(0.95);//预加重因子γ
			float32x4_t hamming = vld1q_f32(&hammingWindow[j]);
			back = vmulq_f32(mult, back);
			front = vsubq_f32(front, back);
			front = vmulq_f32(front, hamming);
			vst1q_f32(&frame[j], front);
		}
		//第二步：FFT
		double* Xi = new double[length_frame];//虚部
		memset(Xi, 0, sizeof(double) * length_frame);
		FFTSerial(length_frame, frame, Xi);
		//第三步：功率谱、梅尔频率及梅尔滤波
		//注意：如果使用双声道数据，此处应改为i<length_frame / 2 + 1
		//考虑使用SIMD+二分查找，但应该会增加内存IO开销
		#pragma omp parallel for num_threads(8)
		for (int i = 0; i < length_frame; i++)
		{
			double power = (frame[i] * frame[i] + Xi[i] * Xi[i]) / length_frame;//功率谱
			double frequency = (nSamplesPerSec / 2) * i / (length_frame / 2);
			/*double Mel_frequency = Mel_Scale(1, frequency);
			for (int j = 0; j < number_filterbanks; j++)
			{
				if (frequency_boundary[j] < Mel_frequency && Mel_frequency <= frequency_boundary[j + 1])
				{
					double lower_frequency = Mel_Scale(-1, frequency_boundary[j]);
					double upper_frequency = Mel_Scale(-1, frequency_boundary[j + 1]);

					filterbank[j] += power * (frequency - lower_frequency) / (upper_frequency - lower_frequency);
				}
				else if (frequency_boundary[j + 1] <= Mel_frequency && Mel_frequency < frequency_boundary[j + 2])
				{
					double lower_frequency = Mel_Scale(-1, frequency_boundary[j + 1]);
					double upper_frequency = Mel_Scale(-1, frequency_boundary[j + 2]);

					filterbank[j] += power * (upper_frequency - frequency) / (upper_frequency - lower_frequency);
				}
			}*/

			float* upper_p = lower_bound(actual_boundary, actual_boundary + number_filterbanks + 2, frequency);
			int end_sort = upper_p - actual_boundary;//区间终点编号
			int start_sort = end_sort - 1;//区间起点编号
			if (start_sort >= 0 && end_sort < number_filterbanks + 1)//不是最后一个区间及其以后，通过后一个滤波器上升沿
				filterbank[start_sort] += power * (frequency - actual_boundary[start_sort]) / (actual_boundary[end_sort] - actual_boundary[start_sort]);
			if (start_sort >= 1 && end_sort < number_filterbanks + 2)//不是第一个区间及其以前，通过前一个滤波器下降沿
				filterbank[start_sort - 1] += power * (actual_boundary[end_sort] - frequency) / (actual_boundary[end_sort] - actual_boundary[start_sort]);
		}
		//取对数，SIMD
		#pragma omp parallel for num_threads(8)
		for (int i = 0; i < number_filterbanks; i++)
		{
			filterbank[i] = log(filterbank[i]);
		}
		//第四步：离散余弦变换
		DCT(number_filterbanks, filterbank);
		delete[] frame;
		delete[] Xi;
	}
	//逐帧处理完毕，此处必须同步
	//第五步：动态特征提取：一阶/二阶差分
	// deltas，一阶差分
	#pragma omp parallel num_threads(8)
	{
		//此处添加openMP
		#pragma omp for //num_threads(5)
		for (int i = 0; i < number_feature_vectors; i++)
		{
			int prev = (i == 0) ? (0) : (i - 1);
			int next = (i == number_feature_vectors - 1) ? (number_feature_vectors - 1) : (i + 1);
			//此处添加SIMD
			//for (int j = 0; j < number_filterbanks; j++)
			//{
			//	feature_vector[i][number_filterbanks + j] = (feature_vector[next][j] - feature_vector[prev][j]) / 2;
			//}

			int j = 0;
			float32x4_t div = vmovq_n_f32(2);
			for (;(number_filterbanks - j) & 3;++j)
				feature_vector[i][number_filterbanks + j] = (feature_vector[next][j] - feature_vector[prev][j]) / 2;
			for (;j < number_filterbanks;j += 4)
			{
				float32x4_t temp1 = vld1q_f32(&feature_vector[next][j]);
				float32x4_t temp2 = vld1q_f32(&feature_vector[prev][j]);
				temp1 = vsubq_f32(temp1, temp2);
				temp1 = vdivq_f32(temp1, div);
				vst1q_f32(&feature_vector[i][number_filterbanks + j], temp1);
			}
		}

		// delta-deltas，二阶差分
		//此处添加openMP
		#pragma omp for //num_threads(5)
		for (int i = 0; i < number_feature_vectors; i++)
		{
			int prev = (i == 0) ? (0) : (i - 1);
			int next = (i == number_feature_vectors - 1) ? (number_feature_vectors - 1) : (i + 1);
			//此处添加SIMD
			/*for (int j = number_filterbanks; j < 2 * number_filterbanks; j++)
			{
				feature_vector[i][number_filterbanks + j] = (feature_vector[next][j] - feature_vector[prev][j]) / 2;
			}*/

			int j = number_filterbanks;
			float32x4_t div = vmovq_n_f32(2);
			for (;(2 * number_filterbanks - j) & 3;++j)
				feature_vector[i][number_filterbanks + j] = (feature_vector[next][j] - feature_vector[prev][j]) / 2;
			for (;j < 2 * number_filterbanks;j += 4)
			{
				float32x4_t temp1 = vld1q_f32(&feature_vector[next][j]);
				float32x4_t temp2 = vld1q_f32(&feature_vector[prev][j]);
				temp1 = vsubq_f32(temp1, temp2);
				temp1 = vdivq_f32(temp1, div);
				vst1q_f32(&feature_vector[i][number_filterbanks + j], temp1);
			}
		}
	}
	delete[] actual_boundary;
}

int main()
{
	gettimeofday(&tv_begin, 0);

	int comm_sz;
	int my_rank;
	int number_feature_vectors[TRAINNUM];//每段音频的特征数量
	//MPI_Init(NULL, NULL);
	//MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	//MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	//if (my_rank == 0)
	//{
		float* hammingWindow = new float[length_frame];
		//此处使用openMP或SIMD
		#pragma omp parallel for num_threads(8)
		for (int j = 0; j < length_frame; j++)
		{
			hammingWindow[j] = 0.54 - 0.46 * cos(2 * pi * j / (length_frame - 1));
		}

		// //此处添加openMP指令，限制线程数
		//for (int n = 0;n < TRAINNUM;n++)
		//{
			int n = 0;
			string dir_path = "";
			string addr = dir_path + to_string(n) + ".wav";
			Wav wav(addr.c_str());
			number_feature_vectors[n] = (wav.length_buffer - length_frame) / stride + 1;
			//此处添加buffer、args、hammingWindow的发语句
			//此处添加feature_vector的收语句
		//}
	//}
	//else
	//{
		float* data = wav.buffer;
		int length_buffer = wav.length_buffer;
		int nSamplesPerSec = wav.waveformatex.SampleRate;
		int numberFeatureVectors = number_feature_vectors[0];
		//此处添加buffer、args、hammingWindow的收语句
		float(*feature_vector)[number_filterbanks * 3] = new float[numberFeatureVectors][number_filterbanks * 3];
		processClip(data, feature_vector, hammingWindow,  length_buffer, nSamplesPerSec, numberFeatureVectors);
	//}
		gettimeofday(&tv_end, 0);
		string waddr = dir_path + to_string(n) + ".txt";
		FILE* file = fopen(waddr.c_str(), "wt");

		//将.wav的MFCC特征写入到文件中，每帧一行。每行78维数据。
		for (int i = 0; i < number_feature_vectors[n]; i++)
		{
			for (int j = 0; j < 3 * number_filterbanks; j++)
			{
				fprintf(file, "%f ", feature_vector[i][j]);
				//printf("%f\n", feature_vector[i][j]);
			}
			fprintf(file, "\n");
		}
		fclose(file);
		delete[] hammingWindow;
		delete[] feature_vector;
		//_time /= counter;
        _time += ((ll)tv_end.tv_sec - (ll)tv_begin.tv_sec)*1000.0 + ((ll)tv_end.tv_usec - (ll)tv_begin.tv_usec)/1000.0;
    	cout<<_time<<'\t';
}