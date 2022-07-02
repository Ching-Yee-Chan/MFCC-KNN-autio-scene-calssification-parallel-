#include<string>
#include<windows.h>
#include <xmmintrin.h> //SSE
#include <immintrin.h> //SVML
#include <algorithm>
#include <mpi.h>
#include "wav_multiple.h"
#define TRAINNUM 256
using namespace std;
typedef long long ll;
const double pi = 3.14159265358979323846;
int stride = 256;  //步长
int length_frame = 512; //帧长，由于要做傅里叶变换，必须为2的整数次幂
int log_length = 9;
const int number_filterbanks = 26;//过滤器数量，最终得到3*number_filterbanks维的特征数据

ll head, tail, freq;
double _time = 0;
int counter = 0;

void FFT(int length, float* Xr, float* Xi)
{
	//int log_length = (int)(log((double)length) / log(2.0));
	//此处使用openMP进行并行化，记得加锁！
	//#pragma omp parallel for num_threads(2)
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
			if (L < 4)
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
			else
			{
				//#pragma omp parallel for num_threads(2)
				for (int k = 0; k < L; k += 4)
				{
					__m128 arg = _mm_set_ps(-pi * (k + 3) / L, -pi * (k + 2) / L, -pi * (k + 1) / L, -pi * k / L);
					__m128 argSin = _mm_sin_ps(arg);
					__m128 argCos = _mm_cos_ps(arg);
					__m128 Xr_v = _mm_loadu_ps(Xr + j + k + L);
					__m128 Xi_v = _mm_loadu_ps(Xi + j + k + L);
					__m128 first = _mm_mul_ps(Xr_v, argCos);
					__m128 sec = _mm_mul_ps(Xi_v, argSin);
					__m128 xr_v = _mm_sub_ps(first, sec);
					first = _mm_mul_ps(Xr_v, argSin);
					sec = _mm_mul_ps(Xi_v, argCos);
					__m128 xi_v = _mm_add_ps(first, sec);
					__m128 Xr_front = _mm_loadu_ps(Xr + j + k);
					__m128 Xi_front = _mm_loadu_ps(Xi + j + k);
					__m128 temp_r = _mm_sub_ps(Xr_front, xr_v);
					_mm_storeu_ps(Xr + j + k + L, temp_r);
					__m128 temp_i = _mm_sub_ps(Xi_front, xi_v);
					_mm_storeu_ps(Xi + j + k + L, temp_i);
					temp_r = _mm_add_ps(Xr_front, xr_v);
					_mm_storeu_ps(Xr + j + k, temp_r);
					temp_i = _mm_add_ps(Xi_front, xi_v);
					_mm_storeu_ps(Xi + j + k, temp_i);
				}
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
	//#pragma omp parallel for //num_threads(1)
	for (int k = 0; k < length; k++)
	{
		//double sum = 0;
		//SIMD
		//for (int n = 0; n < length; n++)
		//{
		//	sum += ((k == 0) ? (sqrt(0.5)) : (1)) * X[n] * cos(pi * (n + 0.5) * k / length);
		//}
		//temp[k] = sum * sqrt(2.0 / length);

		int n = 0;
		float tempSum = 0;
		for(;(length-n)&3;++n)
			tempSum += X[n] * cos(pi * (n + 0.5) * k / length);
		__m128 sumVec = _mm_set1_ps(0);
		__m128 c = _mm_set1_ps(pi * k / length);
		for (;n < length;n += 4)
		{
			__m128 temp1 = _mm_loadu_ps(&X[n]);
			__m128 temp2 = _mm_set_ps(n + 3.5, n + 2.5, n + 1.5, n + 0.5);
			temp2 = _mm_mul_ps(c, temp2);
			temp2 = _mm_cos_ps(temp2);
			temp1 = _mm_mul_ps(temp1, temp2);
			sumVec = _mm_add_ps(sumVec, temp1);
		}
		sumVec = _mm_hadd_ps(sumVec, sumVec);
		sumVec = _mm_hadd_ps(sumVec, sumVec);
		_mm_store_ss(&temp[k], sumVec);
		temp[k] += tempSum;
		temp[k] *= sqrt(2.0 / length);
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
	//#pragma omp parallel for //num_threads(8)
	//for (int i = 0; i < number_filterbanks + 2; i++)
	//{
	//	frequency_boundary[i] = min_Mels_frequency + interval * i;
	//}

	//SSE
	int i = 0;
	__m128 intv_vector = _mm_set_ps1(interval);
	__m128 minMel_vector = _mm_set_ps1(min_Mels_frequency);
	for (;i < number_filterbanks + 2 && ((number_filterbanks + 2 - i) & 3);++i)
	{
		float frequency_boundary = min_Mels_frequency + interval * i;
		actual_boundary[i] = Mel_Scale(-1, frequency_boundary);
	}
	for (;i < number_filterbanks + 2;i += 4)
	{
		__m128 i_vector = _mm_set_ps(i + 3, i + 2, i + 1, i);
		__m128 adder2 = _mm_mul_ps(intv_vector, i_vector);
		adder2 = _mm_add_ps(minMel_vector, adder2);
		//_mm_storeu_ps(&frequency_boundary[i], adder2);
		__m128 diver = _mm_set1_ps(1125);
		__m128 subber = _mm_set1_ps(1);
		__m128 mult = _mm_set1_ps(700);
		adder2 = _mm_div_ps(adder2, diver);
		adder2 = _mm_exp_ps(adder2);
		adder2 = _mm_sub_ps(adder2, subber);
		adder2 = _mm_mul_ps(mult, adder2);
		_mm_storeu_ps(&actual_boundary[i], adder2);
	}

	//在此处添加openMP指令
	//#pragma omp parallel num_threads(8)
	#pragma omp parallel for num_threads(4)
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
			__m128 front = _mm_loadu_ps(&data[i + j]);
			__m128 back;
			if (!i && !j)
				back = _mm_set_ps(data[2], data[1], data[0], 0);
			else back = _mm_loadu_ps(&data[i + j - 1]);
			__m128 mult = _mm_set_ps1(0.95);//预加重因子γ
			__m128 hamming = _mm_loadu_ps(&hammingWindow[j]);
			back = _mm_mul_ps(mult, back);
			front = _mm_sub_ps(front, back);
			front = _mm_mul_ps(front, hamming);
			_mm_storeu_ps(&frame[j], front);
		}
		//第二步：FFT
		float* Xi = new float[length_frame];//虚部
		memset(Xi, 0, sizeof(float) * length_frame);
		FFT(length_frame, frame, Xi);
		//第三步：功率谱、梅尔频率及梅尔滤波
		//注意：如果使用双声道数据，此处应改为i<length_frame / 2 + 1
		//考虑使用SIMD+二分查找，但应该会增加内存IO开销
		#pragma omp parallel for num_threads(2)
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
		//#pragma omp parallel for// num_threads(8)
		//for (int i = 0; i < number_filterbanks; i++)
		//{
		//	filterbank[i] = log(filterbank[i]);
		//}

		for (int i = 0;i < number_filterbanks;i+=4)
		{
			__m128 temp = _mm_loadu_ps(&filterbank[i]);
			temp = _mm_log_ps(temp);
			_mm_storeu_ps(&filterbank[i], temp);
		}
		//第四步：离散余弦变换
		DCT(number_filterbanks, filterbank);
		delete[] frame;
		delete[] Xi;
	}
	//逐帧处理完毕，此处必须同步
	//第五步：动态特征提取：一阶/二阶差分
	// deltas，一阶差分
	#pragma omp parallel num_threads(5)
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
			__m128 div = _mm_set1_ps(2);
			for (;(number_filterbanks - j) & 3;++j)
				feature_vector[i][number_filterbanks + j] = (feature_vector[next][j] - feature_vector[prev][j]) / 2;
			for (;j < number_filterbanks;j += 4)
			{
				__m128 temp1 = _mm_loadu_ps(&feature_vector[next][j]);
				__m128 temp2 = _mm_loadu_ps(&feature_vector[prev][j]);
				temp1 = _mm_sub_ps(temp1, temp2);
				temp1 = _mm_div_ps(temp1, div);
				_mm_storeu_ps(&feature_vector[i][number_filterbanks + j], temp1);
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
			__m128 div = _mm_set1_ps(2);
			for (;(2 * number_filterbanks - j) & 3;++j)
				feature_vector[i][number_filterbanks + j] = (feature_vector[next][j] - feature_vector[prev][j]) / 2;
			for (;j < 2 * number_filterbanks;j += 4)
			{
				__m128 temp1 = _mm_loadu_ps(&feature_vector[next][j]);
				__m128 temp2 = _mm_loadu_ps(&feature_vector[prev][j]);
				temp1 = _mm_sub_ps(temp1, temp2);
				temp1 = _mm_div_ps(temp1, div);
				_mm_storeu_ps(&feature_vector[i][number_filterbanks + j], temp1);
			}
		}
	}
	delete[] actual_boundary;
}

int main()
{
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//QueryPerformanceCounter((LARGE_INTEGER*)&tail);

	int comm_sz;
	int my_rank;
	int number_feature_vectors[TRAINNUM];//每段音频的特征数量
	float hammingWindow[512];
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	//float* hammingWindow = new float[length_frame];
	//此处使用openMP或SIMD
	//#pragma omp parallel for num_threads(4)
	//for (int j = 0; j < length_frame; j++)
	//{
	//	hammingWindow[j] = 0.54 - 0.46 * cos(2 * pi * j / (length_frame - 1));
	//}

	__m128 subber = _mm_set1_ps(0.54);
	__m128 subbee = _mm_set1_ps(0.46);
	__m128 mult1 = _mm_set1_ps(2 * pi / (length_frame - 1));
	for (int j = 0; j < length_frame; j += 4)
	{
		__m128 mult2 = _mm_set_ps(j + 3, j + 2, j + 1, j);
		mult2 = _mm_mul_ps(mult1, mult2);
		mult2 = _mm_cos_ps(mult2);
		mult2 = _mm_mul_ps(subbee, mult2);
		mult2 = _mm_sub_ps(subber, mult2);
		_mm_storeu_ps(&hammingWindow[j], mult2);
	}

	if (my_rank == 0)
	{
		//此处添加openMP指令，限制线程数
		int count = 0;
		int label[TRAINNUM];
		for (int n = 0;count < TRAINNUM;n++)
		{
			bool ok;
			string dir_path = "D://数据//作业//并行//final1//final1//0//";
			string addr = dir_path + to_string(n) + ".wav";
			Wav wav(addr.c_str(), ok);
			if (!ok) continue;
			label[count] = n;
			number_feature_vectors[count] = (wav.length_buffer - length_frame) / stride + 1;
			//此处添加buffer、args、hammingWindow的发语句
			int args[3] = { wav.length_buffer , wav.waveformatex.SampleRate , number_feature_vectors[count] };
			MPI_Send(args, 3, MPI_INT, count % (comm_sz - 1) + 1, count + TRAINNUM, MPI_COMM_WORLD);
			MPI_Send(wav.buffer, wav.length_buffer, MPI_FLOAT, count % (comm_sz - 1) + 1, count, MPI_COMM_WORLD);
			count++;
		}
	}
	else
	{
		//float* data = wav.buffer;
		//int length_buffer = wav.length_buffer;
		//int nSamplesPerSec = wav.waveformatex.SampleRate;
		//int numberFeatureVectors = number_feature_vectors[0];
		for (int num = my_rank - 1;num < TRAINNUM;num += comm_sz - 1)
		{
			cout << my_rank << endl;
			//此处添加buffer、args、hammingWindow的收语句
			int args[3];
			MPI_Recv(args, 3, MPI_INT, 0, num + TRAINNUM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			cout << "recerved"<<num<<endl;
			int length_buffer = args[0];
			int nSamplesPerSec = args[1];
			int numberFeatureVectors = args[2];
			float* data = new float[length_buffer];
			MPI_Recv(data, length_buffer, MPI_FLOAT, 0, num, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			cout << "recerved"<<endl;
			float(*feature_vector)[number_filterbanks * 3] = new float[numberFeatureVectors][number_filterbanks * 3];
			processClip(data, feature_vector, hammingWindow, length_buffer, nSamplesPerSec, numberFeatureVectors);
			MPI_Request req;
			//MPI_Isend(feature_vector, numberFeatureVectors * number_filterbanks * 3, MPI_FLOAT, 0, num, MPI_COMM_WORLD, &req);
			//MPI_Send(feature_vector, numberFeatureVectors * number_filterbanks * 3, MPI_FLOAT, 0, num, MPI_COMM_WORLD);
			string outpath = "D://数据//作业//并行//final1//final1//out//";
			string waddr = outpath + to_string(num) + ".txt";
			FILE* file;
			fopen_s(&file, waddr.c_str(), "wt");

			//将.wav的MFCC特征写入到文件中，每帧一行。每行78维数据。
			for (int i = 0; i < numberFeatureVectors; i++)
			{
				for (int j = 0; j < 3 * number_filterbanks; j++)
				{
					fprintf(file, "%f ", feature_vector[i][j]);
				}
				fprintf(file, "\n");
			}
			fclose(file);
			delete[] feature_vector;
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	_time = (double)(tail - head) * 1000.0 / freq;
	std::cout << "节点" << my_rank << "运行完成。共读取" << TRAINNUM << "段音频，耗时" << _time << '\n';
	MPI_Finalize();
	//delete[] hammingWindow;
}