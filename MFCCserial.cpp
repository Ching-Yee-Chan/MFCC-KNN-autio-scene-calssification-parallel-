
#include <stdio.h>
#include <memory.h>
#include <iostream>
using namespace std;
typedef struct wave_tag
{
    char ChunkID[4];                    // "RIFF"标志
    unsigned int ChunkSize;             // 文件长度(WAVE文件的大小, 不含前8个字节)
    char Format[4];                     // "WAVE"标志
    char SubChunk1ID[4];                // "fmt "标志
    unsigned int SubChunk1Size;         // 过渡字节(不定)
    unsigned short int AudioFormat;     // 格式类别(10H为PCM格式的声音数据)
    unsigned short int NumChannels;     // 通道数(单声道为1, 双声道为2)
    unsigned int SampleRate;            // 采样率(每秒样本数), 表示每个通道的播放速度
    unsigned int ByteRate;              // 波形音频数据传输速率, 其值为:通道数*每秒数据位数*每样本的数据位数/8
    unsigned short int BlockAlign;      // 每样本的数据位数(按字节算), 其值为:通道数*每样本的数据位值/8
    unsigned short int BitsPerSample;   // 每样本的数据位数, 表示每个声道中各个样本的数据位数.
    //char SubChunk2ID[4];                // 数据标记"data"
    //unsigned int SubChunk2Size;         // 语音数据的长度

} waveft;

class Wav
{
private:
    char* buffer8;
    short* buffer16;
    float* buffer32;

public:
    //bool recording;
    unsigned char* data;//data中储存音频的有效数据
    waveft waveformatex;
    int length_buffer;
    int length_wav;//音频有效数据的长度(采样点数)
    int thislabel;
    /************************************************************************/
    Wav(const char* path)
    {
        //创建对象，并初始化对象的变量值
        memset(&waveformatex, 0, sizeof(waveformatex));
        length_buffer = 0;
        length_wav = 0;

        buffer8 = new char[length_buffer];
        buffer16 = new short[length_buffer];
        buffer32 = new float[length_buffer];
        data = new unsigned char[length_wav];
        thislabel = Load(path);
    }
    /************************************************************************/

    void WavToBuffer()
    {
        switch (waveformatex.BitsPerSample)
        {
        case 8:
            buffer8 = (char*)realloc(buffer8, sizeof(char) * (length_buffer = length_wav));

            for (int i = 0; i < length_buffer; i++)
            {
                buffer8[i] = (char)data[i];
            }
            break;
        case 16:
            buffer16 = (short*)realloc(buffer16, sizeof(short) * (length_buffer = length_wav / 2));

            for (int i = 0; i < length_buffer; i++)
            {
                buffer16[i] = (short)((data[2 * i + 1] << 8) | data[2 * i]);
                if (buffer16[i])
                    int j = 0;
            }
            break;
        case 32:
            buffer32 = (float*)realloc(buffer32, sizeof(float) * (length_buffer = length_wav / 4));

            for (int i = 0; i < length_buffer; i++)
            {
                buffer32[i] = *(float*)&data[4 * i];
            }
            break;
        }
    }

    /************************************************************************/
    int Load(const char* path)
    {
        FILE* file;
        fopen_s(&file, path, "rb");
        if (!file)
        {
            fprintf(stderr, "[Load] [%s not found]\n", path);
            return -1;
        }
        int chunk;
        fread(&waveformatex, sizeof(struct wave_tag), 1, file);
        cout << waveformatex.BitsPerSample << endl;
        cout << waveformatex.NumChannels << endl;
        cout << waveformatex.SampleRate << endl;
        fread(&chunk, 4, 1, file);
        while (chunk != 0x61746164 && fread(&chunk, 4, 1, file) != EOF);
        fread(&length_wav, sizeof(int), 1, file);
        data = (unsigned char*)realloc(data, length_wav);//申请length_wav大小的空间，空间的首地址为data
        cout << length_wav << endl;
        fread(data, length_wav, 1, file);//将音频的有效数据存储到data中。
        fclose(file);
        //if (waveformatex.BitsPerSample != 16 || waveformatex.NumChannels != 2 || waveformatex.SampleRate != 22050)
            //return 1;
        //else
            return 0;
    }
    /************************************************************************/

    double Get_Buffer(int index)
    {
        if (0 <= index && index < length_buffer)
        {
            switch (waveformatex.BitsPerSample)
            {
            case 8:
                return (buffer8[index] + 0.5) / 127.5;
            case 16:
                return (buffer16[index] + 0.5) / 32767.5;
            case 32:
                return buffer32[index];
            }
        }
        return 0;
    }

};

#ifndef MFCC_H
#define MFCC_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
//#include "wav.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
using namespace std;
//DCT的具体实现
void Discrete_Cosine_Transform(int direction, int length, vector<double>& X)
{
    double pi = 3.14159265358979323846;

    vector<double> x(length, 0);

    for (int i = 0; i < length; i++)
    {
        x[i] = X[i];
    }
    for (int k = 0; k < length; k++)
    {
        double sum = 0;

        if (direction == 1)
        {
            for (int n = 0; n < length; n++)
            {
                sum += ((k == 0) ? (sqrt(0.5)) : (1)) * x[n] * cos(pi * (n + 0.5) * k / length);
            }
        }
        else if (direction == -1)
        {
            for (int n = 0; n < length; n++)
            {
                sum += ((n == 0) ? (sqrt(0.5)) : (1)) * x[n] * cos(pi * n * (k + 0.5) / length);
            }
        }
        X[k] = sum * sqrt(2.0 / length);
    }
}
void DCT(int direction, int length, vector<double>& X)
{
    if (direction == 1 || direction == -1)
    {
        Discrete_Cosine_Transform(direction, length, X);
        return;
    }
    //fprintf(stderr, "[DCT], [direction = {-1 (inversed transform), 1 (forward transform)}\n");
}



//FFT的具体实现
void Fast_Fourier_Transform(int direction, int length, vector<double>& Xr, vector<double>& Xi)
{
    int log_length = (int)(log((double)length) / log(2.0));

    double pi = 3.14159265358979323846;

    for (int i = 0, j = 0; i < length; i++, j = 0)
    {
        for (int k = 0; k < log_length; k++)
        {
            j = (j << 1) | (1 & (i >> k));
        }
        if (j < i)
        {
            double t;

            t = Xr[i];
            Xr[i] = Xr[j];
            Xr[j] = t;

            t = Xi[i];
            Xi[i] = Xi[j];
            Xi[j] = t;
        }
    }
    for (int i = 0; i < log_length; i++)
    {
        int L = (int)pow(2.0, i);

        for (int j = 0; j < length - 1; j += 2 * L)
        {
            for (int k = 0; k < L; k++)
            {
                double argument = direction * -pi * k / L;

                double xr = Xr[j + k + L] * cos(argument) - Xi[j + k + L] * sin(argument);
                double xi = Xr[j + k + L] * sin(argument) + Xi[j + k + L] * cos(argument);

                Xr[j + k + L] = Xr[j + k] - xr;
                Xi[j + k + L] = Xi[j + k] - xi;
                Xr[j + k] = Xr[j + k] + xr;
                Xi[j + k] = Xi[j + k] + xi;
            }
        }
    }
    if (direction == -1)
    {
        for (int k = 0; k < length; k++)
        {
            Xr[k] /= length;
            Xi[k] /= length;
        }
    }
}
void FFT(int direction, int length, vector<double>& Xr, vector<double>& Xi)
{
    int log_length = log((double)length) / log(2.0);

    if (direction != 1 && direction != -1)
    {
        //fprintf(stderr, "[FFT], [direction = {-1 (inversed transform), 1 (forward transform)}\n");
        return;
    }
    if (1 << log_length != length)
    {
        //fprintf(stderr, "[FFT], [length must be a power of 2]\n");
        return;
    }
    Fast_Fourier_Transform(direction, length, Xr, Xi);
}


//梅尔频率范围
double Mel_Scale(int direction, double x)
{
    switch (direction)
    {
    case -1:
        return 700.0 * (exp(x / 1125.0) - 1);
    case 1:
        return 1125.0 * log(1 + x / 700.0);
    }
    //fprintf(stderr, "[Mel_Scale], [direction = {-1 (inversed transform), 1 (forward transform)}\n");
    return 0;
}


//获取完整的MFCC特征(无能量值)，包括FFT、取绝对值、Mel滤波、取对数、DCT,最后返回feature_vector[]一帧的特征向量
void MFCC(int length_frame, int length_DFT, int number_coefficients, int number_filterbanks, int sample_rate, vector<double> frame, vector<double>& feature_vector)
{
    double max_Mels_frequency = Mel_Scale(1, sample_rate / 2);//采样频率范围
    double min_Mels_frequency = Mel_Scale(1, 300);
    double interval = (max_Mels_frequency - min_Mels_frequency) / (number_filterbanks + 1);

    //double *filterbank = new double[number_filterbanks];
    vector<double> filterbank(number_filterbanks, 0);
    //double *Xr = new double[length_DFT];
    vector<double> Xr(length_DFT, 0);
    //double *Xi = new double[length_DFT];
    vector<double> Xi(length_DFT, 0);

    for (int i = 0; i < length_DFT; i++)
    {
        Xr[i] = (i < length_frame) ? (frame[i]) : (0);
        Xi[i] = 0;
    }

    //FFT
    FFT(1, length_DFT, Xr, Xi);

    for (int i = 0; i < length_DFT; i++)
    {
        double frequency = (sample_rate / 2) * i / (length_DFT / 2);
        double Mel_frequency = Mel_Scale(1, frequency);
        //取平方值
        double power = (Xr[i] * Xr[i] + Xi[i] * Xi[i]) / length_frame;

        //梅尔滤波
        for (int j = 0; j < number_filterbanks; j++)
        {
            double frequency_boundary[] = { min_Mels_frequency + interval * (j + 0), min_Mels_frequency + interval * (j + 1), min_Mels_frequency + interval * (j + 2) };

            if (frequency_boundary[0] <= Mel_frequency && Mel_frequency <= frequency_boundary[1])
            {
                double lower_frequency = Mel_Scale(-1, frequency_boundary[0]);
                double upper_frequency = Mel_Scale(-1, frequency_boundary[1]);

                filterbank[j] += power * (frequency - lower_frequency) / (upper_frequency - lower_frequency);
            }
            else if (frequency_boundary[1] <= Mel_frequency && Mel_frequency <= frequency_boundary[2])
            {
                double lower_frequency = Mel_Scale(-1, frequency_boundary[1]);
                double upper_frequency = Mel_Scale(-1, frequency_boundary[2]);

                filterbank[j] += power * (upper_frequency - frequency) / (upper_frequency - lower_frequency);
            }
        }
    }
    //取对数
    for (int i = 0; i < number_filterbanks; i++)
    {
        filterbank[i] = log(filterbank[i]);
    }

    //DCT
    DCT(1, number_filterbanks, filterbank);

    //获取MFCC特征向量
    for (int i = 0; i < number_coefficients; i++)
    {
        feature_vector[i] = filterbank[i];
    }
}

int main()//参数是data_length采样点的语音段
{
    int stride = 256;  //步长
    int length_frame = 512; //帧长
    int length_DFT = 512;//傅里叶点数
    int number_coefficients = 26;//离散变换维度，最终得到3*number_coefficients维的特征数据
    int number_filterbanks = 26;//过滤器数量
    //int signal = 0;
    int number_feature_vectors;//该.wav有多少帧
    int nSamplesPerSec;// 采样频率(每秒样本数), 表示每个通道的播放速度

    double pi = 3.14159265358979323846;
    int wi;
    for (wi = 0; wi <= 0; wi++)
    {
        string dir_path = "D://数据//作业//并行//myMFCC//myMFCC//";
        stringstream strs;
        strs << wi;
        string strg = strs.str();//将int字符类型转换成string类型
        string addr = dir_path + strg + ".wav";
        Wav wav(addr.c_str());
        //Wav wav(data,data_length,samsize);
        wav.WavToBuffer();
        if (wav.thislabel == 1)
            break;
        nSamplesPerSec = wav.waveformatex.SampleRate;//类对象的成员变量的结构体成员变量
        number_feature_vectors = (wav.length_buffer - length_frame) / stride + 1;
        cout << number_feature_vectors << endl;
        vector<vector<double> > feature_vector(number_feature_vectors, vector<double>(3 * number_coefficients, 0));

        // MFCC
        for (int i = 0; i <= wav.length_buffer - length_frame; i += stride)
        {
            //double *frame = new double[length_frame];
            vector<double> frame(length_frame, 0);

            // pre-emphasis，预加重
            for (int j = 0; j < length_frame; j++)
            {
                if (i + j < wav.length_buffer)
                {
                    frame[j] = wav.Get_Buffer(i + j) - 0.95 * wav.Get_Buffer(i + j - 1);
                }
                else
                {
                    frame[j] = 0;
                    //frame[j] = 0;//另外一种方法
                }
            }

            // windowing，加汉明窗
            for (int j = 0; j < length_frame; j++)
            {
                frame[j] *= 0.54 - 0.46 * cos(2 * pi * j / (length_frame - 1));
            }

            MFCC(length_frame, length_DFT, number_coefficients, number_filterbanks, nSamplesPerSec, frame, feature_vector[i / stride]);//进行处理的是第i/stride帧，每帧长length_frame
        }

        //至此得到二维特征向量feature_vector

        // deltas，一阶差分
        for (int i = 0; i < number_feature_vectors; i++)
        {
            int prev = (i == 0) ? (0) : (i - 1);
            int next = (i == number_feature_vectors - 1) ? (number_feature_vectors - 1) : (i + 1);

            for (int j = 0; j < number_coefficients; j++)
            {
                feature_vector[i][number_coefficients + j] = (feature_vector[next][j] - feature_vector[prev][j]) / 2;
            }
        }

        // delta-deltas，二阶差分
        for (int i = 0; i < number_feature_vectors; i++)
        {
            int prev = (i == 0) ? (0) : (i - 1);
            int next = (i == number_feature_vectors - 1) ? (number_feature_vectors - 1) : (i + 1);

            for (int j = number_coefficients; j < 2 * number_coefficients; j++)
            {
                feature_vector[i][number_coefficients + j] = (feature_vector[next][j] - feature_vector[prev][j]) / 2;
            }
        }
        string waddr = dir_path + strg + ".txt";
        FILE* file;
        fopen_s(&file, waddr.c_str(), "wt");

        //将.wav的MFCC特征写入到文件中，每帧一行。每行39维数据。
        for (int i = 0; i < number_feature_vectors; i++)
        {
            for (int j = 0; j < 3 * number_coefficients; j++)
            {
                fprintf(file, "%lf ", feature_vector[i][j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
    }
    //signal = judge(feature_vector,number_feature_vectors,3 * number_coefficients);//if 1 : abnormal ; if 0 : normal
    //citer++;
    //cout<<citer<<endl;
    //return signal;
}

#endif // MFCC_H