#include<string.h>
#include <iostream>
using namespace std;
//文件头读取
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

//音频文件处理类
class Wav
{
public:
    unsigned char* data;//data中储存音频的有效数据
    float* buffer;//转码后的有效数据
    waveft waveformatex;
    int length_buffer;
    int length_wav;//音频有效数据的长度(采样点数)
    /************************************************************************/
    Wav(const char* path)
    {
        FILE* file = fopen(path, "rb");
        if (!file)
        {
            fprintf(stderr, "[Load] [%s not found]\n", path);
            return;
        }
        int chunk;
        fread(&waveformatex, sizeof(struct wave_tag), 1, file);
        fread(&chunk, 4, 1, file);  //读取data字段
        while (chunk != 0x61746164 && fread(&chunk, 4, 1, file) != EOF);
        //注意，如果音频使用过转码，此处将会出现无法对齐的情况
        fread(&length_wav, sizeof(int), 1, file);   //读取数据长度
        data = new unsigned char [length_wav];//申请length_wav大小的空间，空间的首地址为data
        cout << "采样点比特数" << waveformatex.BitsPerSample << endl;
        cout << "声道数" <<waveformatex.NumChannels << endl;
        cout << "采样率"<< waveformatex.SampleRate << endl;
        cout << "音频数据长度" <<length_wav << endl;
        fread(data, length_wav, 1, file);//将音频的有效数据存储到data中。
        fclose(file);
        WavToBuffer();//考虑这句话的位置
    }
    /************************************************************************/
    void WavToBuffer()
    {
        switch (waveformatex.BitsPerSample)
        {
        case 8:
            buffer = new float[length_buffer = length_wav];

            for (int i = 0; i < length_buffer; i++)
            {
                buffer[i] = (float)data[i] + 0.5 / 127.5;
            }
            break;
        case 16://本数据集使用这个
            buffer = new float[length_buffer = length_wav / 2];
            //此处添加openMP指令，块划分
            for (int i = 0; i < length_buffer; i++)
            {
                buffer[i] = short((data[2 * i + 1] << 8) | data[2 * i]);
                buffer[i]  = (buffer[i] + 0.5) / 32767.5;       //此处使用SIMD
            }
            break;
        case 32:
            buffer = new float(length_buffer = length_wav / 4);

            for (int i = 0; i < length_buffer; i++)
            {
                buffer[i] = *(float*)&data[4 * i];
            }
            break;
        }
        delete[] data;
        data = NULL;
    }
    /************************************************************************/
    ~Wav()
    {
        delete[] buffer;
    }
};