#include<string.h>
#include <iostream>
using namespace std;
//�ļ�ͷ��ȡ
typedef struct wave_tag
{
    char ChunkID[4];                    // "RIFF"��־
    unsigned int ChunkSize;             // �ļ�����(WAVE�ļ��Ĵ�С, ����ǰ8���ֽ�)
    char Format[4];                     // "WAVE"��־
    char SubChunk1ID[4];                // "fmt "��־
    unsigned int SubChunk1Size;         // �����ֽ�(����)
    unsigned short int AudioFormat;     // ��ʽ���(10HΪPCM��ʽ����������)
    unsigned short int NumChannels;     // ͨ����(������Ϊ1, ˫����Ϊ2)
    unsigned int SampleRate;            // ������(ÿ��������), ��ʾÿ��ͨ���Ĳ����ٶ�
    unsigned int ByteRate;              // ������Ƶ���ݴ�������, ��ֵΪ:ͨ����*ÿ������λ��*ÿ����������λ��/8
    unsigned short int BlockAlign;      // ÿ����������λ��(���ֽ���), ��ֵΪ:ͨ����*ÿ����������λֵ/8
    unsigned short int BitsPerSample;   // ÿ����������λ��, ��ʾÿ�������и�������������λ��.
    //char SubChunk2ID[4];                // ���ݱ��"data"
    //unsigned int SubChunk2Size;         // �������ݵĳ���
} waveft;

//��Ƶ�ļ�������
class Wav
{
public:
    unsigned char* data;//data�д�����Ƶ����Ч����
    float* buffer;//ת������Ч����
    waveft waveformatex;
    int length_buffer;
    int length_wav;//��Ƶ��Ч���ݵĳ���(��������)
    /************************************************************************/
    Wav(const char* path)
    {
        FILE* file;
        fopen_s(&file, path, "rb");
        if (!file)
        {
            fprintf(stderr, "[Load] [%s not found]\n", path);
            return;
        }
        int chunk;
        fread(&waveformatex, sizeof(struct wave_tag), 1, file);
        fread(&chunk, 4, 1, file);  //��ȡdata�ֶ�
        while (chunk != 0x61746164 && fread(&chunk, 4, 1, file) != EOF);
        //ע�⣬�����Ƶʹ�ù�ת�룬�˴���������޷���������
        fread(&length_wav, sizeof(int), 1, file);   //��ȡ���ݳ���
        data = new unsigned char [length_wav];//����length_wav��С�Ŀռ䣬�ռ���׵�ַΪdata
        cout << "�����������" << waveformatex.BitsPerSample << endl;
        cout << "������" <<waveformatex.NumChannels << endl;
        cout << "������"<< waveformatex.SampleRate << endl;
        cout << "��Ƶ���ݳ���" <<length_wav << endl;
        fread(data, length_wav, 1, file);//����Ƶ����Ч���ݴ洢��data�С�
        fclose(file);
        WavToBuffer();//������仰��λ��
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
        case 16://�����ݼ�ʹ�����
            buffer = new float[length_buffer = length_wav / 2];
            //�˴����openMPָ��黮��
            for (int i = 0; i < length_buffer; i++)
            {
                buffer[i] = short((data[2 * i + 1] << 8) | data[2 * i]);
                buffer[i]  = (buffer[i] + 0.5) / 32767.5;       //�˴�ʹ��SIMD
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