#include <iostream>
#include <math.h>

using namespace std;

const int N = (int)1e4;
const int KConst = (int)1e2;

double X[N][KConst];                // случайная величина
int mean, variance;                 // параметры: матожидание, дисперсия
int nX;                             // размер выборки случайной величины
int K;                              // кол-во смесей

int main(int argc, const char * argv[]) {
    srand((unsigned int)time(0));
    
    // вспомогательные переменные
    double R[N];                            // базовая случайная величина [0 1]
    unsigned int a, m;                      // параметры для генерации базовой случ. вел.
    double s = 0;                           // вспомогательная случайная величина
    int nR;                                 // размер выборки базовой случайной величины

    nR = 12;
    nX = 10;
    a = 9, m = 13;
    K = 4;
    
    // для каждой смеси
    for (int k = 0; k < K; k++) {
        
        // инициализируем начальные значения
        R[0] = (double)(rand())/RAND_MAX;
        mean = rand() % nX + 1;
        variance = rand() % nX + 1;
        
        // требуется генерация нескольких сл. базовых чисел R для получения одного нормального сл. значения
        for (int j = 0; j < nX; j++) { // для каждого значения X[j][]
            
            for (int i = 0; i < nR; i++) {
                // Lehmer multiplicative congruential generator (MCG). R - принадлежит отрезку [0 1]
                R[i + 1] = fmod(fmod(a * R[i], m), 1);
            }
            
            // Ниже - Моделирование нормальной случайной величины на основе центральной предельной теоремы
            for (int i = 0; i < nR; i++) {
                // случайная величина s ввиде суммы базовых чисел
                s += R[i];
            }
            
            // генерируем одно случайное число для каждой итерации
            X[j][k] = mean + sqrt(variance) / (sqrt(nR / 12)) * (s - nR / 2);
            
            // в следующей итерации j: R[0] отлично от R[0] предыдущей итерации
            R[0] = R[nR - 1];
            
            // обнуляем для корректной работы в след. итерации j
            s = 0;
        }
    }
    
    for (int i = 0; i < nX; i++) {
        for (int k = 0; k < K; k++) {
            cout << X[i][k] << " ";
        }
        cout << endl;
    }
    
    return 0;
}
