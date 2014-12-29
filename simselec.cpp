#include <iostream>
#include <math.h>
#include <random>
#include <vector>

// симуляция выборки из смеси одномерных (для простоты) нормальных распределений.

using std::cout;
using std::endl;

const int N = (int)1e4;

double X[N];
int mean, variance;                 // параметры: матожидание, дисперсия
int K;                              // кол-во смесей
int T;                              // размер выборки случайной величины

int gcd(int left, int right) {
    return !right ? left : gcd(right, left % right);
}

int lcm(int left, int right) {
    return left * right / gcd(left, right);
}

std::vector<int> getDenominators(std::vector<double> const &weights) {
    std::vector<int> denomenators;
    for (int i = 0; i < weights.size(); i++) {
        denomenators.push_back(1 / weights[i]);
    }
    return denomenators;
}

// Find the LCM of the denominators of the probabilities
int getL(std::vector<int> &denominators) {
    int right, left = 1;
    while (!denominators.empty()) {
        right = denominators.back();
        denominators.pop_back();
        left = lcm(left, right);
    }
    return left;
}

// Simulating a Loaded Die with a Fair Die
int sample_categorical(std::vector<double> const &weights) {
    std::vector<int> denoms = getDenominators(weights);
    
    // Initialization
    // step 1
    int L = getL(denoms);
    
    // step 2
    std::vector<int> A(L);
    
    // step 3
    int nextSideDie = 0;
    for (int i = 0; i < weights.size(); i++) {
        int nextSequence = L * weights[i];
        for (int j = 0; j < nextSequence; j++) {
            A[nextSideDie + j] = i;
        }
        nextSideDie += nextSequence;
    }
    
    // Generation
    // step 1
    srand((unsigned int)time(0));
    int S = rand() % L;
    
    // step 2
    int k = A[S];
    
    return k;
}

double sample_gaussian(int mean, int variance) {
    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    std::normal_distribution<> distribution(mean, variance);
    return distribution(generator);
}

std::vector<double> sample_mixture(std::vector<double> const &weights,    // \pi
                                   std::vector<double> const &means,      // \mu
                                   std::vector<double> const &variances,  // \Sigma
                                   size_t T) {
    std::vector<double> res(T);
    for (size_t t = 0; t < T; ++t) {
//        // кидаем "нечестный" кубик с указанными весами, можете сами подумать
//        // как это сделать, а можете прочитать прекрасную статью по ссылке.
        size_t k = sample_categorical(weights);
//        // генерирует величину из нормального распределения с указанным средним
//        // и дисперсией, рекомендую воспользоваться std::normal_distribution.
        res[t] = sample_gaussian(means[k], variances[k]);
    }

    return res;
}

// Inverse transform sampling
void genSeqOfNormDistr();

int main(int argc, const char * argv[]) {
    srand((unsigned int)time(0));
    
    T = 10;
    K = 3;
    
    double initArr_W[] = {0.2, 0.3, 0.5};
    int    initArr_M[] = {1, 2, 3};
    int    initArr_V[] = {4, 3, 2};
    
    std::vector<double> res(T);    
    std::vector<double> weights(initArr_W, initArr_W + K);
    std::vector<double> means(initArr_M, initArr_M + K);
    std::vector<double> variances(initArr_V, initArr_V + K);
    
    res = sample_mixture(weights, means, variances, T);
    for (int i = 0; i < T; i++) {
        cout << res[i] << "\n";
    }
    
//    genSeqOfNormDistr();
    
    return 0;
}

void genSeqOfNormDistr() {
    srand((unsigned int)time(0));
    T = 10;
    mean = rand() % T + 1;
    variance = rand() % T + 1;
    
    /* Метод обратного преобразования */
    double Y;
    for (int j = 0; j < T; j++) {
        Y = (double)(rand()) / RAND_MAX;
        X[j] = mean + variance * sqrt(abs(-2 * logf(variance * Y * sqrt(2 * M_PI))));
    }
    for (int j = 0; j < T; j++) {
        cout << X[j] << endl;
    }cout << endl;
}
