
// utils.cpp
//
// g++ -std=c++11 utils.cpp -shared -o utils.dll
//



#include <cstdlib>
#include <cmath>
#include <memory>
#include <iostream>
#include <algorithm>
#include <set>
#include <vector>


using namespace std;



struct POINT {
    int r;
    int c;

    struct cmp {
        bool operator()(const POINT& k1, const POINT& k2) {
            int res = 0;
            if (k1.r > k2.r) {
                res = 1;
            }
            else if (k1.r < k2.r) {
                res = -1;
            }
            else {
                if (k1.c > k2.c) {
                    res = 1;
                }
                else if (k1.c < k2.c) {
                    res = -1;
                }
            }
            return res > 0;
        }
    };
};

struct RECT {
    int top;
    int bottom;
    int left;
    int right;
};



void detect_rectangle(const double* img, int rows, int cols, int start_r, int start_c, set<POINT, POINT::cmp>& visited, int& top, int& bottom, int& left, int& right) {
    vector<POINT> Q;
    Q.reserve(10000);

    top = rows-1,
    bottom = 0,
    left = cols-1,
    right = 0;

    POINT key = {start_r, start_c};
    Q.push_back(key);
    visited.insert(key);

    while (Q.size()) {
        key = Q.back();
        Q.pop_back();

        // left
        if (0 < key.c) {
            POINT p = {key.r, key.c - 1};
            if (0 < img[p.r * cols + p.c]) {
                if (visited.end() == visited.find(p)) {
                    Q.push_back(p);
                    visited.insert(p);
                    
                    if (left > p.c)
                        left = p.c;
                    if (right < p.c)
                        right = p.c;
                }
            }
        } 

        // right
        if (cols > key.c + 1) {
            POINT p = {key.r, key.c + 1};
            if (0 < img[p.r * cols + p.c]) {
                if (visited.end() == visited.find(p)) {
                    Q.push_back(p);
                    visited.insert(p);
                    
                    if (left > p.c)
                        left = p.c;
                    if (right < p.c)
                        right = p.c;
                }
            }
        } 

        // top
        if (0 < key.r) {
            POINT p = {key.r - 1, key.c};
            if (0 < img[p.r * cols + p.c]) {
                if (visited.end() == visited.find(p)) {
                    Q.push_back(p);
                    visited.insert(p);
                    
                    if (top > p.r)
                        top = p.r;
                    if (bottom < p.r)
                        bottom = p.r;
                }
            }
        }
   
        // bottom
        if (rows > key.r + 1) {
            POINT p = {key.r + 1, key.c};
            if (0 < img[p.r * cols + p.c]) {
                if (visited.end() == visited.find(p)) {
                    Q.push_back(p);
                    visited.insert(p);
                    
                    if (top > p.r)
                        top = p.r;
                    if (bottom < p.r)
                        bottom = p.r;
                }
            }
        }   
    }
}






extern "C" {



    void get_statistics(const double* vec, int size, double* mean, double* skewness, double* variance, double* kurtosis) {
        double n, M1, M2, M3, M4;
        double delta, delta_n, delta_n2, term1;
        // init
        n = M1 = M2 = M3 = M4 = 0.;

        for (size_t i = 0; i < size; ++i) {
            // mvsk
            double n1 = n++;
            delta = double(vec[i]) - M1;
            delta_n = delta / n;
            delta_n2 = delta_n * delta_n;
            term1 = delta * delta_n * n1;
            M1 += delta_n;
            M4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3;
            M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2;
            M2 += term1;
        }

        double m = M1;                                 // mean
        double v = M2 / (n - 1.0);                     // variance
        double s = ::sqrt(n) * M3 / ::pow(M2, 1.5);    // skewness
        double k = n * M4 / (M2 * M2) - 3.0;           // kurtosis

    
        if (mean) 
            *mean = m;
        if (skewness)
            *skewness  = s;
        if (variance)
            *variance = v;
        if (kurtosis)
            *kurtosis = k;
    }


    void get_frequencies(const double* frames, 
                         int rows, 
                         int cols, 
                         int frames_num, 
                         double* freq,
                         double MEAN_MUL,
                         double LOW_VAL,
                         double HIGH_VAL
                         ) {
        size_t frame_size = rows * cols;
        const double* prev = frames;

        unique_ptr<double[]> deltas(new double[frame_size]);

        for (int f = 1; f < frames_num; ++f) {
            const double* curr = &prev[frame_size];

            // get delta
            for (int p = 0; p < frame_size; ++p) {
                deltas[p] = (prev[p] - curr[p]) * (prev[p] - curr[p]);
            }

            double mean, std;
            get_statistics(deltas.get(), frame_size, &mean, nullptr, &std, nullptr);

            for (int p = 0; p < frame_size; ++p) {
                double deviation = deltas[p] - mean;
                if (mean * MEAN_MUL <= deviation /*&& deviation < mean * (MEAN_MUL+3)*/) {
                    freq[p] += 1.;
                }
            }
            
            // prepare for the next iteration 
            prev = curr;  
        }

        for (int p = 0; p < frame_size; ++p) {
            if (freq[p] < LOW_VAL || HIGH_VAL < freq[p])
                freq[p] = 0;
        }
    }

    void get_frequencies2(const double* frames, 
                         int rows, 
                         int cols, 
                         int frames_num, 
                         double* freq,
                         double MEAN_MUL,
                         double LOW_VAL,
                         double HIGH_VAL
                         ) {
        size_t frame_size = rows * cols;
        const double* prev = frames;

        unique_ptr<double[]> deltas(new double[frame_size]);

        for (int f = 1; f < frames_num; ++f) {
            const double* curr = &prev[frame_size];

            // get delta
            for (int p = 0; p < frame_size; ++p) {
                double d = sqrt((prev[p] - curr[p]) * (prev[p] - curr[p]));
                if (MEAN_MUL <= (d / prev[p])) {
                    deltas[p] = 1.;
                }
                else {
                    deltas[p] = 0.;
                }
            }

            for (int p = 0; p < frame_size; ++p) {
                freq[p] += deltas[p];
            }
            
            // prepare for the next iteration 
            prev = curr;  
        }

        for (int p = 0; p < frame_size; ++p) {
            if (freq[p] < LOW_VAL || HIGH_VAL < freq[p])
                freq[p] = 0;
        }
    }


    void filter(const double* freq, int rows, int cols, int F, int S, double K, double* new_freq) {
        int new_cols = (cols - F) / S + 1;
        int new_rows = (rows - F) / S + 1;

        double N = F * F;
        double mv;

        for (int c = 0; c < new_cols; ++c) {
            double sum = 0;
            for (int i = 0; i < F; ++i)
                for (int j = 0; j < F; ++j)
                    sum += freq[i * cols + c*S + j]; 

            mv = sum / N;
            if (K < mv) {
                new_freq[c] = mv;
            }
            else {
                new_freq[c] = 0;
            }

            for (int r = 1; r < new_rows; ++r) {
                for (int j = 0; j < F; ++j) {
                    for (int i = 0; i < S; ++i) {
                        sum -= freq[(r*S-(i+1)) * cols + c*S + j];
                        sum += freq[(r*S+(F-(i+1))) * cols + c*S + j];
                    }
                }
                mv = sum / N;
                if (K < mv) {
                    new_freq[r * new_cols + c] = mv;
                }
                else {
                    new_freq[r * new_cols + c] = 0;
                }
            }
        }
    }


    void rects_detect(const double* img, int rows, int cols, void** handle, int* size) {
        set<POINT, POINT::cmp> visited;

        vector<RECT> rects;
        rects.reserve(1000);

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (0 == img[r * cols + c])
                    continue;

                POINT key = {r, c};
                if (visited.end() != visited.find(key))
                    continue;

                int top, bottom, left, right;
                detect_rectangle(img, rows, cols, r, c, visited, top, bottom, left, right);
               
                RECT rect = {top, bottom, left, right};
                rects.push_back(rect); 
            }
        }

        if (0 < rects.size()) {
            int rects_num = rects.size();
            unique_ptr<RECT[]> tmp(new RECT[rects_num]);
 
            for (int r = 0; r < rects_num; ++r) {
                tmp[r] = rects[r];
            }

            visited.clear();
            rects.clear();

            *handle = tmp.release();
            *size = rects_num;
 
        }
        else {
            *handle = nullptr;
            *size = 0;
        }
    }

    void rects_fetch(const RECT* handle, int size, int* dest) {
        for (int r = 0; r < size; ++r) {
            dest[r * 4 + 0] = handle[r].top;
            dest[r * 4 + 1] = handle[r].bottom;
            dest[r * 4 + 2] = handle[r].left;
            dest[r * 4 + 3] = handle[r].right;
        }
    }

    void rects_free(const RECT* handle) {
        delete [] handle;
    }



    void img_resize(const double* src, int rows, int cols, double* dst, int dst_rows, int dst_cols) {
        int dst_vert_count = rows;
        int dst_horz_count = cols;

        int src_vert_count = dst_rows;
        int src_horz_count = dst_cols;

        int dst_r = 0;
        int dst_c = 0;

        int src_r = 0;
        int src_c = 0;


        int src_total = rows * cols;
        int dst_total = dst_rows * dst_cols;


        while ( (dst_r * dst_cols + dst_c) < dst_total && (src_r * cols + src_c) < src_total) {
            // hors
            if (src_horz_count >= dst_horz_count) {
                dst[dst_r * dst_cols + dst_c] += dst_horz_count * src[src_r * cols + src_c];
                src_horz_count -= dst_horz_count;
                dst_horz_count = 0;
            }
            else {
                dst[dst_r * dst_cols + dst_c] += src_horz_count * src[src_r * cols + src_c];
                src_horz_count = 0;
                dst_horz_count -= src_horz_count;
            }

            if (0 == src_horz_count && 0 == dst_horz_count) {
                src_c = (src_c + 1) < cols ? src_c + 1 : 0;
                src_r += src_c == 0 ? 1 : 0;

                dst_c = (dst_c + 1) < dst_cols ? dst_c + 1 : 0;
                dst_r += dst_c == 0 ? 1 : 0;

                src_horz_count = dst_cols;
                dst_horz_count = cols;
            }
            else if (0 == src_horz_count) {
                src_c = (src_c + 1) < cols ? src_c + 1 : 0;
                src_r += src_c == 0 ? 1 : 0;
 
                src_horz_count = dst_cols;
            }
            else if (0 == dst_horz_count) {
                dst_c = (dst_c + 1) < dst_cols ? dst_c + 1 : 0;
                dst_r += dst_c == 0 ? 1 : 0;
 
                dst_horz_count = cols;
            }

            // vert

        }

        for (int i = 0; i < dst_total; ++i) {
            //dst[i] /= src_total;
            dst[i] /= cols;
        }
    }

}
















