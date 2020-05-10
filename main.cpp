#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <memory>
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

const int BMP_HEADER_END = 0x8a;
const int BYTES_PER_PIXEL = 4;
const unsigned char MAX_ALPHA = 255;
const int MAX_ALPHA_POW = 8;


class FileCloser {
    public:
        FileCloser() = default;
        ~FileCloser() = default;
        void operator()(FILE* file_pointer) {
            if(file_pointer) {
                fclose(file_pointer);
            }
        }
};

class BMPFile {
    private:
        unsigned char* data_ = nullptr;
        long long size_ = 0;

        void ClearData() {
            if(data_) {
                delete[] data_;
            }
        }

    public:
        BMPFile() noexcept : data_(nullptr), size_(0){}

        BMPFile(const BMPFile& other) {
            ClearData();
            size_ = other.size_;
            data_ = new unsigned char[other.size_]();
            memcpy(data_, other.data_, size_);   
        }

        BMPFile& operator=(const BMPFile& other) {
            ClearData();
            size_ = other.size_;
            data_ = new unsigned char[other.size_]();
            memcpy(data_, other.data_, size_);
            
            return *this;
        }

        BMPFile(BMPFile&& other) noexcept {
            std::swap(*this, other);
        }

        BMPFile& operator=(BMPFile&& other) noexcept {
            std::swap(*this, other);
            return *this;
        }

        ~BMPFile() {
            ClearData();
        }

        BMPFile(const char* filename) {
            auto bmp_file = std::unique_ptr<FILE, FileCloser>(fopen(filename, "r"), FileCloser());

            if(!bmp_file.get()) {
                throw std::runtime_error("This file does not exist!");
            }
            
            fseek(bmp_file.get(), 0, SEEK_END);
            size_ = ftell(bmp_file.get());
            fseek(bmp_file.get(), 0, SEEK_SET);

            data_ = new unsigned char[size_]();
            fread(data_, sizeof(unsigned char), size_, bmp_file.get());
        }

        long long Size() const noexcept {
            return size_;
        }

        const unsigned char* Data() const noexcept {
            return data_;
        }

        void SaveToFile(const char* filename) {
            auto bmp_file = std::unique_ptr<FILE, FileCloser>(fopen(filename, "w"), FileCloser());
            fwrite(data_, sizeof(unsigned char), size_, bmp_file.get());
        }

        void ComposeAlpha(const BMPFile& other) {
            if(other.size_ != size_){
                throw std::runtime_error("Both pictures must be the same size!");
            }

            for(int i = BMP_HEADER_END; i < size_; i += BYTES_PER_PIXEL){
                unsigned char src_alpha = other.data_[i + 3];

                int* dest_pixel_pointer = reinterpret_cast<int*>(data_ + i);
                __m128i dest_colors = _mm_cvtsi32_si128(*dest_pixel_pointer);     //loaded pixel data to first 32 bits of vector
                int src_pixel = *(reinterpret_cast<int*>(other.data_ + i));
                __m128i src_colors = _mm_cvtsi32_si128(src_pixel); 

                __m128i src_alpha_vec = _mm_set1_epi32(src_alpha); //load vector of 4 instances of 8-bit alpha
                __m128i alpha_vec = _mm_set1_epi32(MAX_ALPHA - src_alpha);

                //mask for rearraging RGB pixels data
                __m128i mask = _mm_setr_epi8(0, 0x80, 0x80, 0x80, 1, 0x80, 0x80, 0x80, 2, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80); 
                dest_colors = _mm_shuffle_epi8(dest_colors, mask);
                src_colors = _mm_shuffle_epi8(src_colors, mask); // rearranging rgb data


                dest_colors = _mm_mullo_epi32(dest_colors, alpha_vec); //applying alpha to dest
                src_colors = _mm_mullo_epi32(src_colors, src_alpha_vec); //applying alpha to src
                src_colors = _mm_add_epi32(src_colors, dest_colors);

                __m128i shift =  _mm_setr_epi32(MAX_ALPHA_POW, 0, 0, 0); 
                src_colors = _mm_sra_epi32(src_colors, shift);

                __m128i rev_mask = _mm_setr_epi8(0, 4, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
                src_colors = _mm_shuffle_epi8(src_colors, rev_mask);

                *dest_pixel_pointer = _mm_cvtsi128_si32(src_colors);
                data_[i + 3] = MAX_ALPHA;
            }
        }

        friend void swap(BMPFile& first, BMPFile& second) noexcept {
            std::swap(first.data_, second.data_);
            std::swap(first.size_, second.size_);
        }
    
};

int main() {
    auto cat_file = BMPFile("pictures/cat.bmp");
    auto book_file = BMPFile("pictures/book.bmp");
    cat_file.ComposeAlpha(book_file);
    cat_file.SaveToFile("pictures/composed.bmp");
    return 0;
}
