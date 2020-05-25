#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <memory>
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

const int BMP_FILE_SIZE_OFFSET = 0x2;
const int BMP_FILE_OFFBITS_OFFSET = 0xA;
const int BMP_FILE_WIDTH_OFFSET = 0x12;
const int BMP_FILE_HEIGHT_OFFSET = 0x16;


const int BYTES_PER_PIXEL = 4;
const unsigned char MAX_ALPHA = 255;
const int MAX_ALPHA_POW = 8;

struct BMPHeader {
    unsigned short      bfType              = 0;
    unsigned int        bfSize              = 0;
    unsigned short      bfReserved1         = 0;
    unsigned short      bfReserved2         = 0;
    unsigned int        bfOffBits           = 0;

    unsigned int        bV5Size             = 0;
    unsigned int        bV5Width            = 0;
    unsigned int        bV5Height           = 0;
    unsigned short      bV5Planes           = 0;
    unsigned short      bV5BitCount         = 0;
    unsigned int        biV5Compression     = 0;
    unsigned int        bV5SizeImage        = 0;
    unsigned int        bV5PelsPerMeter     = 0;
    unsigned int        bV5YPelsPerMeter    = 0;
    unsigned int        bV5ClrUsed          = 0;
    unsigned int        bV5ClrImportant     = 0;
};


class BMPFile {
    private:
        std::unique_ptr<unsigned char[]> data_;
        unsigned char* bitmap_;

        int size_ = 0;
        BMPHeader header = {};

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

    public:
        BMPFile() noexcept : size_(0) {};
        ~BMPFile() = default;

        BMPFile(const BMPFile& other) = delete;

        BMPFile& operator=(const BMPFile& other) = delete;

        BMPFile(BMPFile&& other) noexcept {
            std::swap(*this, other);
        }

        BMPFile& operator=(BMPFile&& other) noexcept {
            std::swap(*this, other);
            return *this;
        }

        BMPFile(const char* filename) {
            auto bmp_file = std::unique_ptr<FILE, FileCloser>(fopen(filename, "r"), FileCloser());

            if(!bmp_file.get()) {
                throw std::runtime_error("This file does not exist!");
            }
            fseek(bmp_file.get(), 0, SEEK_END);
            size_ = ftell(bmp_file.get());
            fseek(bmp_file.get(), 0, SEEK_SET);

            data_ = std::unique_ptr<unsigned char[]>(new unsigned char[size_]());
            fread(data_.get(), sizeof(unsigned char), size_, bmp_file.get());

            header = *(reinterpret_cast<BMPHeader*>(data_.get()));
            bitmap_ = data_.get() + header.bfOffBits;
        }

        int Size() const noexcept {
            return size_;
        }

        int Height() const noexcept {
            return header.bV5Height;
        }

        int Width() const noexcept {
            return header.bV5Width;
        }

        const unsigned char* Data() const noexcept {
            return data_.get();
        }

        void SaveToFile(const char* filename) {
            auto bmp_file = std::unique_ptr<FILE, FileCloser>(fopen(filename, "w"), FileCloser());
            fwrite(data_.get(), sizeof(unsigned char), size_, bmp_file.get());
        }

        void ComposeAlpha(const BMPFile& other, int x, int y) {
            if(x + other.Width() > Width() || y + other.Height() > Height()){
                throw std::runtime_error("Argument picture must be smaller than dest!");
            }

            for(int i = 0; i < other.Height() ; ++i) {
                for(int j = 0; j < other.Width(); ++j) {
                    int src_position = i * other.Width() * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL;
                    int dest_position = (y + i) * Width() * BYTES_PER_PIXEL + (x + j) * BYTES_PER_PIXEL;

                     unsigned char src_alpha = other.bitmap_[src_position + 3];

                    int* dest_pixel_pointer = reinterpret_cast<int*>(bitmap_ + dest_position);
                    __m128i dest_colors = _mm_cvtsi32_si128(*dest_pixel_pointer);     //loaded pixel data to first 32 bits of vector
                    int src_pixel       = *(reinterpret_cast<int*>(other.bitmap_ + src_position));
                    __m128i src_colors  = _mm_cvtsi32_si128(src_pixel); 

                    __m128i src_alpha_vec = _mm_set1_epi32(src_alpha); //load vector of 4 instances of 8-bit alpha
                    __m128i alpha_vec     = _mm_set1_epi32(MAX_ALPHA - src_alpha);

                    //mask for rearraging RGB pixels data
                    __m128i mask = _mm_setr_epi8(0, 0x80, 0x80, 0x80, 1, 0x80, 0x80, 0x80, 2, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80); 
                    dest_colors  = _mm_shuffle_epi8(dest_colors, mask);
                    src_colors   = _mm_shuffle_epi8(src_colors, mask); // rearranging rgb data


                    __m128i dest_colors_alpha_applied = _mm_mullo_epi32(dest_colors, alpha_vec); //applying alpha to dest
                    __m128i src_colors_alpha_applied  = _mm_mullo_epi32(src_colors, src_alpha_vec); //applying alpha to src
                    __m128i res_colors                = _mm_add_epi32(src_colors_alpha_applied, dest_colors_alpha_applied);

                    __m128i shift =  _mm_setr_epi32(MAX_ALPHA_POW, 0, 0, 0); 
                    res_colors    = _mm_sra_epi32(res_colors, shift);

                    __m128i rev_mask = _mm_setr_epi8(0, 4, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
                    res_colors       = _mm_shuffle_epi8(res_colors, rev_mask);

                    *dest_pixel_pointer = _mm_cvtsi128_si32(res_colors);
                    bitmap_[dest_position + 3] = MAX_ALPHA;
                }
            }
        }

        friend void swap(BMPFile& first, BMPFile& second) noexcept {
            std::swap(first.data_, second.data_);
            std::swap(first.size_, second.size_);
            std::swap(first.header, second.header);
            std::swap(first.bitmap_, second.bitmap_);
        }
    
};

int main() {
    auto cat_file = BMPFile("pictures/cat.bmp");
    auto book_file = BMPFile("pictures/book.bmp");
    cat_file.ComposeAlpha(book_file, 20, 400);
    cat_file.SaveToFile("pictures/composed.bmp");
    return 0;
}
