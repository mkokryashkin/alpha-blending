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


class BMPFile {
    private:
        std::unique_ptr<unsigned char[]> data_;
        std::unique_ptr<unsigned char[]> header_;

        int size_ = 0;
        int offbits_ = 0;
        int psize_ = 0;
        int width_ = 0;
        int height_ = 0;

        void CopyData(const BMPFile& other) {
            size_ = other.size_;
            offbits_ = other.offbits_;
            psize_ = other.psize_;
            width_ = other.width_;
            height_ = other.height_;

            data_ = std::unique_ptr<unsigned char[]>(new unsigned char[psize_]());
            header_ = std::unique_ptr<unsigned char[]>(new unsigned char[offbits_]());

            memcpy(data_.get(), other.data_.get(), psize_);  
            memcpy(header_.get(), other.header_.get(), offbits_);
        }

        void ReadProperty(int& member, int offset, FILE* file) {
            fseek(file, offset, SEEK_SET);
            fread(&member, sizeof(int), 1, file);
        }

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
        BMPFile() noexcept : size_(0), offbits_(0), psize_(0), width_(0), height_(0){}
        ~BMPFile() = default;

        BMPFile(const BMPFile& other) {
             CopyData(other);
        }

        BMPFile& operator=(const BMPFile& other) {
            CopyData(other);
            return *this;
        }

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
            
            ReadProperty(size_, BMP_FILE_SIZE_OFFSET, bmp_file.get());
            ReadProperty(offbits_, BMP_FILE_OFFBITS_OFFSET, bmp_file.get());
            ReadProperty(width_, BMP_FILE_WIDTH_OFFSET, bmp_file.get());
            ReadProperty(height_, BMP_FILE_HEIGHT_OFFSET, bmp_file.get());

            psize_ = size_ - offbits_;

            header_ = std::unique_ptr<unsigned char[]>(new unsigned char[offbits_]());
            data_ = std::unique_ptr<unsigned char[]>(new unsigned char[psize_]());

            fseek(bmp_file.get(), 0, SEEK_SET);
            fread(header_.get(), sizeof(unsigned char), offbits_, bmp_file.get());

            fseek(bmp_file.get(), offbits_, SEEK_SET);
            fread(data_.get(), sizeof(unsigned char), psize_, bmp_file.get());
        }

        int Size() const noexcept {
            return size_;
        }

        int Height() const noexcept {
            return height_;
        }

        int Width() const noexcept {
            return width_;
        }

        const unsigned char* Data() const noexcept {
            return data_.get();
        }

        void SaveToFile(const char* filename) {
            auto bmp_file = std::unique_ptr<FILE, FileCloser>(fopen(filename, "w"), FileCloser());
            fwrite(header_.get(), sizeof(unsigned char), offbits_, bmp_file.get());
            fseek(bmp_file.get(), offbits_, SEEK_SET);
            fwrite(data_.get(), sizeof(unsigned char), psize_, bmp_file.get());
        }

        void ComposeAlpha(const BMPFile& other, int x, int y) {
            for(int i = 0; i < psize_; i += BYTES_PER_PIXEL){
                unsigned char src_alpha = other.data_.get()[i + 3];

                int* dest_pixel_pointer = reinterpret_cast<int*>(data_.get() + i);
                __m128i dest_colors = _mm_cvtsi32_si128(*dest_pixel_pointer);     //loaded pixel data to first 32 bits of vector
                int src_pixel = *(reinterpret_cast<int*>(other.data_.get() + i));
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
                data_.get()[i + 3] = MAX_ALPHA;
            }
        }

        friend void swap(BMPFile& first, BMPFile& second) noexcept {
            std::swap(first.data_, second.data_);
            std::swap(first.size_, second.size_);
            std::swap(first.offbits_, second.offbits_);
            std::swap(first.header_, second.header_);
            std::swap(first.psize_, second.psize_);
            std::swap(first.height_, second.height_);
            std::swap(first.width_, second.width_);
        }
    
};

int main() {
    auto cat_file = BMPFile("pictures/cat.bmp");
    auto book_file = BMPFile("pictures/book.bmp");
    cat_file.ComposeAlpha(book_file, 0, 0);
    cat_file.SaveToFile("pictures/composed.bmp");
    std::cout << "height: " << cat_file.Height() << std::endl;
    std::cout << "width: " << cat_file.Width() << std::endl; 
    return 0;
}
