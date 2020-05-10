#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <memory>

const int BMP_HEADER_END = 0x8a;
const int BYTES_PER_PIXEL = 4;
const unsigned char MAX_ALPHA = 255;
const int MAX_ALPHA_POW = 8;

void _mm_mullo_pi16(unsigned int* first_block, unsigned int* second_block) {
    for(int i = 0; i < 4; ++i) {
        *(first_block + i) = *(first_block + i) * (*second_block + i);
    }
}

void _mm256_add_epi8(unsigned int* first_block, unsigned int* second_block) {
    for(int i = 0; i < 4; ++i) {
        *(first_block + i) = *(first_block + i) + *(second_block + i);
    }
} 

void _mm_sra_epi16(unsigned int* block, int count) {
    for(int i = 0; i < 4; ++i) {
        *(block + i) = *(block + i) >> count;
    }
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

                unsigned int dest_colors[4] = {};
                unsigned int src_colors[4] = {};
                unsigned int alpha_blocks[4] = {};
                unsigned int src_alpha_blocks[4] = {};

                for(int j = 0; j < 3; ++j){
                    dest_colors[j] = data_[i + j];
                    src_colors[j] = other.data_[i + j];
                    alpha_blocks[j] = MAX_ALPHA - src_alpha;
                    src_alpha_blocks[j] = src_alpha;
                }

                _mm_mullo_pi16(dest_colors, alpha_blocks);
                _mm_mullo_pi16(src_colors, src_alpha_blocks);
                _mm256_add_epi8(src_colors, dest_colors);
                _mm_sra_epi16(src_colors, MAX_ALPHA_POW);

                for(int j = 0; j < 3; ++j){
                    data_[i + j] = static_cast<char>(src_colors[j]);
                }
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
