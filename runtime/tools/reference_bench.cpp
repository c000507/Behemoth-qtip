#include "../tests/fixtures.hpp"
#include <chrono>
#include <iostream>
int main(int argc, char **argv) {
    try {
        int length = argc > 1 ? std::stoi(argv[1]) : 129;
        if (length < 1 || length > 131072)
            throw std::invalid_argument("Context outside [1,131072]");
        auto f = behemoth::test::fixture({length}, {{length - 1}});
        auto start = std::chrono::steady_clock::now();
        auto r = behemoth::reference_attention(f.layout, f.data, true, length > 1024 ? 1024 : 0);
        double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        auto baseline = behemoth::reference_attention(f.layout, f.data, false);
        auto error = behemoth::compare_attention(r.output, baseline.output);
        std::cout << "CPU REFERENCE ONLY; not GPU performance or GPU correctness\ncontext="
                  << length << " seconds=" << seconds << " quantization_rms=" << error.rms_error
                  << " max_abs=" << error.max_abs << '\n';
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
