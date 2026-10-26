#pragma once
#include "numeric.hpp"
#include <vector>

namespace behemoth {
class PagePool {
    std::vector<int> free_;
    std::vector<bool> used_;

  public:
    explicit PagePool(int pages);
    size_t free_pages() const {
        return free_.size();
    }
    int capacity() const {
        return int(used_.size());
    }
    bool grow(std::vector<int> &table, size_t tokens);
    void release(std::vector<int> &table);
    static size_t bytes_per_page(int layers = 88, int kv_heads = 4);
    static int capacity_for_budget(size_t total, size_t resident, size_t workspace,
                                   double fraction = .9);
};
} // namespace behemoth
