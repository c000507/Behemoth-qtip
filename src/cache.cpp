#include "behemoth/cache.hpp"
#include "behemoth/attention.hpp"
#include <algorithm>
#include <climits>
#include <unordered_set>

namespace behemoth {
PagePool::PagePool(int pages) {
    if (pages < 1)
        throw std::invalid_argument("KV pool needs at least one page");
    used_.resize(size_t(pages), false);
    free_.reserve(size_t(pages));
    for (int i = pages - 1; i >= 0; --i)
        free_.push_back(i);
}
bool PagePool::grow(std::vector<int> &table, size_t tokens) {
    size_t need = ceil_div(tokens, page_tokens);
    if (need <= table.size())
        return true;
    if (need - table.size() > free_.size())
        return false;
    table.reserve(need);
    while (table.size() < need) {
        int id = free_.back();
        free_.pop_back();
        used_[size_t(id)] = true;
        table.push_back(id);
    }
    return true;
}
void PagePool::release(std::vector<int> &table) {
    std::unordered_set<int> seen;
    for (int id : table)
        if (id < 0 || id >= capacity() || !used_[size_t(id)] || !seen.insert(id).second)
            throw std::logic_error("Invalid/double KV release");
    for (int id : table) {
        used_[size_t(id)] = false;
        free_.push_back(id);
    }
    table.clear();
}
size_t PagePool::bytes_per_page(int layers, int kv_heads) {
    if (layers < 1 || kv_heads < 1)
        throw std::invalid_argument("Invalid KV dimensions");
    return checked_mul(size_t(layers), checked_mul(size_t(kv_heads) * page_tokens,
                                                   2 * head_dim + 2 * sizeof(float)));
}
int PagePool::capacity_for_budget(size_t total, size_t resident, size_t workspace,
                                  double fraction) {
    if (!(fraction > 0 && fraction <= 1))
        throw std::invalid_argument("Memory fraction outside (0,1]");
    size_t budget = static_cast<size_t>(double(total) * fraction);
    if (resident > budget || workspace > budget - resident)
        throw std::runtime_error("Weights/workspace exceed memory budget");
    size_t n = (budget - resident - workspace) / bytes_per_page();
    if (!n)
        throw std::runtime_error("No memory available for KV pages");
    return int(std::min(n, size_t(INT_MAX)));
}
} // namespace behemoth
