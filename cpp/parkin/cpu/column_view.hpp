/**
 * @file column_view.hpp
 * @brief A borrowed column of floating-point values.
 */
#ifndef PARKIN_CPU_COLUMN_VIEW_HPP
#define PARKIN_CPU_COLUMN_VIEW_HPP

#include <cstddef>

namespace parkin::cpu {

/**
 * @brief One component of a set of particles, read where the caller keeps it.
 *
 * The view owns nothing. Element i is at `data + i * stride`, so a column can
 * be a field of an array of records, every n-th element of an array, an array
 * read backwards (a negative stride) or one value repeated (a zero stride).
 * An element need not be aligned for its type.
 */
struct ColumnView {
  const std::byte* data;  ///< Address of the first element.
  std::ptrdiff_t stride;  ///< Distance in bytes from one element to the next.
};

}  // namespace parkin::cpu

#endif  // PARKIN_CPU_COLUMN_VIEW_HPP
