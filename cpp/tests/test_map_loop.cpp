/**
 * @file test_map_loop.cpp
 * @brief Tests of the loop that applies a map kernel to a range of elements.
 *
 * The loop is tested with a kernel of exact arithmetic, so that a wrong
 * result can only come from the loads, the stores or the ranges.
 */
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <tuple>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <xsimd/xsimd.hpp>

#include "parkin/cpu/column_view.hpp"
#include "parkin/cpu/map_loop.hpp"
#include "parkin/kernels/math.hpp"
#include "test_support.hpp"

namespace {

using parkin::cpu::ColumnView;
using parkin::cpu::kMapChunkSize;
using parkin::cpu::ScalarLanes;
using parkin::cpu::SimdLanes;
using parkin::tests::column_of;

/**
 * @brief A kernel whose result shows which element of each column it was given.
 *
 * a + 64 b + 4096 c is exact in single precision for whole numbers below 64.
 */
struct PositionalSum {
  static constexpr std::size_t kInputCount = 3;
  static constexpr std::size_t kOutputCount = 1;

  template <class V>
  [[nodiscard]] std::array<V, kOutputCount> operator()(
      const std::array<V, kInputCount>& values) const noexcept {
    using T = parkin::math::ScalarType<V>;
    return {values[0] + V(T(64)) * values[1] + V(T(4096)) * values[2]};
  }
};

/** @brief A kernel with two outputs: a + b and a - b. */
struct SumAndDifference {
  static constexpr std::size_t kInputCount = 2;
  static constexpr std::size_t kOutputCount = 2;

  template <class V>
  [[nodiscard]] std::array<V, kOutputCount> operator()(
      const std::array<V, kInputCount>& values) const noexcept {
    return {values[0] + values[1], values[0] - values[1]};
  }
};

/** @brief Three columns in which element i differs from its neighbours in each. */
template <class T>
struct Columns {
  explicit Columns(std::size_t count) : a(count), b(count), c(count) {
    for (std::size_t index = 0; index < count; ++index) {
      a[index] = static_cast<T>(index % 61);
      b[index] = static_cast<T>(index % 59);
      c[index] = static_cast<T>(index % 53);
    }
  }

  [[nodiscard]] std::array<ColumnView, 3> views() const noexcept {
    return {column_of(a), column_of(b), column_of(c)};
  }

  /** @brief What PositionalSum gives for each element. */
  [[nodiscard]] std::vector<T> sums() const {
    std::vector<T> result(a.size());
    for (std::size_t index = 0; index < a.size(); ++index) {
      result[index] = PositionalSum{}(std::array<T, 3>{a[index], b[index], c[index]})[0];
    }
    return result;
  }

  std::vector<T> a, b, c;
};

/** @brief Apply PositionalSum to the elements [begin, end) of three columns. */
template <class Lanes>
void run(std::span<const ColumnView, 3> columns, std::vector<typename Lanes::Scalar>& out,
         std::size_t begin, std::size_t end) noexcept {
  std::byte* const output = reinterpret_cast<std::byte*>(out.data());
  parkin::cpu::map_block<PositionalSum, Lanes>(columns.data(), &output, begin, end);
}

/// The loop for one element at a time and for one batch at a time, in each precision.
using LanesTypes =
    std::tuple<ScalarLanes<float>, ScalarLanes<double>, SimdLanes<float, xsimd::default_arch>,
               SimdLanes<double, xsimd::default_arch>>;

TEMPLATE_LIST_TEST_CASE("Every element is computed from its own inputs, whatever the length",
                        "[map_loop]", LanesTypes) {
  using T = typename TestType::Scalar;
  // One element, less than a vector, a chunk less one, a chunk, a chunk and
  // one, and several chunks with a partial one.
  const std::array<std::size_t, 6> counts{
      1, 3, kMapChunkSize - 1, kMapChunkSize, kMapChunkSize + 1, 1000};

  for (const std::size_t count : counts) {
    CAPTURE(count);
    const Columns<T> columns(count);
    std::vector<T> out(count, T(-1));

    run<TestType>(columns.views(), out, 0, count);

    CHECK(out == columns.sums());
  }
}

TEMPLATE_LIST_TEST_CASE("Only the range asked for is written", "[map_loop]", LanesTypes) {
  using T = typename TestType::Scalar;
  constexpr std::size_t kCount = 1000;
  constexpr std::size_t kBegin = kMapChunkSize;
  constexpr std::size_t kEnd = 700;
  const Columns<T> columns(kCount);
  std::vector<T> out(kCount, T(-1));

  run<TestType>(columns.views(), out, kBegin, kEnd);

  std::vector<T> expected(kCount, T(-1));
  const std::vector<T> sums = columns.sums();
  std::copy(sums.begin() + kBegin, sums.begin() + kEnd, expected.begin() + kBegin);
  CHECK(out == expected);
}

TEMPLATE_LIST_TEST_CASE("An empty range writes nothing", "[map_loop]", LanesTypes) {
  using T = typename TestType::Scalar;
  const Columns<T> columns(600);
  std::vector<T> out(600, T(-1));

  run<TestType>(columns.views(), out, 0, 0);
  run<TestType>(columns.views(), out, kMapChunkSize, kMapChunkSize);

  CHECK(out == std::vector<T>(600, T(-1)));
}

TEMPLATE_LIST_TEST_CASE("A column is read where it is, whatever its stride and alignment",
                        "[map_loop]", LanesTypes) {
  using T = typename TestType::Scalar;
  constexpr std::size_t kCount = 1000;
  const Columns<T> columns(kCount);
  std::vector<T> out(kCount, T(-1));

  SECTION("fields of an array of records") {
    struct Record {
      T a;
      std::int8_t tag;
      T b;
      T c;
    };
    std::vector<Record> records(kCount);
    for (std::size_t index = 0; index < kCount; ++index) {
      records[index] = {columns.a[index], 0, columns.b[index], columns.c[index]};
    }
    const auto field = [](const T& first) {
      return ColumnView{reinterpret_cast<const std::byte*>(&first),
                        static_cast<std::ptrdiff_t>(sizeof(Record))};
    };
    const std::array<ColumnView, 3> views{field(records[0].a), field(records[0].b),
                                          field(records[0].c)};

    run<TestType>(views, out, 0, kCount);
  }

  SECTION("elements that are not aligned for their type") {
    // Records of three values and one byte, which start one byte into the buffer.
    constexpr std::size_t kRecordSize = 3 * sizeof(T) + 1;
    std::vector<std::byte> buffer(1 + kCount * kRecordSize);
    const std::array<const std::vector<T>*, 3> sources{&columns.a, &columns.b, &columns.c};
    std::array<ColumnView, 3> views{};
    for (std::size_t column = 0; column < sources.size(); ++column) {
      std::byte* const first = buffer.data() + 1 + column * sizeof(T);
      for (std::size_t index = 0; index < kCount; ++index) {
        std::memcpy(first + index * kRecordSize, &(*sources[column])[index], sizeof(T));
      }
      views[column] = {first, static_cast<std::ptrdiff_t>(kRecordSize)};
    }

    run<TestType>(views, out, 0, kCount);
  }

  SECTION("arrays read backwards") {
    const auto backwards = [](const std::vector<T>& values, std::vector<T>& reversed) {
      reversed.assign(values.rbegin(), values.rend());
      return ColumnView{reinterpret_cast<const std::byte*>(&reversed.back()),
                        -static_cast<std::ptrdiff_t>(sizeof(T))};
    };
    std::vector<T> a, b, c;
    const std::array<ColumnView, 3> views{backwards(columns.a, a), backwards(columns.b, b),
                                          backwards(columns.c, c)};

    run<TestType>(views, out, 0, kCount);
  }

  CHECK(out == columns.sums());
}

TEMPLATE_LIST_TEST_CASE("A stride of zero repeats one value", "[map_loop]", LanesTypes) {
  using T = typename TestType::Scalar;
  constexpr std::size_t kCount = 600;
  Columns<T> columns(kCount);
  const T repeated = 7;
  const std::array<ColumnView, 3> views{
      column_of(columns.a), ColumnView{reinterpret_cast<const std::byte*>(&repeated), 0},
      column_of(columns.c)};
  std::vector<T> out(kCount, T(-1));

  run<TestType>(views, out, 0, kCount);

  columns.b.assign(kCount, repeated);
  CHECK(out == columns.sums());
}

TEMPLATE_LIST_TEST_CASE("A kernel with several outputs writes each to its own array", "[map_loop]",
                        LanesTypes) {
  using T = typename TestType::Scalar;
  constexpr std::size_t kCount = 600;
  const Columns<T> columns(kCount);
  const std::array<ColumnView, 2> views{column_of(columns.a), column_of(columns.b)};
  std::vector<T> sums(kCount, T(-1));
  std::vector<T> differences(kCount, T(-1));
  const std::array<std::byte*, 2> outputs{reinterpret_cast<std::byte*>(sums.data()),
                                          reinterpret_cast<std::byte*>(differences.data())};

  parkin::cpu::map_block<SumAndDifference, TestType>(views.data(), outputs.data(), 0, kCount);

  std::vector<T> expected_sums(kCount);
  std::vector<T> expected_differences(kCount);
  for (std::size_t index = 0; index < kCount; ++index) {
    expected_sums[index] = columns.a[index] + columns.b[index];
    expected_differences[index] = columns.a[index] - columns.b[index];
  }
  CHECK(sums == expected_sums);
  CHECK(differences == expected_differences);
}

}  // namespace
