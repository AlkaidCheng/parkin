/**
 * @file module.cpp
 * @brief The Python module parkin._core: bindings of the C++ core.
 */
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "parkin/cpu/build_info.hpp"
#include "parkin/cpu/column_view.hpp"
#include "parkin/cpu/cpu_features.hpp"
#include "parkin/cpu/instruction_set.hpp"
#include "parkin/cpu/map_executor.hpp"
#include "parkin/cpu/map_kernels.hpp"

namespace nb = nanobind;

namespace {

/**
 * @brief The memory of a Python object that exports the buffer protocol.
 *
 * The buffer protocol describes strides in bytes and places no alignment
 * requirement on the data, so it can lend every column that
 * parkin::cpu::ColumnView can describe, such as a field of a packed record
 * array. The export is released when this object is destroyed, which must
 * happen while the interpreter lock is held.
 */
class BufferView {
 public:
  /**
   * @param exporter The object to borrow from.
   * @param flags PyBUF_* flags of the request.
   * @throws nb::python_error With the exporter's error when it cannot
   *         provide the buffer, for example a read-only one asked to be
   *         writable.
   */
  BufferView(nb::handle exporter, int flags) {
    if (PyObject_GetBuffer(exporter.ptr(), &view_, flags) != 0) {
      throw nb::python_error();
    }
  }
  BufferView(const BufferView&) = delete;
  BufferView& operator=(const BufferView&) = delete;
  ~BufferView() { PyBuffer_Release(&view_); }

  [[nodiscard]] const Py_buffer& view() const noexcept { return view_; }

 private:
  Py_buffer view_{};
};

/**
 * @brief The element type of a buffer, when it is float or double in native byte order.
 *
 * The format is a `struct` format string. NumPy writes "f" and "d", with the
 * prefix "=" for a field that is not aligned within its record, and with "<"
 * or ">" when the byte order is spelled out.
 */
[[nodiscard]] std::optional<parkin::cpu::FloatType> float_type_of(const Py_buffer& view) {
  if (view.format == nullptr) {
    return std::nullopt;
  }
  std::string_view format(view.format);
  const char native_order = std::endian::native == std::endian::little ? '<' : '>';
  if (!format.empty() &&
      (format.front() == '@' || format.front() == '=' || format.front() == native_order)) {
    format.remove_prefix(1);
  }
  if (format == "f" && view.itemsize == static_cast<Py_ssize_t>(sizeof(float))) {
    return parkin::cpu::FloatType::kFloat32;
  }
  if (format == "d" && view.itemsize == static_cast<Py_ssize_t>(sizeof(double))) {
    return parkin::cpu::FloatType::kFloat64;
  }
  return std::nullopt;
}

/**
 * @brief Apply a map kernel to columns lent by Python objects.
 *
 * Nothing is converted: an argument that the core cannot read as it is gets
 * refused. The interpreter lock is released while the kernel runs.
 */
void run_map(const std::string& kernel_name, nb::sequence inputs, nb::handle out,
             std::size_t thread_count, std::size_t block_size,
             const std::optional<std::string>& instruction_set_name) {
  const std::optional<parkin::cpu::MapKernel> kernel = parkin::cpu::find_map_kernel(kernel_name);
  if (!kernel) {
    throw std::invalid_argument("unknown map kernel '" + kernel_name + "'");
  }
  parkin::cpu::MapOptions options;
  options.thread_count = thread_count;
  options.block_size = block_size;
  options.instruction_set = parkin::cpu::widest_instruction_set();
  if (instruction_set_name) {
    const std::optional<parkin::cpu::InstructionSet> instruction_set =
        parkin::cpu::find_instruction_set(*instruction_set_name);
    if (!instruction_set) {
      throw std::invalid_argument("unsupported instruction set '" + *instruction_set_name + "'");
    }
    options.instruction_set = *instruction_set;
  }

  const BufferView out_buffer(out, PyBUF_RECORDS);
  const Py_buffer& out_view = out_buffer.view();
  const std::optional<parkin::cpu::FloatType> float_type = float_type_of(out_view);
  if (!float_type) {
    throw nb::type_error("out must hold float32 or float64 values in native byte order");
  }
  if (out_view.ndim != 1 || out_view.strides[0] != out_view.itemsize ||
      reinterpret_cast<std::uintptr_t>(out_view.buf) %
              static_cast<std::size_t>(out_view.itemsize) !=
          0) {
    throw std::invalid_argument("out must be one-dimensional, contiguous and aligned");
  }
  const Py_ssize_t element_count = out_view.shape[0];

  const std::size_t input_count = nb::len(inputs);
  if (input_count != parkin::cpu::map_kernel_input_count(*kernel)) {
    throw std::invalid_argument("the kernel '" + kernel_name + "' reads " +
                                std::to_string(parkin::cpu::map_kernel_input_count(*kernel)) +
                                " columns, got " + std::to_string(input_count));
  }
  // A BufferView can be neither copied nor moved, so the vector holds owners.
  std::vector<std::unique_ptr<BufferView>> input_buffers;
  std::vector<parkin::cpu::ColumnView> columns;
  input_buffers.reserve(input_count);
  columns.reserve(input_count);
  for (nb::handle input : inputs) {
    input_buffers.push_back(std::make_unique<BufferView>(input, PyBUF_RECORDS_RO));
    const Py_buffer& view = input_buffers.back()->view();
    if (float_type_of(view) != float_type) {
      throw nb::type_error("every input must hold values of the type of out, in native byte order");
    }
    if (view.ndim != 1 || view.shape[0] != element_count) {
      throw std::invalid_argument("every input must be one-dimensional and as long as out");
    }
    columns.push_back({static_cast<const std::byte*>(view.buf), view.strides[0]});
  }

  {
    const nb::gil_scoped_release release;
    parkin::cpu::run_map(*kernel, *float_type, columns, static_cast<std::byte*>(out_view.buf),
                         static_cast<std::size_t>(element_count), options);
  }
}

[[nodiscard]] std::vector<std::string> supported_instruction_sets() {
  std::vector<std::string> names;
  for (const parkin::cpu::InstructionSet instruction_set :
       parkin::cpu::supported_instruction_sets()) {
    names.emplace_back(parkin::cpu::instruction_set_name(instruction_set));
  }
  return names;
}

}  // namespace

NB_MODULE(_core, core) {
  core.doc() = "Compiled core of parkin.";

  core.def("compiler", &parkin::cpu::compiler,
           "Return the id and version of the compiler that built the core.");
  core.def("xsimd_version", &parkin::cpu::xsimd_version,
           "Return the version of the xsimd headers the core was compiled with.");
  core.def("detected_cpu_features", &parkin::cpu::detected_cpu_features,
           "Return the SIMD instruction sets the running CPU supports.");
  core.def("supported_instruction_sets", &supported_instruction_sets,
           "Return the instruction sets kernels can run with, narrowest first.");
  core.def("run_map", &run_map, nb::arg("kernel"), nb::arg("inputs"), nb::arg("out"),
           nb::arg("thread_count"), nb::arg("block_size"), nb::arg("instruction_set").none(),
           "Apply a map kernel to one-dimensional float columns and write the result to out.");
}
