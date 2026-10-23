// Shared-object backend adapter for the unmodified DeepJIT disk cache.
// No cross-process lock or distributed coordination is added here.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <deep_jit/cache/disk.hpp>
#include <deep_jit/utils/hash.hpp>

namespace py = pybind11;
namespace fs = std::filesystem;

class ArtifactCache {
    deep_jit::DiskCache cache_;

public:
    explicit ArtifactCache(const std::vector<std::string>& roots)
        : cache_([&] {
              std::vector<fs::path> paths;
              for (const auto& root : roots) paths.emplace_back(root);
              return paths;
          }()) {}

    py::tuple get_or_build(const std::string& tag, const std::string& signature,
                          const py::function& builder) {
        const auto digest = deep_jit::hash::get_hex_digest(signature);
        auto entry = cache_.entry(tag, digest);
        const bool hit = entry.hit;
        if (!hit) {
            builder(entry.path.string());
            const auto artifact = entry.path / "module.so";
            if (!fs::is_regular_file(artifact) || fs::file_size(artifact) == 0)
                throw std::runtime_error("builder did not produce module.so");
            py::gil_scoped_release release;
            entry.commit();
        }
        return py::make_tuple((entry.path / "module.so").string(), digest, hit);
    }
};

PYBIND11_MODULE(_deepjit_cache, m) {
    m.doc() = "DeepJIT DiskCache adapter for TVM-FFI shared-library artifacts";
    py::class_<ArtifactCache>(m, "ArtifactCache")
        .def(py::init<const std::vector<std::string>&>())
        .def("get_or_build", &ArtifactCache::get_or_build);
    m.def("digest", &deep_jit::hash::get_hex_digest);
}
