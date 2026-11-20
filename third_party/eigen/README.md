# Bundled Eigen headers

Eigen 3.4.0, copied unchanged from the official release archive:

https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz

Archive SHA-256: `8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72`.

Only the complete `Eigen/` header directory and upstream `COPYING.*` files
are bundled. Upstream tests, documentation, demos, unsupported modules and
build scripts are omitted. The adjacent CMakeLists.txt is project-owned wiring.

Eigen is primarily MPL-2.0 licensed. See COPYING.README, the included license
texts and individual header notices for applicable terms and exceptions.
Upstream copyright and license notices are retained without modification.

CPU and GPU CMake builds use these headers; the direct GPU build also uses
them by default. No separate Eigen installation or network download is needed.
These are ordinary repository files, included in GitHub Download ZIP archives.
