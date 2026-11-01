#include "icp.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace icp {

Cloud read_cloud(const std::string& file) {
    Cloud points;
    std::string ext = std::filesystem::path(file).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    if (ext == ".bin") {
        std::ifstream in(file, std::ios::binary|std::ios::ate);
        if (!in) {
            throw std::runtime_error("Cannot read "+file);
        }
        const auto bytes = in.tellg();
        if (bytes <= 0 || bytes%std::streamoff(16) != 0) {
            throw std::runtime_error("KITTI file must contain 16-byte xyzi float32 records");
        }
        points.reserve(static_cast<std::size_t>(bytes) / 16);
        in.seekg(0);
        float a[4];
        while (in.read(reinterpret_cast<char*>(a), sizeof a)) {
            points.push_back({a[0], a[1], a[2]});
        }
        if (!in.eof()) {
            throw std::runtime_error("Read failure "+file);
        }
    }
    else if (ext == ".xyz" || ext == ".txt") {
        std::ifstream in(file);
        if (!in) {
            throw std::runtime_error("Cannot read "+file);
        }
        std::string line;
        int lineno = 0;
        while (std::getline(in, line)) {
            ++lineno;
            const auto pos = line.find_first_not_of(" \t\r");
            if (pos == std::string::npos || line[pos] == '#') {
                continue;
            }
            std::istringstream row(line);
            Point a{};
            if (!(row >> a.x >> a.y >> a.z)) {
                throw std::runtime_error("Invalid xyz line "+std::to_string(lineno));
            }
            points.push_back(a);
            // Extra columns (e.g. intensity) ignored.
        }
    }
    else {
        throw std::runtime_error("Use .xyz/.txt or KITTI .bin: "+file);
    }
    if (points.empty()) {
        throw std::runtime_error("No points in "+file);
    }
    for (auto a:points) {
        if (!std::isfinite(a.x) || !std::isfinite(a.y) || !std::isfinite(a.z)) {
            throw std::runtime_error("Nonfinite point in "+file);
        }
    }
    return points;
}

Pose read_pose(const std::string& file) {
    std::ifstream in(file);
    if (!in) {
        throw std::runtime_error("Cannot read pose "+file);
    }
    std::vector<double> v;
    double x;
    while (in >> x) {
        v.push_back(x);
    }
    if (!in.eof() || (v.size() != 12 && v.size() != 16)) {
        throw std::runtime_error("Pose must be 12 or 16 whitespace-separated numbers");
    }
    for (double a:v) {
        if (!std::isfinite(a)) {
            throw std::runtime_error("Nonfinite pose");
        }
    }
    if (v.size() == 16 && (std::abs(v[12]) + std::abs(v[13]) + std::abs(v[14]) + std::abs(v[15] - 1)>1e-8)) {
        throw std::runtime_error("Invalid homogeneous bottom row");
    }
    Pose p;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            p.R[3 * i + j] = v[4 * i + j];
        }
        p.t[i] = v[4 * i + 3];
    }
    double orth_error = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double dot = 0;
            for (int k = 0; k < 3; ++k) {
                dot += p.R[3 * k + i] * p.R[3 * k + j];
            }
            orth_error += std::abs(dot - (i == j));
        }
    }
    const auto& r = p.R;
    double det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
    if (orth_error>1e-3 || std::abs(det - 1)>1e-3) {
        throw std::runtime_error("Initial rotation must be in SO(3)");
    }
    return p;
}

void write_result(const std::string& file, const Result& r, const Options& o) {
    auto parent = std::filesystem::path(file).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream f(file);
    if (!f) {
        throw std::runtime_error("Cannot write "+file);
    }
    f << std::setprecision(17);
    f << "{\n  \"schema_version\": 1,\n  \"backend\": \"" << r.backend << "\",\n";
    f << "  \"source_points\": " << r.source_points << ", \"target_points\": " << r.target_points << ",\n";
    f << "  \"iterations\": " << r.trace.size() << ", \"requested_iterations\": " << o.iterations << ",\n";
    f << "  \"fixed_iterations\": " << (o.fixed_iterations?"true":"false") << ", \"converged\": " << (r.converged?"true":"false") << ",\n";
    f << "  \"max_distance\": " << o.max_distance << ",\n";
    f << "  \"T_target_source\": [";
    for (int a = 0; a < 4; ++a) {
        if (a) {
            f << ", ";
        }
        f << "[";
        for (int b = 0; b < 4; ++b) {
            if (b) {
                f << ", ";
            }
            f << (a == 3?(b == 3?1.0:0.0):(b == 3?r.pose.t[a]:r.pose.R[3 * a + b]));
        }
        f << "]";
    }
    f << "],\n";
    f << "  \"final_inliers\": " << r.final_inliers << ", \"final_rmse\": " << r.final_rmse << ",\n";
    f << "  \"setup_ms\": " << r.setup_ms << ", \"loop_wall_ms\": " << r.loop_wall_ms << ",\n";
    f << "  \"final_eval_ms\": " << r.final_eval_ms << ", \"registration_wall_ms\": " << r.registration_wall_ms << ",\n";
    f << "  \"trace\": [\n";
    for (std::size_t k = 0; k<r.trace.size(); ++k) {
        const auto& t = r.trace[k];
        f << "    {\"iteration\":" << t.number << ",\"inliers\":" << t.inliers << ",\"rmse_before_update\":" << t.rmse_before_update;
        f << ",\"delta_translation\":" << t.delta_translation << ",\"delta_rotation_rad\":" << t.delta_rotation;
        f << ",\"transform_ms\":" << t.times.transform_ms << ",\"match_ms\":" << t.times.match_ms;
        f << ",\"reduce_ms\":" << t.times.reduce_ms << ",\"download_ms\":" << t.times.download_ms << ",\"solve_ms\":" << t.solve_ms << "}";
        f << (k + 1<r.trace.size()?",\n":"\n");
    }
    f << "  ]\n}\n";
    if (!f) {
        throw std::runtime_error("Write failure "+file);
    }
}
}
