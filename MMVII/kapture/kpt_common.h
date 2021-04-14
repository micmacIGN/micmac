#ifndef KPT_COMMON_H
#define KPT_COMMON_H

#if __cplusplus < 201703L
#error C++ 17 is needed
#endif

#include <string>
#include <vector>
#include <iostream>
#include <experimental/filesystem>
#include <stdexcept>

#define KAPTURE_USE_EIGEN

#ifdef KAPTURE_USE_EIGEN
#include <Eigen/Geometry>
#endif


namespace std {
using namespace experimental;
}


namespace Kapture {

extern bool debugOn;

const std::string KAPTURE_FORMAT_1_0 = "1.0";

const std::string KAPTURE_FORMAT_CURRENT = KAPTURE_FORMAT_1_0;
const std::string KAPTURE_FORMAT_HEADER = "# kapture format: " + KAPTURE_FORMAT_CURRENT;
const std::string KAPTURE_FORMAT_PARSING_RE = "# kapture format\\s*:\\s*(\\d+\\.\\d+)\\s*";

typedef std::filesystem::path Path;
typedef std::vector<Path> PathList;
typedef std::vector<std::string> StringList;
typedef uint32_t timestamp_t;

enum class DType{Unknown,UINT8,UINT16,UINT32,UINT64,FLOAT32,FLOAT64};

const char *dtypeToStr(DType t);
DType dtypeFromStr(const std::string &s);

std::vector<char> readBinaryFile(std::istream& is);
std::vector<char> readBinaryFile(const Path& p);

#ifdef KAPTURE_USE_EIGEN
typedef Eigen::Quaternion<double> QRot;
#else
struct QRot {
public:
    QRot() : mW(0),mX(0),mY(0),mZ(0) {};
    QRot(double w, double x, double y, double z) : mW(w),mX(x),mY(y),mZ(z) {}
    double w() const { return mW; }
    double x() const { return mX; }
    double y() const { return mY; }
    double z() const { return mZ; }
    double& w() { return mW; }
    double& x() { return mX; }
    double& y() { return mY; }
    double& z() { return mZ; }

private:
    double mW,mX,mY,mZ;
};
#endif

struct Vec3D {
public:
    Vec3D() : mX(0),mY(0),mZ(0) {};
    Vec3D(double x, double y, double z) : mX(x),mY(y),mZ(z) {}
    double x() const { return mX; }
    double y() const { return mY; }
    double z() const { return mZ; }
    double& x() { return mX; }
    double& y() { return mY; }
    double& z() { return mZ; }

private:
    double mX,mY,mZ;
};

class Error : public std::runtime_error
{
public:
    Error(const std::string& errorMsg,
              const std::string& file, size_t line, const std::string& func);

    std::string errorMsg() const { return mErrorMsg; }
    std::string file() const { return mFile; }
    std::string func() const { return mFunc; }
    size_t line() const { return mLine; }

private:
    const std::string& mErrorMsg;
    const std::string& mFile;
    size_t mLine;
    const std::string& mFunc;
};


} // namespace Kapture

#endif // KPT_COMMON_H
