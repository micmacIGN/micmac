#ifndef KPT_COMMON_H
#define KPT_COMMON_H

#if __cplusplus < 201703L
#error C++ 17 is needed
#endif

#include <string>
#include <vector>
#include <set>
#include <iostream>
#include <filesystem>
#include <stdexcept>

// #define KAPTURE_USE_EIGEN

#ifdef KAPTURE_USE_EIGEN
#include <Eigen/Geometry>
#endif




// FIXME CM: Faire les write
// FIXME CM: Plus de verif sur la presence/type des donnes dans fichiers .txt (ex: float, int, ...)
// FIXME CM: Verif si Vect = read(..) est aussi rapide que read(&Vect,...)
// FIXME CM: Keypoints, features, ... : faire container pseudo std::vector

namespace Kapture {

typedef std::filesystem::path Path;
typedef std::vector<Path> PathList;
typedef std::vector<std::string> StringList;
typedef uint32_t timestamp_t;


const std::string KAPTURE_FORMAT_1_0 = "1.0";
const std::string KAPTURE_FORMAT_1_1 = "1.1";

const std::string KAPTURE_FORMAT_CURRENT = KAPTURE_FORMAT_1_1;
const std::string KAPTURE_FORMAT_HEADER = "# kapture format: " + KAPTURE_FORMAT_CURRENT;
const std::string KAPTURE_FORMAT_PARSING_RE = "# kapture format\\s*:\\s*(\\d+\\.\\d+)\\s*";

const std::set<std::string> KAPTURE_FORMAT_SUPPORTED = {KAPTURE_FORMAT_1_0, KAPTURE_FORMAT_1_1};


std::vector<char> readBinaryFile(std::istream& is);
std::vector<char> readBinaryFile(const Path& p);


inline Path sensorsDir() { return "sensors"; }
inline Path sensorsPath() { return  sensorsDir() / "sensors.txt"; }
inline Path trajectoriesPath() { return  sensorsDir() / "trajectories.txt"; }
inline Path rigsPath() { return  sensorsDir() / "rigs.txt"; }
inline Path recordsCameraPath() { return  sensorsDir() / "records_camera.txt"; }
inline Path recordsDataDir() { return  sensorsDir() / "records_data"; }

inline Path reconstructionDir() {return "reconstruction"; }

inline Path keypointsDir(const std::string& keypoints_type) {return reconstructionDir() / "keypoints" / keypoints_type; }
inline Path keypointsTypePath(const std::string& keypoints_type) {return keypointsDir(keypoints_type) / "keypoints.txt"; }
inline Path keypointsPath(const Path& imagePath, const std::string& keypoints_type) {
    Path p = keypointsDir(keypoints_type) / imagePath;
    p += ".kpt";
    return p;
}

inline Path matchesDir(const std::string& match_type) {return reconstructionDir() / "matches" / match_type; }
inline Path matchesPath(Path img1, Path img2,const std::string& matchType)
{
    if (img1 > img2)
        std::swap(img1,img2);
    img1 += ".overlapping";
    img2 += ".matches";
    return matchesDir(matchType) / img1 / img2;
}





enum class DType{Unknown,UINT8,UINT16,UINT32,UINT64,FLOAT32,FLOAT64};

const char *dtypeToStr(DType t);
DType dtypeFromStr(const std::string &s);

template <typename T>
constexpr DType DTypeEnum() { return DType::Unknown;}

template<>
constexpr DType DTypeEnum<float>() { return DType::FLOAT32;}

template<>
constexpr DType DTypeEnum<double>() { return DType::FLOAT64;}


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
    const std::string mErrorMsg;
    const std::string mFile;
    size_t mLine;
    const std::string mFunc;
};


} // namespace Kapture

#endif // KPT_COMMON_H
