#ifndef KPT_RECONSTRUCTION_H
#define KPT_RECONSTRUCTION_H

#include "kpt_common.h"

namespace Kapture {

struct Pair {
    typedef  std::vector<Pair> List;

    Pair (float x1, float y1, float x2, float y2) : x1(x1),y1(y1),x2(x2),y2(y2) {}

    float x1,y1;
    float x2,y2;
};


template <typename T>
class Points3D {
public:
    typedef std::vector<Points3D<T>> List;

    Points3D() : mHasRgb(0) {}
    Points3D(T x, T y, T z) : mX(x),mY(y),mZ(z),mHasRgb(0) {}
    Points3D(T x, T y, T z, unsigned r, unsigned g, unsigned b) : mX(x),mY(y),mZ(z),mR(r),mG(g),mB(b),mHasRgb(1) {}

    void setXYZ(T x, T y, T z) { mX=x; mY=y; mZ=z;}
    void setRGB(unsigned r, unsigned g, unsigned b) { mR=r; mG=g; mB=b; mHasRgb = 1;}

    T x() const { return mX; }
    T y() const { return mY; }
    T z() const { return mZ; }

    unsigned r() const { return mR; }
    unsigned g() const { return mG; }
    unsigned b() const { return mB; }

    bool hasRGB() const { return mHasRgb;}

    static List read(const Path &path);

private:
    T mX,mY,mZ;
    uint8_t mR,mG,mB;
    uint8_t mHasRgb;
};


class Observations {
public:
    typedef std::vector<Observations> List;

    Observations() {}
    Observations(unsigned point3D_id,
                 const std::string& keypoints_type,
                 const std::string& image_path,
                 unsigned feature_id)
        : mPoint3d_id(point3D_id),mKeypoints_type(keypoints_type),
          mImage_path(image_path),mFeature_id(feature_id)
    {}
    unsigned point3d_id() const { return mPoint3d_id; }
    std::string keypoints_type() const { return mKeypoints_type; }
    std::string image_path() const { return mImage_path; }
    unsigned feature_id() const { return mFeature_id; }

    static List read(const Path& path);

private:
    unsigned mPoint3d_id;
    std::string mKeypoints_type;
    std::string mImage_path;
    unsigned mFeature_id;
};


class KeypointsType {
public:
    KeypointsType() : mDType(DType::Unknown),mDSize(-1) {}
    KeypointsType(const std::string& name, DType type, int size) : mName(name),mDType(type),mDSize(size) {}

    std::string name() const { return mName; }
    DType dtype() const { return mDType; }
    int dsize() const { return mDSize; }

    static KeypointsType read(const Path& path);
private:
    std::string mName;
    DType mDType;
    int mDSize;
};


template <typename T>
class Keypoints {
public:
    Keypoints() : mDSize(0) {}
    Keypoints(int dSize, int nElem) : mData(dSize * nElem),mDSize(dSize) {}

    T operator() (int i, int j) const  { return mData[i*mDSize+j]; };
    T x(int i) const { return operator()(i,0); }
    T y(int i) const { return operator()(i,1); }

    int dsize() const { return mDSize; }
    static constexpr DType dtype() { return DTypeEnum<T>() ;}

    static Keypoints<T> read(const Path& path, int dsize);
private:
    std::vector<T> mData;
    int mDSize;
};


class DescriptorsType {
public:
    DescriptorsType() : mDType(DType::Unknown),mDSize(-1) {}
    DescriptorsType(const std::string& name,
                    DType type,
                    int size,
                    const std::string& keypoints_type,
                    const std::string& metric_type
                    )
        : mName(name),mDType(type),mDSize(size),
          mKeypoints_type(keypoints_type),mMetric_type(metric_type)
    {}

    std::string name() const { return mName; }
    DType dtype() const { return mDType; }
    int dsize() const { return mDSize; }
    std::string keypoints_type() const { return mKeypoints_type;}
    std::string metric_type() const { return mMetric_type;}

    static DescriptorsType read(const Path& path);

private:
    std::string mName;
    DType mDType;
    int mDSize;
    std::string mKeypoints_type;
    std::string mMetric_type;
};

template <typename T>
class Descriptors {
public:
    Descriptors() : mDSize(0) {}

    T operator()(int i, int j) const  { return mData[i*mDSize+j]; };

    int dsize() const { return mDSize; }
    static constexpr DType dtype() { return DTypeEnum<T>() ;}

    void read(const Path& path, int dsize);
private:
    std::vector<T> mData;
    int mDSize;
};


class GlobalFeaturesType {
public:
    GlobalFeaturesType() : mDType(DType::Unknown),mDSize(-1) {}
    GlobalFeaturesType(const std::string& name,
                    DType type,
                    int size,
                    const std::string& metric_type
                    )
        : mName(name),mDType(type),mDSize(size),
          mMetric_type(metric_type)
    {}

    std::string name() const { return mName; }
    DType dtype() const { return mDType; }
    int dsize() const { return mDSize; }
    std::string metric_type() const { return mMetric_type;}

    static GlobalFeaturesType read(const Path& path);

private:
    std::string mName;
    DType mDType;
    int mDSize;
    std::string mMetric_type;
};

template <typename T>
class GlobalFeatures  : public Descriptors<T> {
};


class Matches {
public:
    typedef std::vector<Matches> List;

    Matches() {}

    double idx1() const { return mIdx1; }
    double idx2() const { return mIdx2; }
    double score() const { return mScore; }

    static List read(const Path& path);

private:
    double mIdx1,mIdx2,mScore;
};

} // namespace Kapture

#endif // KPT_RECONSTRUCTION_H
