#ifndef KPT_SENSORS_H
#define KPT_SENSORS_H

#include <string>
#include <vector>
#include <memory>

#include "kpt_common.h"

namespace Kapture {

// TODO: Camera prend un std::vector<double> ?

class Sensor;
class Camera;
class ImageRecord;
class Trajectory;
class Rig;

typedef std::vector<Sensor> SensorList;
typedef std::vector<Camera> CameraList;
typedef std::vector<ImageRecord> ImageRecordList;
typedef std::vector<Trajectory> TrajectoryList;
typedef std::vector<Rig> RigList;


class Sensor
{
public:
    enum Type {UNKNOWN_TYPE, CAMERA, DEPTH, GNSS, LIDAR, WIFI, MAGNETIC, PRESSURE};
    Sensor() : mType(UNKNOWN_TYPE) {}
    explicit Sensor(Type type) : mType(type) {}
    Sensor(const std::string& id, const std::string& name, const std::string& typeStr,
           StringList::const_iterator first, StringList::const_iterator last);

    std::string device() const {return mDevice ;}
    std::string name() const { return mName;}
    Type type() const { return mType;}
    std::string typeStr() const  { return mType==UNKNOWN_TYPE ?  mTypeStr : typeToStr(mType);}
    StringList params() const { return mParams;}

private:
    static Type typeFromStr(const std::string &  s);
    static std::string typeToStr(Type typeStr);

    std::string mDevice;
    std::string mName;
    Type mType;
    std::string mTypeStr;
    std::vector<std::string> mParams;
};



class Camera : public Sensor
{
public:
    enum Model {
        UNKNOWN_CAMERA,
        SIMPLE_PINHOLE, PINHOLE,
        SIMPLE_RADIAL, RADIAL,
        OPENCV, OPENCV_FISHEYE, FULL_OPENCV,
        FOV,
        SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, THIN_PRISM_FISHEYE,
    };

    Camera() : Sensor(CAMERA),mModel(UNKNOWN_CAMERA) {}
    explicit Camera(const Sensor& sensor);
    Camera(const std::string& id, const std::string& name,
           const StringList::const_iterator& begin, const StringList::const_iterator& end);

    std::string modelStr() const { return params()[0];}
    Model model() const { return mModel;}
    std::vector<double> modelParams() const { return mModelParams;}


private:
    static std::string modelToStr(Model model);
    static Model modelFromStr(const std::string &  s);
    Model mModel;
    std::vector<double> mModelParams;
};


class ImageRecord
{
public:
    ImageRecord();
    ImageRecord(timestamp_t timestamp, const std::string& device, const Path& image)
        : mTimestamp(timestamp),mDevice(device),mImage(image),mCamera(nullptr),mTrajectory(nullptr) {}

    timestamp_t timestamp() const { return mTimestamp; }
    std::string device() const { return mDevice; }
    Path image() const { return mImage; }
    const Camera* camera() const { return mCamera;}
    const Trajectory* trajectory() const { return mTrajectory;}

    void setCamera(const Camera* camera) { mCamera = camera;}
    void setTrajectory(const Trajectory * trajectory) { mTrajectory = trajectory;}
private:
    timestamp_t mTimestamp;
    std::string mDevice;
    Path mImage;
    const Camera *mCamera;
    const Trajectory *mTrajectory;
};


class Orientation
{
public:
    Orientation() : mHasRot(false),mHasVec(false) {}
    Orientation(double w, double x, double y, double z) : mQ(w,x,y,z),mHasRot(true),mHasVec(false) {}
    Orientation(const QRot& q) : mQ(q),mHasRot(true),mHasVec(false) {}
    Orientation(double x, double y, double z) : mV(x,y,z),mHasRot(true),mHasVec(false) {}
    Orientation(const QRot& q, const Vec3D& v) : mQ(q),mV(v),mHasRot(true),mHasVec(true) {}

    bool hasRot() const { return mHasRot; }
    bool hasVec() const { return mHasVec; }

    void setRot(const QRot& q)  { mQ=q; mHasRot=true; }
    void setVec(const Vec3D& v) { mV=v; mHasVec=true; }
    void setRot(double w, double x, double y, double z) { setRot(QRot(w,x,y,z)); }
    void setVec(double x, double y, double z) { setVec(Vec3D(x,y,z)); }

    const QRot& q() const { return mQ; }
    const Vec3D& v() const { return mV; }

private:
    QRot mQ;
    Vec3D mV;
    bool mHasRot;
    bool mHasVec;
};

class Trajectory : public Orientation
{
public:
    Trajectory() {}

    Trajectory(timestamp_t timestamp, const std::string& device)
        : mTimestamp(timestamp),mDevice(device) {}

    timestamp_t timestamp() const { return mTimestamp; }
    std::string device() const { return mDevice; }

private:
    timestamp_t mTimestamp;
    std::string mDevice;
};


class Rig : public Orientation
{
public:
    Rig() {}
    Rig(const std::string& name, const std::string& device)
        : mName(name),mDevice(device) {}

    std::string name() const { return mName; }
    std::string device() const { return mDevice; }

private:
    std::string mName;
    std::string mDevice;
};


} // namespace Kapture

#endif // KPT_SENSORS_H
