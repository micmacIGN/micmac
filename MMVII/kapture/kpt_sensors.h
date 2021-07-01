#ifndef KPT_SENSORS_H
#define KPT_SENSORS_H

#include <string>
#include <vector>
#include <memory>

#include "kpt_common.h"

namespace Kapture {


class Sensor;
class Camera;
class RecordsCamera;
class Trajectory;
class Rig;

class Sensor
{
public:
    typedef std::vector<Sensor> List;

    enum Type {UNKNOWN_TYPE, CAMERA, DEPTH, GNSS, LIDAR, WIFI, MAGNETIC, PRESSURE};
    Sensor() : mSensor_type(UNKNOWN_TYPE) {}
    explicit Sensor(Type type) : mSensor_type(type) {}
    Sensor(const std::string& id, const std::string& name, const std::string& typeStr,
           StringList::const_iterator first, StringList::const_iterator last);

    std::string sensor_device_id() const {return mSensor_device_id ;}
    std::string name() const { return mName;}
    Type sensor_type() const { return mSensor_type;}
    std::string typeStr() const  { return mSensor_type==UNKNOWN_TYPE ?  mTypeStr : typeToStr(mSensor_type);}
    StringList sensor_params() const { return mSensor_params;}


    static Sensor::List read(const Path &path);

private:
    static Type typeFromStr(const std::string &  s);
    static std::string typeToStr(Type typeStr);

    std::string mSensor_device_id;
    std::string mName;
    Type mSensor_type;
    std::string mTypeStr;
    std::vector<std::string> mSensor_params;
};



class Camera : public Sensor
{
public:
    typedef std::vector<Camera> List;

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

    std::string modelStr() const { return sensor_params()[0];}
    Model model() const { return mModel;}
    std::vector<double> model_params() const { return mModel_params;}

    static Camera::List read(const Path &path);
private:
    static std::string modelToStr(Model model);
    static Model modelFromStr(const std::string &  s);
    Model mModel;
    std::vector<double> mModel_params;
};


class RecordsCamera
{
public:
    typedef std::vector<RecordsCamera> List;

    RecordsCamera();
    RecordsCamera(timestamp_t timestamp, const std::string& device_id, const Path& image_path)
        : mTimestamp(timestamp),mDevice_id(device_id),mImage_path(image_path),mCamera(nullptr),mTrajectory(nullptr) {}

    timestamp_t timestamp() const { return mTimestamp; }
    std::string device_id() const { return mDevice_id; }
    Path image_path() const { return mImage_path; }
    const Camera* camera() const { return mCamera;}
    const Trajectory* trajectory() const { return mTrajectory;}

    void setCamera(const Camera* camera) { mCamera = camera;}
    void setTrajectory(const Trajectory * trajectory) { mTrajectory = trajectory;}

    static RecordsCamera::List read(const Path &path);

private:
    timestamp_t mTimestamp;
    std::string mDevice_id;
    Path mImage_path;
    const Camera *mCamera;
    const Trajectory *mTrajectory;
};


class Orientation
{
public:
    Orientation() : mHasRot(false),mHasVec(false) {}
    Orientation(double w, double x, double y, double z) : mQ(w,x,y,z),mHasRot(true),mHasVec(false) {}
    Orientation(const QRot& q) : mQ(q),mHasRot(true),mHasVec(false) {}
    Orientation(double x, double y, double z) : mT(x,y,z),mHasRot(true),mHasVec(false) {}
    Orientation(const QRot& q, const Vec3D& t) : mQ(q),mT(t),mHasRot(true),mHasVec(true) {}

    bool hasRot() const { return mHasRot; }
    bool hasVec() const { return mHasVec; }

    void setRot(const QRot& q)  { mQ=q; mHasRot=true; }
    void setVec(const Vec3D& t) { mT=t; mHasVec=true; }
    void setRot(double w, double x, double y, double z) { setRot(QRot(w,x,y,z)); }
    void setVec(double x, double y, double z) { setVec(Vec3D(x,y,z)); }

    const QRot& q() const { return mQ; }
    const Vec3D& t() const { return mT; }

private:
    QRot mQ;
    Vec3D mT;
    bool mHasRot;
    bool mHasVec;
};

class Trajectory : public Orientation
{
public:
    typedef std::vector<Trajectory> List;

    Trajectory() {}

    Trajectory(timestamp_t timestamp, const std::string& device_id)
        : mTimestamp(timestamp),mDevice_id(device_id) {}

    timestamp_t timestamp() const { return mTimestamp; }
    std::string device_id() const { return mDevice_id; }


    static Trajectory::List read(const Path &path);

private:
    timestamp_t mTimestamp;
    std::string mDevice_id;
};

class Rig : public Orientation
{
public:
    typedef std::vector<Rig> List;

    Rig() {}
    Rig(const std::string& rig_devide_id, const std::string& sensor_device_id)
        : mRig_device_id(rig_devide_id),mSensor_device_id(sensor_device_id) {}

    std::string rig_device_id() const { return mRig_device_id; }
    std::string sensor_device_id() const { return mSensor_device_id; }

    static Rig::List read(const Path &path);

private:
    std::string mRig_device_id;
    std::string mSensor_device_id;
};


} // namespace Kapture

#endif // KPT_SENSORS_H
