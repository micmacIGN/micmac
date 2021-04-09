#ifndef KPT_SENSOR_H
#define KPT_SENSOR_H

#include <string>
#include <vector>
#include <memory>

#include "kpt_common.h"

namespace Kapture {

class Sensor;
class Camera;
typedef std::vector<Sensor> SensorList;
typedef std::vector<Camera> CameraList;

class Sensor
{
public:
    enum Type {UNKNOWN_TYPE, CAMERA, DEPTH, GNSS, LIDAR, WIFI, MAGNETIC, PRESSURE};
    Sensor() : mType(UNKNOWN_TYPE) {}
    explicit Sensor(Type type) : mType(type) {}
    Sensor(const std::string& id, const std::string& name, const std::string& typeStr,
           StringList::const_iterator first, StringList::const_iterator last);

    std::string deviceId() const {return mDeviceId ;}
    std::string name() const { return mName;}
    Type type() const { return mType;}
    std::string typeStr() const  { return mType==UNKNOWN_TYPE ?  mTypeStr : typeToStr(mType);}
    StringList params() const { return mParams;}

private:
    static Type typeFromStr(const std::string &  s);
    static std::string typeToStr(Type typeStr);

    std::string mDeviceId;
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


} // namespace Kapture

#endif // KPT_SENSOR_H
