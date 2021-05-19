#include "kpt_sensors.h"
#include "kpt_project.h"
#include "kpt_internal.h"
#include <array>
#include <regex>
#include <functional>
#include <fstream>

#include <initializer_list>

namespace Kapture {

// First elemt must be default value
static EnumAndStr<Sensor::Type> sensorTypeNames = {
    MAKE_TENUM_STR(Sensor,UNKNOWN_TYPE),
    MAKE_TENUM_STR(Sensor,CAMERA),
    MAKE_TENUM_STR(Sensor,DEPTH),
    MAKE_TENUM_STR(Sensor,GNSS),
    MAKE_TENUM_STR(Sensor,LIDAR),
    MAKE_TENUM_STR(Sensor,WIFI),
    MAKE_TENUM_STR(Sensor,MAGNETIC),
    MAKE_TENUM_STR(Sensor,PRESSURE),
};

static EnumAndStr<Camera::Model> cameraModelNames = {
    MAKE_TENUM_STR(Camera,UNKNOWN_CAMERA),
    MAKE_TENUM_STR(Camera,SIMPLE_PINHOLE),
    MAKE_TENUM_STR(Camera,PINHOLE),
    MAKE_TENUM_STR(Camera,SIMPLE_RADIAL),
    MAKE_TENUM_STR(Camera,RADIAL),
    MAKE_TENUM_STR(Camera,OPENCV),
    MAKE_TENUM_STR(Camera,OPENCV_FISHEYE),
    MAKE_TENUM_STR(Camera,FULL_OPENCV),
    MAKE_TENUM_STR(Camera,FOV),
    MAKE_TENUM_STR(Camera,SIMPLE_RADIAL_FISHEYE),
    MAKE_TENUM_STR(Camera,RADIAL_FISHEYE),
    MAKE_TENUM_STR(Camera,THIN_PRISM_FISHEYE),
};


Sensor::Sensor(const std::string &id, const std::string &name, const std::string &typeStr,
               StringList::const_iterator first, StringList::const_iterator last)
    : mDevice(id), mName(name),mType(typeFromStr(typeStr))
{
    if (mType == UNKNOWN_TYPE)
        mTypeStr = typeStr;
    while (first != last) {
        mParams.push_back(*first);
        first++;
    }
}



Sensor::Type Sensor::typeFromStr(const std::string &s)
{
    return Type(sensorTypeNames.fromStr(s));
}

std::string Sensor::typeToStr(Sensor::Type type)
{
    return sensorTypeNames.toStr(type);
}



Camera::Camera(const std::string &id, const std::string &name, const StringList::const_iterator &begin, const StringList::const_iterator &end)
    : Camera(Sensor(id,name,"CAMERA",begin,end))
{
}

Camera::Camera(const Sensor &sensor) : Sensor(sensor)
{
    if (params().size() < 1 || type() != CAMERA)
        errorf(Error,"Can't convert sensor '%s', type '%s' to Camera", sensor.device().c_str(), sensor.typeStr().c_str());
    mModel = modelFromStr(params()[0]);
    PosixLocale setPosix;
    for (size_t i=1; i< params().size(); i++)
        mModelParams.push_back(std::stod(params()[i]));
}


Camera::Model Camera::modelFromStr(const std::string &s)
{
    return Model(cameraModelNames.fromStr(s));
}


std::string Camera::modelToStr(Camera::Model model)
{
    return cameraModelNames.toStr(model);
}




} // namespace Kapture
