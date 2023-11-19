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
    {Sensor::UNKNOWN_TYPE,"unknown_type"},
    {Sensor::CAMERA,"camera"},
    {Sensor::DEPTH,"depth"},
    {Sensor::GNSS,"gnss"},
    {Sensor::LIDAR,"lidar"},
    {Sensor::WIFI,"wifi"},
    {Sensor::MAGNETIC,"magnetic"},
    {Sensor::PRESSURE,"pressure"},
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
    : mSensor_device_id(id), mName(name),mSensor_type(typeFromStr(typeStr))
{
    if (mSensor_type == UNKNOWN_TYPE)
        mTypeStr = typeStr;
    while (first != last) {
        mSensor_params.push_back(*first);
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


Sensor::List Sensor::read(const Path& path)
{
    Sensor::List sensors;
    csvParse (path,{-3},[&sensors](const StringList& values,...) {
        sensors.emplace_back(values[0],values[1],values[2],values.begin()+3, values.end());
        return true;
    });
    return sensors;
}



Camera::Camera(const std::string &id, const std::string &name, const StringList::const_iterator &begin, const StringList::const_iterator &end)
    : Camera(Sensor(id,name,"camera",begin,end))
{
}

Camera::Camera(const Sensor &sensor) : Sensor(sensor)
{
    if (sensor_params().size() < 1 || sensor_type() != CAMERA)
        errorf(Error,"Can't convert sensor '%s', type '%s' to Camera", sensor.sensor_device_id().c_str(), sensor.typeStr().c_str());
    mModel = modelFromStr(sensor_params()[0]);
    PosixLocale setPosix;
    for (size_t i=1; i< sensor_params().size(); i++)
        mModel_params.push_back(std::stod(sensor_params()[i]));
}


Camera::Model Camera::modelFromStr(const std::string &s)
{
    return Model(cameraModelNames.fromStr(s));
}


std::string Camera::modelToStr(Camera::Model model)
{
    return cameraModelNames.toStr(model);
}

Camera::List Camera::read(const Path &path)
{
    Camera::List cameras;

    csvParse (path,{-4},
              [&cameras](const StringList& values, const std::string& fName, unsigned line) {
        try {
            cameras.emplace_back(values[0],values[1],values.begin()+3, values.end());
        } catch (...) {
            errorf(Error, "Can't create Camera from %s line %d",fName.c_str(),line);
        }
        return true;
    });
    return cameras;
}


RecordsCamera::List RecordsCamera::read(const Path &path)
{
    RecordsCamera::List poses;

    csvParse (path,{3},
              [&poses](const StringList& values, const std::string& fName, unsigned line) {
        try {
            poses.emplace_back(std::stoul(values[0]),values[1],values[2]);
        } catch (...) {
            errorf(Error, "Can't create Camera from %s line %d",fName.c_str(),line);
        }
        return true;
    });
    return poses;
}



Trajectory::List Trajectory::read(const Path &path)
{
    Trajectory::List trajectories;

    csvParse (path,{2,6,9},
              [&trajectories](const StringList& values, const std::string& fName, unsigned line) {
        try {
            timestamp_t timestamp = std::stoul(values[0]);
            Trajectory traj(timestamp,values[1]);
            if (values.size() > 2 && (values[2]!="" && values[3] != "" && values[4] != "" && values[5] != ""))
                traj.setRot(stod(values[2]),stod(values[3]),stod(values[4]),stod(values[5]));
            if (values.size() > 6 && (values[6]!="" && values[7] != "" && values[8] != ""))
                traj.setVec(stod(values[6]),stod(values[7]),stod(values[8]));
            trajectories.push_back(traj);
        } catch (...) {
            errorf(Error, "Can't create Trajectory from %s line %d",fName.c_str(),line);
        }
        return true;
    });
    return trajectories;
}

Rig::List Rig::read(const Path &path)
{
    Rig::List rigs;

    csvParse (path,{2,6,9},
              [&rigs](const StringList& values, const std::string& fName, unsigned line) {
        try {
            Rig rig(values[0],values[1]);
            if (values.size() > 2 && (values[2]!="" && values[3] != "" && values[4] != "" && values[5] != ""))
                rig.setRot(stod(values[2]),stod(values[3]),stod(values[4]),stod(values[5]));
            if (values.size() > 6 && (values[6]!="" && values[7] != "" && values[8] != ""))
                rig.setVec(stod(values[6]),stod(values[7]),stod(values[8]));
            rigs.push_back(rig);
        } catch (...) {
            errorf(Error, "Can't create Rig from %s line %d",fName.c_str(),line);
        }
        return true;
        });
    return rigs;
}



} // namespace Kapture
