#include "kpt_project.h"
#include <fstream>
#include <regex>
#include "kpt_internal.h"

namespace Kapture {

Project Project::theProject;

static inline Path relativePath(PathType type)
{
    switch (type) {
    case SENSORS_FILE : return "sensors/sensors.txt";
    case TRAJECTORIES_FILE: return "sensors/trajectories.txt";
    case RIGS_FILE: return "sensors/rigs.txt";
    case RECORDS_CAMERA_FILE: return "sensors/records_camera.txt";
    case RECORDS_DIR: return "sensors/records_data";
    case POINTS3D_FILE: return "reconstruction/points3d.txt";
    case KEYPOINTS_DIR: return "reconstruction/keypoints";
    case KEYPOINTS_FILE: return "reconstruction/keypoints/keypoints.txt";
    case DESCRIPTOR_DIR: return "reconstruction/descriptors";
    case DESCRIPTORS_FILE: return "reconstruction/descriptors/descriptors.txt";
    case GLOBAL_FEATURES_FILE: return "reconstruction/global_features/global_features.txt";
    case OBSERVATIONS_FILE: return "reconstruction/observations.txt";
    case MATCHES_DIR: return "reconstruction/matches";
    }
    errorf(Error,"Internal Error: Path type %u unexpected",(unsigned) type);
}

Path Project::path(const Path& relPath) const
{
    return root() / relPath;
}

Path Project::path(PathType pathType) const
{
    return path(relativePath(pathType));
}


void Project::setRoot(const Path path)
{
    mRoot = path;
    if (! checkVersion())
        errorf(Error,"'%s': not a supported version.",this->path(SENSORS_FILE).string().c_str());
    mKeypointsDef = KeypointsDef();
    mCameras.clear();
    mImageRecords.clear();
    mTrajectories.clear();
    mRigs.clear();
}

void Project::load()
{
    // TODO: traiter errreurs ? (pas de traj, de rigs, ...
    mCameras = readCameras();
    mTrajectories = readTrajectories();
    mRigs = readRigs();
    mImageRecords = readImageRecords();
    for (auto& i : mImageRecords) {
        i.setCamera(camera(i.device()));
        i.setTrajectory(trajectory(i.timestamp(),i.device()));
    }
}


std::string Project::currentVersion()
{
    return KAPTURE_FORMAT_CURRENT;
}


std::string Project::version()
{
    if (mVersion.size())
        return mVersion;

    std::ifstream is(path(SENSORS_FILE));
    if (!is)
        errorf(Error,"%s not found",path(SENSORS_FILE).string().c_str());
    std::string firstLine;
    if (! getline(is,firstLine))
        return mVersion;

    std::regex re(KAPTURE_FORMAT_PARSING_RE);
    std::smatch matches;

    if (!std::regex_match(firstLine,matches,re))
        return mVersion;
    mVersion = matches[1].str();
    return mVersion;
}

bool Project::checkVersion()
{
    if (version() == KAPTURE_FORMAT_1_0)
        return true;
    return false;
}



const Camera *Project::camera(const std::string &device)
{
    for (auto& c : mCameras) {
        if (c.device() == device)
            return &c;
    }
    return nullptr;
}

const Trajectory *Project::trajectory(timestamp_t timestamp, const std::string &device)
{
    for (auto& t : mTrajectories) {
        if (t.timestamp() == timestamp && t.device() == device)
            return &t;
    }
    return nullptr;
}

PathList Project::imagesMatch(const std::string &name) const
{
    PathList paths;
    std::regex re(name,std::regex::icase);

    for (const auto& i : imageRecords()) {
        if (std::regex_search(i.image().string(),re))
                paths.emplace_back(i.image());
    }
    return paths;
}

Path Project::imageName(const std::string &name) const
{
    PathList paths=imagesMatch(name);
    return paths.size() == 1 ? paths[0] : Path();
}

Path Project::imagePath(const Path &path) const
{
    return this->path(RECORDS_DIR) / path;
}

Path Project::imagePath(const std::string &name) const
{
    auto path = imageName(name);
    return name.empty() ? Path() : imagePath(path);
}

Path Project::imagePath(const char *name) const
{
    return imagePath(std::string(name));
}



std::vector<std::string> Project::parseLine(const std::string line)
{
    std::vector<std::string> result;
    size_t pos = 0;
    size_t wordStart;
    size_t wordEnd;

    // Nota: std::string is garanteed to be null-terminated (C++ 11)
    while (::isspace(line[pos])) pos++;
    if (line[pos] == 0 || line[pos] == '#')     // empty or comment line
        return result;

    while (true) {
        wordStart = pos;
        while (line[pos] != ',' && line[pos] != 0) pos++;
        wordEnd = pos;
        if (wordEnd > wordStart) {
            while (::isspace(line[--wordEnd]));
            result.push_back(line.substr(wordStart, wordEnd - wordStart+1));
        } else {
            result.push_back("");
        }
        if (line[pos] == 0)
            return result;
        pos++;
        while (::isspace(line[pos])) pos++;
    }
}

void Project::csvParse(const Path &path, unsigned nbMinValue,
                       const std::vector<std::pair<unsigned,std::string>> matches,
                       std::function<bool(const StringList& values, const std::string& fName, unsigned line)> f)
{
    std::string line;
    std::ifstream is(path);
    std::vector<std::pair<unsigned,std::regex>> filters;
    unsigned nLine = 0;

    if (! is)
        return;

    for (const auto& [pos,s]:  matches) {
        if (s.size()) {
            try {
                filters.emplace_back(make_pair(pos,std::regex(s,std::regex::icase)));
            } catch (std::regex_error&) {
                errorf (Error,"Bad regex filter '%s'",s.c_str());
            }
        }
    }

    while (getline(is, line)) {
        nLine++;
        auto values = Project::parseLine(line);
        if (values.size() == 0)
            continue;
        for (const auto& [pos,re] : filters) {
            if (pos < values.size() && ! std::regex_search(values[pos],re))
                goto nextLine;
        }
        if (values.size() < nbMinValue)
            errorf(Error,"In '%s', line %u has only %lu elements (at least %u required)",path.string().c_str(),nLine,values.size(), nbMinValue);
        for (const auto& [pos,re] : filters) {
            if (pos >= values.size() && ! std::regex_search(values[pos],re))
                goto nextLine;
        }
        if (! f(values,path.string(),nLine))
            return;
nextLine:;
    }
}

void Project::csvParse(PathType pType, unsigned nbMinValue,
                       const std::vector<std::pair<unsigned, std::string> > matches,
                       std::function<bool (const StringList &, const std::string& fName, unsigned line)> f) const
{
    csvParse(path(pType), nbMinValue, matches, f);
}


bool Project::readKeypointDef()
{
    csvParse (KEYPOINTS_FILE,3,{},
                      [this](const StringList& values,...) {
        this->mKeypointsDef.name = values[0];
        this->mKeypointsDef.type = dtypeFromStr(values[1]);
        this->mKeypointsDef.size = std::stoul(values[2]);
        return true;
    });

    if (mKeypointsDef.type != DType::Unknown && mKeypointsDef.size < 256)
        return true;
    mKeypointsDef.type = DType::Unknown;
    mKeypointsDef.size = 0;
    return false;
}

const KeypointsDef& Project::keypointsDef()
{
    if (! mKeypointsDef.hasBeenRead())
        readKeypointDef();
    return mKeypointsDef;
}

void Project::keypointsDefCheck()
{
    if (keypointsDef().type == DType::Unknown)
        errorf(Error,"Invalid or unsupported keypoints description %s",path(KEYPOINTS_FILE).string().c_str());

}

SensorList Project::readSensors(const std::string& id,  const std::string& name,  const std::string& type) const
{
    SensorList sensors;
    csvParse (SENSORS_FILE,3,{{0,id},{1,name},{2,type}},
              [&sensors](const StringList& values,...) {
        sensors.emplace_back(values[0],values[1],values[2],values.begin()+3, values.end());
        return true;
    });
    return sensors;
}

SensorList Project::readSensors() const
{
    return readSensors("","","");
}


CameraList Project::readCameras(const std::string &id, const std::string &name, const std::string &model) const
{
    CameraList cameras;

    csvParse (SENSORS_FILE,4,{{0,id},{1,name},{2,"CAMERA"},{3,model}},
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

TrajectoryList Project::readTrajectories(const std::string &device, int64_t min, int64_t max) const
{
    TrajectoryList trajectories;

    csvParse (TRAJECTORIES_FILE,9,{{1,device}},
              [&trajectories,min,max](const StringList& values, const std::string& fName, unsigned line) {
        try {
            timestamp_t timestamp = std::stoul(values[0]);
            if (min>=0 && timestamp < min)
                return true;
            if (max>=0 && timestamp > max)
                return true;
            Trajectory traj(timestamp,values[1]);
            if (values[2]!="" && values[3] != "" && values[4] != "" && values[5] != "")
                traj.setRot(stod(values[2]),stod(values[3]),stod(values[4]),stod(values[5]));
            if (values[6]!="" && values[7] != "" && values[8] != "")
                traj.setVec(stod(values[6]),stod(values[7]),stod(values[8]));
            trajectories.emplace_back(traj);
        } catch (...) {
            errorf(Error, "Can't create Trajectory from %s line %d",fName.c_str(),line);
        }
        return true;
    });
    return trajectories;
}

TrajectoryList Project::readTrajectories() const
{
    return readTrajectories("",-1,-1);
}

RigList Project::readRigs(const std::string &name, const std::string &device) const
{
    RigList rigs;

    csvParse (RIGS_FILE,9,{{0,name},{1,device}},
              [&rigs](const StringList& values, const std::string& fName, unsigned line) {
        try {
            Rig rig(values[0],values[1]);
            if (values[2]!="" && values[3] != "" && values[4] != "" && values[5] != "")
                rig.setRot(stod(values[2]),stod(values[3]),stod(values[4]),stod(values[5]));
            if (values[6]!="" && values[7] != "" && values[8] != "")
                rig.setVec(stod(values[6]),stod(values[7]),stod(values[8]));
            rigs.emplace_back(rig);
        } catch (...) {
            errorf(Error, "Can't create Rig from %s line %d",fName.c_str(),line);
        }
        return true;
    });
    return rigs;
}

RigList Project::readRigs() const
{
    return readRigs("","");
}


CameraList Project::readCameras() const
{
    return readCameras("","","");
}

ImageRecordList Project::readImageRecords(const std::string &device, const std::string &path, int64_t min, int64_t max) const
{
    ImageRecordList poses;

    csvParse (RECORDS_CAMERA_FILE,3,{{1,device},{2,path}},
              [&poses,min,max](const StringList& values, const std::string& fName, unsigned line) {
        try {
            timestamp_t timestamp = std::stoul(values[0]);
            if (min>=0 && timestamp < min)
                return true;
            if (max>=0 && timestamp > max)
                return true;
            poses.emplace_back(timestamp,values[1],values[2]);
        } catch (...) {
            errorf(Error, "Can't create Camera from %s line %d",fName.c_str(),line);
        }
        return true;
    });
    return poses;
}

ImageRecordList Project::readImageRecords() const
{
    return readImageRecords("","");
}


Path Project::matchPath(Path img1, Path img2) const
{
    if (img1 > img2)
        std::swap(img1,img2);
    img1 += ".overlapping";
    img2 += ".matches";
    return path(MATCHES_DIR) / img1 / img2;
}

void Project::prepareReadMatches(const Path& image1, const Path& image2, std::ifstream& mStream, std::ifstream& kpt1, std::ifstream& kpt2, bool& swapImg)
{
    keypointsDefCheck();

    Path img1 = image1;
    Path img2 = image2;

    if (img2 < img1) {
        swap(img1,img2);
        swapImg = true;
    }

    Path kpt1Path = path(KEYPOINTS_DIR) / img1;
    kpt1Path += ".kpt";
    Path kpt2Path = path(KEYPOINTS_DIR) / img2;
    kpt2Path += ".kpt";

    Path matchFile = matchPath(img1,img2);
    mStream = std::ifstream(matchFile,  std::ios::in | std::ios::binary);
    if (!mStream)
        errorf (Error,"Can't read file %s",matchFile.string().c_str());

    kpt1 = std::ifstream(kpt1Path,  std::ios::in | std::ios::binary);
    if (!kpt1)
        errorf (Error,"Can't read file %s",kpt1Path.string().c_str());
    kpt2 = std::ifstream(kpt2Path,  std::ios::in | std::ios::binary);
    if (!kpt2)
        errorf (Error,"Can't read file %s",kpt2Path.string().c_str());
}

std::vector<std::pair<std::string,std::string>> Project::allCoupleMatches()
{
    keypointsDefCheck();
    std::vector<std::pair<std::string, std::string>> all;

    auto images = imagesMatch();
    for (auto img1 = images.cbegin(); img1 != images.cend(); img1++) {
        for (auto img2 = std::next(img1); img2 != images.cend(); img2++) {
            if (*img1 > *img2)
                continue;
            Path matchFile = matchPath(*img1,*img2);
            std::ifstream ms(matchFile, std::ios::binary | std::ios::ate);
            if (ms && ms.tellg() > 0)
                all.emplace_back(make_pair(img1->string(),img2->string()));
        }
    }
    return all;
}



} // namespace Kapture
