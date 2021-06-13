#ifndef KPY_PROJECT_H
#define KPY_PROJECT_H

#include <algorithm>
#include <functional>
#include "kpt_common.h"
#include "kpt_sensors.h"
#include "kpt_features.h"


namespace Kapture {


class Project;

// **********************************************************************
// At this time, all errors throw en exception of type Kapture::Error() !
// **********************************************************************


// Singleton project global to the process if needed
inline Project& project();
inline void setProject(const Path& path);
inline void setProject(const Project& project);



enum PathType {
    SENSORS_FILE, TRAJECTORIES_FILE, RIGS_FILE, RECORDS_CAMERA_FILE,
    POINTS3D_FILE,
    RECORDS_DIR,
    KEYPOINTS_DIR, KEYPOINTS_FILE,
    DESCRIPTOR_DIR, DESCRIPTORS_FILE,
    GLOBAL_FEATURES_FILE,
    OBSERVATIONS_FILE,
    MATCHES_DIR
};

struct KeypointsDef {
    std::string name;
    DType type;
    int size;
    KeypointsDef() : type(DType::Unknown),size(-1) {}
    bool hasBeenRead() const { return size>=0;}
};


class Project
{
public:
    Project () {}

    // Set the "rootPath" of the kapture dataset. Can be absolute or relative.
    explicit Project (const Path& rootPath) { setRoot(rootPath);}
    void setRoot(const Path rootPath);
    Path root() const {return mRoot;}

    void load();

    const CameraList& cameras() const { return mCameras;}
    const Camera* camera(const std::string& device) ;
    const TrajectoryList& trajectories() const { return mTrajectories;}
    const Trajectory* trajectory(timestamp_t timestamp, const std::string& device);
    const RigList& rigs() const { return mRigs;}
    const Rig rig(const std::string rigID);

    const ImageRecordList& imageRecords() const { return mImageRecords;}

    // Kapture version of the dataset
    std::string version();
    // Last supported version of the kapture format
    static std::string currentVersion();

    // Return full pathname (relative to rootPath/records_data) of all images matching (regex) name.
    PathList imagesMatch(const std::string &name="") const;
    // Return full relative pathname of the image matching (regex) name.
    // If none or several match, return empty path
    Path     imageName(const std::string &name) const;

    // Return "absolute path" of an image path : rootPath / records_data / path
    Path     imagePath(const Path &path) const;

    // Return "absolute path" of an image name (regex) : rootpath / records_data / imageName(name)
    Path     imagePath(const std::string &name) const;
    Path     imagePath(const char *name) const;


    // Homologous points

    std::vector<std::pair<std::string,std::string>> allCoupleMatches();

    // MATCH is a class which must have a constructor accepting 4 numbers (float/double) : MATCH(float x1, float y1, float x2, float y2)
    // image path must be relative to rootPath / records_data (use imageName() or returns from readImageRecords() )
    template<typename MATCH>
    void readMatches(const Path& image1, const Path& image2,std::vector<MATCH>& matches);

    // Same, can be used as :  auto matchList = readMatches<MyMatchType>(img1,img2)
    template<typename MATCH>
    std::vector<MATCH> readMatches(const Path& image1, const Path& image2);

    // Return a std::vector<Kapture::Match>,  Kapture::Match is a simple struct of 4 floats: x1,y1,x2,y2 (see: kpt_features.h)
    MatchList readMatches(const Path& image1, const Path& image2)
    {
        return readMatches<Match> (image1,image2);
    }


    // Homologous points, "slow" API: don't load full keypoints file in memory
    template<typename MATCH>
    void readMatchesSlow(const Path& image1, const Path& image2,std::vector<MATCH>& matches);

    template<typename MATCH>
    std::vector<MATCH> readMatchesSlow(const Path& image1, const Path& image2);

    MatchList readMatchesSlow(const Path& image1, const Path& image2)
    {
        return readMatchesSlow<Match> (image1,image2);
    }

private:
    const KeypointsDef &keypointsDef();
    void keypointsDefCheck();
    bool readKeypointDef();

    // Is dataset a supported kapture version ?
    bool checkVersion();

    // Get full path of a file or dir inside dataset: return rootPath / relPath
    Path path(const Path& relPath) const;
    // Same for conventional path : path(SENSORS_FILE) return rootPath / "sensors/sensors.txt"
    Path path(PathType pathType) const;


    static StringList parseLine(const std::string line);

    static void csvParse(const Path &path, unsigned nbMinValue,
                         const std::vector<std::pair<unsigned, std::string> > matches,
                         std::function<bool(const StringList& values, const std::string& fName, unsigned line)> f);
    void csvParse(PathType pType, unsigned nbMinValue,
                         const std::vector<std::pair<unsigned, std::string> > matches,
                         std::function<bool(const StringList& values, const std::string& fName, unsigned line)> f) const;

    // Return all sensors present in sensors.txt (device, name, type, params)
    SensorList readSensors() const;
    // Return sensors matching id, name and type (regex). Empty string match all
    SensorList readSensors(const std::string& id,  const std::string& name,  const std::string& type) const;

    // Return all cameras present in sensors.txt (device, name, CAMERA, model, modelParams)
    CameraList readCameras() const;
    // Filter by regex
    CameraList readCameras(const std::string &id, const std::string &name, const std::string &model) const;

    // Return all trajectories (timestamp, device, quaternion, pos)
    TrajectoryList readTrajectories() const;
    // Return trajectories filtered by regex device or by min/max timetamp
    TrajectoryList readTrajectories(const std::string &device, int64_t min=-1, int64_t max=-1) const;

    // Return all rigs (name, device, quaternion, pos)
    RigList readRigs() const;
    // Return rigs matching name and device (regex)
    RigList readRigs(const std::string &name, const std::string &device) const;

    // Return all image records (timestamp, device, imageName)
    ImageRecordList readImageRecords() const;
    // Return image records filtered by regex or by min/max timetamp
    ImageRecordList readImageRecords(const std::string &device, const std::string &path,int64_t min=-1, int64_t max=-1) const;


    Path matchPath(Path img1, Path img2) const;
    void prepareReadMatches(const Path& image1, const Path& image2, std::ifstream& mStream, std::ifstream& kpt1, std::ifstream& kpt2, bool& swapImg);

    template<typename T, typename M>
    static void doReadMatches(std::istream& mStream, std::istream& kpt1, std::istream& kpt2, unsigned featureSize, bool swapImg, std::vector<M>& matches);

    template<typename T, typename M>
    static void doReadMatchesSlow(std::istream& mStream, std::istream& kpt1, std::istream& kpt2, unsigned featureSize, bool swapImg, std::vector<M>& matches);

    friend Project& project();
    friend void setProject(const Path& path);
    friend void setProject(const Project& project);

    static Project theProject;
    Path mRoot;
    KeypointsDef mKeypointsDef;
    std::string mVersion;

    CameraList mCameras;
    ImageRecordList mImageRecords;
    TrajectoryList mTrajectories;
    RigList mRigs;
};


// Singleton project global to the process
inline Project& project() { return  Project::theProject;}
inline void setProject(const Path& path) { Project::theProject.setRoot(path) ;}
inline void setProject(const Project& project) { Project::theProject = project ;}



// Impl

template<typename T, typename M>
void Project::doReadMatches(std::istream& mStream, std::istream& kpt1, std::istream& kpt2, unsigned featureSize, bool swapImg, std::vector<M>& matches)
{
    matches.clear();
    struct {
        double key1,key2,score;
    } match;

    auto kpt1Vector = readBinaryFile(kpt1);
    auto kpt2Vector = readBinaryFile(kpt2);

    Keypoint<MappedStorage<T>> k1(featureSize);
    Keypoint<MappedStorage<T>> k2(featureSize);

    while (mStream.read((char*)&match, sizeof match)) {
        k1.remap(kpt1Vector.data() + (int)match.key1 * k1.bytes());
        k2.remap(kpt2Vector.data() + (int)match.key2 * k2.bytes());
        if (swapImg) {
            matches.emplace_back(k2.x(),k2.y(),k1.x(),k1.y());
        } else {
            matches.emplace_back(k1.x(),k1.y(),k2.x(),k2.y());
        }
    }
}



template<typename MATCH>
void Project::readMatches(const Path& image1, const Path& image2,std::vector<MATCH>& matches)
{
    bool swapImg;
    std::ifstream mStream,kpt1,kpt2;

    prepareReadMatches(image1, image2, mStream, kpt1, kpt2, swapImg);

    switch (keypointsDef().type) {
    case DType::FLOAT32: doReadMatches<float>(mStream,kpt1,kpt2,keypointsDef().size,swapImg, matches); break;
    case DType::FLOAT64: doReadMatches<double>(mStream,kpt1,kpt2,keypointsDef().size,swapImg, matches); break;
    default: throw Error(std::string("Unsupported data type ") + dtypeToStr(keypointsDef().type) + " for keypoints", __FILE__,__LINE__, __func__);
    }
}


template<typename MATCH>
std::vector<MATCH> Project::readMatches(const Path& image1, const Path& image2)
{
    std::vector<MATCH> matches;
    this->readMatches(image1,image2,matches);
    return matches;
}


template<typename T, typename M>
void Project::doReadMatchesSlow(std::istream& mStream, std::istream& kpt1, std::istream& kpt2, unsigned featureSize, bool swapImg, std::vector<M>& matches)
{
    matches.clear();
    struct {
        double key1,key2,score;
    } match;
    Keypoint<std::vector<T>> k1(featureSize);
    Keypoint<std::vector<T>> k2(featureSize);

    while (mStream.read((char*)&match, sizeof match)) {
        kpt1.seekg(match.key1 * k1.bytes());
        kpt1.read((char*)k1.data(), k1.bytes());
        kpt2.seekg(match.key2 * k2.bytes());
        kpt2.read((char*)k2.data(), k2.bytes());
        if (swapImg) {
            matches.emplace_back(k2.x(),k2.y(),k1.x(),k1.y());
        } else {
            matches.emplace_back(k1.x(),k1.y(),k2.x(),k2.y());
        }
    }
}


template<typename MATCH>
void Project::readMatchesSlow(const Path& image1, const Path& image2, std::vector<MATCH>& matches)
{
    bool swapImg;
    std::ifstream mStream,kpt1,kpt2;

    prepareReadMatches(image1, image2, mStream, kpt1, kpt2, swapImg);

    switch (keypointsDef().type) {
    case DType::FLOAT32: return doReadMatchesSlow<float>(mStream,kpt1,kpt2,keypointsDef().size,swapImg,matches);
    case DType::FLOAT64: return doReadMatchesSlow<double>(mStream,kpt1,kpt2,keypointsDef().size,swapImg,matches);
    default: throw Error(std::string("Unsupported data type ") + dtypeToStr(keypointsDef().type) + " for keypoints", __FILE__,__LINE__, __func__);
    }
}

template<typename MATCH>
std::vector<MATCH> Project::readMatchesSlow(const Path& image1, const Path& image2)
{
    std::vector<MATCH> matches;
    this->readMatchesSlow(image1,image2,matches);
    return matches;
}


} // namespace Kapture

#endif // KPY_PROJECT_H
