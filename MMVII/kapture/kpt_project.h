#ifndef KPY_PROJECT_H
#define KPY_PROJECT_H

#include <algorithm>
#include "kpt_common.h"
#include "kpt_sensors.h"
#include "kpt_reconstruction.h"


namespace Kapture {


class Project;

// **********************************************************************
// At this time, all errors throw en exception of type Kapture::Error() !
// **********************************************************************


// Singleton project global to the process if needed
inline Project& project();
inline void setProject(const Path& path);
inline void setProject(const Project& project);


class Project
{
public:
    Project () {}

    // Set the "rootPath" of the kapture dataset. Can be absolute or relative.
    explicit Project (const Path& rootPath) { setRoot(rootPath);}
    void setRoot(const Path rootPath);
    Path root() const {return mRoot;}

    void load();
    void save(const Path& rpath);

    const Camera::List& cameras() const { return mCameras;}
    const Trajectory::List& trajectories() const { return mTrajectories;}
    const Rig::List& rigs() const { return mRigs;}
    const RecordsCamera::List& imageRecords() const { return mRecordsCamera;}

    Sensor::List readSensors() const { return Sensor::read(localPath(sensorsPath())); }
    Camera::List readCameras() const { return Camera::read(localPath(sensorsPath())); };
    Trajectory::List readTrajectories() const { return Trajectory::read( localPath(trajectoriesPath())); }
    Rig::List readRigs() const { return Rig::read( localPath(rigsPath())); }
    RecordsCamera::List readRecordsCamera() const { return RecordsCamera::read( localPath(recordsCameraPath())); }

    const Camera* camera(const std::string& device) ;
    const Trajectory* trajectory(timestamp_t timestamp, const std::string& device);
    const Rig rig(const std::string rigID);


    // Kapture version of the dataset
    std::string version();
    // Last supported version of the kapture format
    static std::string currentVersion();

    // Return pathname relative to rootPath/records_data of all images matching re(regex) .
    PathList imagesMatch(const std::string &re="") const;
    // Return pathname relative to rootPath/records_data of the unique image matching re(regex).
    // If none or several match, return empty path
    Path     imageMatch(const std::string &re) const;

    // Return absolute path of an image path : rootPath / records_data / path
    Path     imagePath(const Path &path) const;

    // Return absolute path of an image re (regex) : rootpath / records_data / imageMatch(name)
    Path     imagePath(const std::string &re) const;
    Path     imagePath(const char *re) const;


    // Homologous points
    std::vector<std::pair<std::string,std::string>> allCoupleMatches(const std::string &match_type);

    // MATCH is a class which must have a constructor accepting 4 numbers (float/double) : MATCH(float x1, float y1, float x2, float y2)
    // image path must be relative to rootPath / records_data (use imageName() or returns from readImageRecords() )
    template<typename PAIR>
    void readMatches(const Path& image1, const Path& image2, const std::string& keypoint_type, std::vector<PAIR>& pairs);

    // Same, can be used as :  auto matchList = readMatches<MyMatchType>(img1,img2)
    template<typename PAIR>
    std::vector<PAIR> readMatches(const Path& image1, const Path& image2, const std::string& keypoint_type);

    // Return a std::vector<Kapture::Match>,  Kapture::Match is a simple struct of 4 floats: x1,y1,x2,y2 (see: kpt_features.h)
    Pair::List readMatches(const Path& image1, const Path& image2, const std::string& keypoint_type)
    {
        return readMatches<Pair> (image1,image2,keypoint_type);
    }


private:
    // Is dataset a supported kapture version ?
    bool checkVersion();

    Path localPath(const Path& path) const { return root() / path; }


    void prepareReadMatches(const Path& image1, const Path& image2, std::ifstream& mStream, std::ifstream& kpt1, std::ifstream& kpt2, bool& swapImg);

    template<typename T, typename M>
    void doReadMatches(Path image1, Path image2,  const std::string& keypoints_type,
                       const KeypointsType& kType,
                       std::vector<M>& pairs);

    friend Project& project();
    friend void setProject(const Path& path);
    friend void setProject(const Project& project);

    static Project theProject;
    Path mRoot;
    std::string mVersion;

    Camera::List mCameras;
    RecordsCamera::List mRecordsCamera;
    Trajectory::List mTrajectories;
    Rig::List mRigs;
};


// Singleton project global to the process
inline Project& project() { return  Project::theProject;}
inline void setProject(const Path& path) { Project::theProject.setRoot(path) ;}
inline void setProject(const Project& project) { Project::theProject = project ;}



// Impl

template<typename T, typename M>
void Project::doReadMatches(Path image1, Path image2,  const std::string& keypoints_type,
                            const KeypointsType& kType,
                            std::vector<M>& pairs)
{
    bool swapImg = false;
    pairs.clear();
    if (image2 < image1) {
        swap(image1,image2);
        swapImg = true;
    }

    Path kpt1Path = localPath(keypointsPath(image1,keypoints_type));
    Path kpt2Path = localPath(keypointsPath(image2,keypoints_type));
    Path matchFile = localPath(matchesPath(image1,image2,keypoints_type));

    auto kpt1 = Keypoints<T>::read(kpt1Path,kType.dsize());
    auto kpt2 = Keypoints<T>::read(kpt1Path,kType.dsize());
    auto matches = Matches::read(matchFile);

    for (const auto &match : matches) {
        if (swapImg) {
            pairs.emplace_back(kpt2.x(match.idx2()),kpt2.y(match.idx2()),kpt1.x(match.idx1()),kpt1.y(match.idx1()));
        } else {
            pairs.emplace_back(kpt1.x(match.idx1()),kpt1.y(match.idx1()),kpt2.x(match.idx2()),kpt2.y(match.idx2()));
        }
    }
}



template<typename PAIR>
void Project::readMatches(const Path& image1, const Path& image2, const std::string& keypoint_type, std::vector<PAIR>& pairs)
{
    auto kType = KeypointsType::read(localPath(keypointsTypePath(keypoint_type)));

    switch (kType.dtype()) {
    case DType::FLOAT32: doReadMatches<float>(image1,image2,keypoint_type,kType, pairs); break;
    case DType::FLOAT64: doReadMatches<double>(image1,image2,keypoint_type,kType, pairs); break;
    default: throw Error(std::string("Unsupported data type ") + dtypeToStr(kType.dtype()) + " for keypoints", __FILE__,__LINE__, __func__);
    }
}


template<typename PAIR>
std::vector<PAIR> Project::readMatches(const Path& image1, const Path& image2, const std::string& keypoint_type)
{
    std::vector<PAIR> pairs;
    this->readMatches(image1,image2,keypoint_type,pairs);
    return pairs;
}

} // namespace Kapture

#endif // KPY_PROJECT_H
