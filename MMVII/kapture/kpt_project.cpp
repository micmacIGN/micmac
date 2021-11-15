#include "kpt_project.h"
#include <fstream>
#include <regex>
#include "kpt_internal.h"

namespace Kapture {

Project Project::theProject;


void Project::setRoot(const Path path)
{
    mRoot = path;
    mVersion = "";
}

void Project::load()
{
    // FIXME CM: traiter errreurs ? (pas de traj, de rigs, ...
    if (! checkVersion())
        errorf(Error,"'%s': not a supported version.",localPath(sensorsPath()).string().c_str());

    try  {
        mCameras = readCameras();
    } catch (Error& e) {
        std::cerr << "WARNING: Cameras: "<< e.what() << "\n";
        mCameras.clear();
    }

    try  {
        mTrajectories = readTrajectories();
    } catch (Error& e) {
        std::cerr << "WARNING: Trajectories: "<< e.what() << "\n";
        mTrajectories.clear();
    }

    try  {
        mRigs = readRigs();
    } catch (Error& e) {
        std::cerr << "WARNING: Rigs: "<< e.what() << "\n";
        mRigs.clear();
    }


    try  {
        mRecordsCamera = readRecordsCamera();
    } catch (Error& e) {
        std::cerr << "WARNING: RecordsCamera: "<< e.what() << "\n";
        mRigs.clear();
    }

    for (auto& i : mRecordsCamera) {
        i.setCamera(camera(i.device_id()));
        i.setTrajectory(trajectory(i.timestamp(),i.device_id()));
    }
}

void Project::save(const Path& rpath)
{
    std::error_code ec;

    mRoot = rpath;
    Path dir = localPath(sensorsDir());
    std::filesystem::create_directories(dir, ec );
    std::cerr << "ec : " << ec.value() << "\n";
    if (ec.value()) {
        std::cerr << "ec msg : " << ec.value() << ":" << ec.message() << "\n";
        exit (1);
    }
    std::ofstream os(localPath(sensorsPath()));
    os << KAPTURE_FORMAT_HEADER << "\n" ;
    os << "# sensor_id, name, sensor_type, [sensor_params]+\n";
    for (const auto& c : cameras()) {
        os << c.sensor_device_id() << ", " << c.name() << ", " << c.typeStr();
        for (const auto& p : c.sensor_params()) {
            os << ", " << p;
        }
        os << "\n";
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

    std::ifstream is(localPath(sensorsPath()));
    if (!is)
        errorf(Error," %s not found",localPath(sensorsPath()).string().c_str());
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
    return KAPTURE_FORMAT_SUPPORTED.find(version()) != KAPTURE_FORMAT_SUPPORTED.end();
}



const Camera *Project::camera(const std::string &device)
{
    for (auto& c : mCameras) {
        if (c.sensor_device_id() == device)
            return &c;
    }
    return nullptr;
}

const Trajectory *Project::trajectory(timestamp_t timestamp, const std::string &device)
{
    for (auto& t : mTrajectories) {
        if (t.timestamp() == timestamp && t.device_id() == device)
            return &t;
    }
    return nullptr;
}

PathList Project::imagesMatch(const std::string &re) const
{
    PathList paths;
    std::regex regex(re,std::regex::icase);

    for (const auto& i : imageRecords()) {
        if (std::regex_search(i.image_path().string(),regex))
                paths.emplace_back(i.image_path());
    }
    return paths;
}

Path Project::imageMatch(const std::string &re) const
{
    PathList paths=imagesMatch(re);
    return paths.size() == 1 ? paths[0] : Path();
}

Path Project::imagePath(const Path &path) const
{
    return localPath(recordsDataDir()) / path;
}

Path Project::imagePath(const std::string &re) const
{
    auto path = imageMatch(re);
    return path.empty() ? Path() : imagePath(path);
}

Path Project::imagePath(const char *re) const
{
    return imagePath(std::string(re));
}


std::vector<std::pair<std::string,std::string>> Project::allCoupleMatches(const std::string& match_type)
{
    std::vector<std::pair<std::string, std::string>> all;

    auto images = imagesMatch();
    for (auto img1 = images.cbegin(); img1 != images.cend(); img1++) {
        for (auto img2 = std::next(img1); img2 != images.cend(); img2++) {
            if (*img1 > *img2)
                continue;
            Path matchFile = localPath(matchesPath(*img1,*img2, match_type));
            std::ifstream ms(matchFile, std::ios::binary | std::ios::ate);
            if (ms && ms.tellg() > 0)
                all.emplace_back(make_pair(img1->string(),img2->string()));
        }
    }
    return all;
}



#if 0

void Project::prepareReadMatches(const Path& image1, const Path& image2, std::ifstream& mStream, std::ifstream& kpt1, std::ifstream& kpt2, bool& swapImg)
{
    keypointsDefCheck();

}


#endif

} // namespace Kapture
