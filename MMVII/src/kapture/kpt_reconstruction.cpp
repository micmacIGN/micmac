#include "kpt_reconstruction.h"
#include "kpt_internal.h"

#include <fstream>


namespace Kapture {


template<typename T>
typename Points3D<T>::List Points3D<T>::read(const Path& path)
{
    Points3D<T>::List points;

    csvParse (path,{3,6},
              [&points](const StringList& values, const std::string& fName, unsigned line) {
        try {
            Points3D<T> pt(stold(values[0]),stold(values[1]),stold(values[2]));
            if (values.size() == 6)
                pt.setRGB(stoul(values[3]),stoul(values[4]),stoul(values[5]));
            points.push_back(pt);
        } catch (...) {
            errorf(Error, "Can't create Points3D from %s line %d",fName.c_str(),line);
        }
        return true;
    });
    return points;
}

// Explicit instanciations
template class Points3D<float>;
template class Points3D<double>;
template class Points3D<long double>;

Observations::List Observations::read(const Path& path)
{
    Observations::List obs;

    csvParse (path,{4},
              [&obs](const StringList& values, const std::string& fName, unsigned line) {
        try {
            obs.emplace_back(stoul(values[0]),values[1],values[2],stoul(values[3]));
        } catch (...) {
            errorf(Error, "Can't create Points3D from %s line %d",fName.c_str(),line);
        }
        return true;
    });
    return obs;
}


KeypointsType KeypointsType::read(const Kapture::Path &path)
{
    KeypointsType kpt;

    csvParse (path,{3},[&kpt](const StringList& values,...) {
        kpt = KeypointsType(values[0],dtypeFromStr(values[1]),std::stoul(values[2]));
        return true;
    });

    if (kpt.dtype() != DType::Unknown)
        return kpt;
    return KeypointsType();
}

template<typename T>
Keypoints<T> read(const Path& path, int dsize)
{
    std::ifstream is(path, std::ios::binary);
    if (! is)
        errorf(Error,"Can't read file %s",path.string().c_str());

    is.seekg(0, std::ios::end);
    std::streamsize fSize = is.tellg();
    if ( fSize % (dsize * sizeof(T)) != 0)
        errorf(Error,"File %s is not a valid keypoints file",path.string().c_str());

    Keypoints<T> kpt(dsize,fSize / (sizeof (T) * dsize));
    is.seekg(0, std::ios::beg);
    is.read((char*)kpt.data(), fSize);
}


// Explicit instanciations
template class Keypoints<float>;
template class Keypoints<double>;
template class Keypoints<long double>;


DescriptorsType DescriptorsType::read(const Path &path)
{
    DescriptorsType desc;

    csvParse (path,{5},[&desc](const StringList& values,...) {
        desc = DescriptorsType(values[0],
                dtypeFromStr(values[1]),
                std::stoul(values[2]),
                values[3],values[4]);
        return true;
    });

    if (desc.dtype() != DType::Unknown)
        return desc;
    return DescriptorsType();
}

template<typename T>
void Descriptors<T>::read(const Path &path, int dsize)
{
    std::ifstream is(path, std::ios::binary);
    if (! is)
        errorf(Error,"Can't read file %s",path.string().c_str());

    is.seekg(0, std::ios::end);
    std::streamsize fSize = is.tellg();
    if ( fSize % (dsize * sizeof(T)) != 0)
        errorf(Error,"File %s is not a valid keypoints file",path.string().c_str());

    mDSize = dsize;
    mData.resize(fSize / sizeof(T));
    is.seekg(0, std::ios::beg);
    is.read((char*)mData.data(), fSize);
}

// Explicit instanciations
template class Descriptors<float>;
template class Descriptors<double>;
template class Descriptors<long double>;


GlobalFeaturesType GlobalFeaturesType::read(const Path &path)
{
    GlobalFeaturesType gf;

    csvParse (path,{4},[&gf](const StringList& values,...) {
        gf = GlobalFeaturesType(values[0],
                dtypeFromStr(values[1]),
                std::stoul(values[2]),
                values[3]);
        return true;
    });

    if (gf.dtype() != DType::Unknown)
        return gf;
    return GlobalFeaturesType();

}

// Explicit instanciations
template class GlobalFeatures<float>;
template class GlobalFeatures<double>;
template class GlobalFeatures<long double>;

Matches::List Matches::read(const Path &path)
{
    Matches::List ml;

    std::ifstream is(path, std::ios::binary);
    if (! is)
        errorf(Error,"Can't read file %s",path.string().c_str());

    is.seekg(0, std::ios::end);
    std::streamsize fSize = is.tellg();
    if ( fSize % sizeof(Matches) != 0)
        errorf(Error,"File %s is not a valid matches file",path.string().c_str());

    ml.resize(fSize / sizeof(Matches));
    is.seekg(0, std::ios::beg);
    is.read((char*)ml.data(), fSize);
    return ml;
}



} // namespace Kapture
