#ifndef FEATURES_H
#define FEATURES_H

#include <fstream>
#include <vector>

namespace Kapture {


struct Match {
    Match (float x1, float y1, float x2, float y2) : x1(x1),y1(y1),x2(x2),y2(y2) {}

    float x1,y1;
    float x2,y2;
};

typedef  std::vector<Match> MatchList;



template<typename Storage>
class AllocatedOrMapped {
public:
    typedef typename Storage::value_type value_type;
    explicit AllocatedOrMapped(int n) : mData(n) {}
    explicit AllocatedOrMapped(const Storage& s) : mData(s) {}

    size_t bytes() const { return  sizeof (value_type) * mData.size();}

    void remap(void *data) { mData.remap(data); }
    void remap(const void *data) { mData.remap(data); }

    value_type* data() { return mData.data();}
    const value_type* data() const { return mData.data();}

    value_type operator[](int n)  const { return mData[n];}
    value_type & operator[](int n)  { return mData[n];}

private:
    Storage mData;
};


template<typename T>
class MappedStorage
{
public:
    typedef T value_type;

    MappedStorage(int n, void *d=nullptr) : mNbData(n), mData((value_type*)d) {}

    void remap(void *data) { mData = (value_type*)data;}
    void remap(const void *data) { mData = (value_type*)data;}
    size_t size() const { return mNbData;}

    value_type operator[](size_t n) const {return mData[n];}
    value_type& operator[](size_t n) {return mData[n];}
    value_type *data() { return mData;}
    const value_type *data() const { return mData;}

private:
    size_t mNbData;
    value_type *mData;
};

template<typename T>
class Keypoint : public AllocatedOrMapped<T>
{
public:
    explicit Keypoint(int n) : AllocatedOrMapped<T>(n) {}
    typename AllocatedOrMapped<T>::value_type x() const {return this->data()[0];}
    typename AllocatedOrMapped<T>::value_type y() const {return this->data()[1];}
};

template<typename T>
class KeypointVec : public Keypoint<std::vector<T>>
{
public:
    explicit KeypointVec(int n) : Keypoint<std::vector<T>>(n) {}

};

template<typename T>
class KeypointMap : public Keypoint<MappedStorage<T>>
{
public:
    explicit KeypointMap(int n) : Keypoint<std::vector<T>>(n) {}

};




} // namespace Kapture


#endif // FEATURES_H
