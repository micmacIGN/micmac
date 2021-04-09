#ifndef IMAGERECORDS_H
#define IMAGERECORDS_H

#include "kpt_common.h"

namespace Kapture {

class ImageRecord;
typedef std::vector<ImageRecord> ImageRecordList;

class ImageRecord
{
public:
    ImageRecord();
    ImageRecord(uint32_t timestamp, const std::string& device, const Path& image) : mTimestamp(timestamp),mDevice(device),mImage(image) {}

    uint32_t timestamp() const { return mTimestamp; }
    std::string device() const { return mDevice; }
    Path image() const { return mImage; }

private:
    uint32_t mTimestamp;
    std::string mDevice;
    Path mImage;
};

}

#endif // IMAGERECORDS_H
