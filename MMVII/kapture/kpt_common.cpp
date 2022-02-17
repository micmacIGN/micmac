#include "kpt_internal.h"
#include "kpt_common.h"
#include <fstream>

namespace Kapture {

locale_t PosixLocale::posixLocale = (locale_t) 0;



static EnumAndStr<DType> dataTypeNames = {
    {DType::Unknown,"Unknown"},
    {DType::FLOAT32,"float32"},
    {DType::FLOAT32,"float"},
    {DType::FLOAT64,"float64"},
    {DType::FLOAT64,"double"},
//    {DType::UINT8,"uint8"},
//    {DType::UINT16,"uint16"},
//    {DType::UINT32,"uint32"},
//    {DType::UINT64,"uint64"},
};


const char *dtypeToStr(DType t)
{
    return dataTypeNames.toStr(t);
}

DType dtypeFromStr(const std::string &s)
{
    return dataTypeNames.fromStr(s);
}

Error::Error(const std::string &errorMsg, const std::string &file, size_t line, const std::string &func)
    : std::runtime_error(std::string("In function ") + func + ", at " + Path(file).filename().string() + ": " + std::to_string(line) + ": " + errorMsg),
      mErrorMsg(errorMsg),mFile(Path(file).string()),mLine(line),mFunc(func)
{
    std::cout.flush();
}

std::vector<char> readBinaryFile(std::istream& is)
{
    std::vector<char> data;

    if (! is)
        error(Error,"Can't read file");

    is.seekg(0, std::ios::end);
    std::streamsize size = is.tellg();
    is.seekg(0, std::ios::beg);

    data.resize(size);
    is.read(data.data(), size);
    return data;
}

std::vector<char> readBinaryFile(const Path& p)
{
    std::ifstream is(p, std::ios::binary);
    if (! is)
        errorf(Error,"Can't read file %s",p.string().c_str());
    return readBinaryFile(is);
}


} // namespave Kapture
