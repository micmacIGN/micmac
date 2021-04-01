#ifndef KPT_COMMON_H
#define KPT_COMMON_H

#if __cplusplus < 201703L
#error C++ 17 is needed
#endif

#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include <stdexcept>


namespace Kapture {

extern bool debugOn;

const std::string KAPTURE_FORMAT_1_0 = "1.0";

const std::string KAPTURE_FORMAT_CURRENT = KAPTURE_FORMAT_1_0;
const std::string KAPTURE_FORMAT_HEADER = "# kapture format: " + KAPTURE_FORMAT_CURRENT;
const std::string KAPTURE_FORMAT_PARSING_RE = "# kapture format\\s*:\\s*(\\d+\\.\\d+)\\s*";

typedef std::filesystem::path Path;
typedef std::vector<Path> PathList;
typedef std::vector<std::string> StringList;

enum class DType{Unknown,UINT8,UINT16,UINT32,UINT64,FLOAT32,FLOAT64};

const char *dtypeToStr(DType t);
DType dtypeFromStr(const std::string &s);

std::vector<char> readBinaryFile(std::istream& is);
std::vector<char> readBinaryFile(const Path& p);

class Error : public std::runtime_error
{
public:
    Error(const std::string& errorMsg,
              const std::string& file, size_t line, const std::string& func);

    std::string errorMsg() const { return mErrorMsg; }
    std::string file() const { return mFile; }
    std::string func() const { return mFunc; }
    size_t line() const { return mLine; }

private:
    const std::string& mErrorMsg;
    const std::string& mFile;
    size_t mLine;
    const std::string& mFunc;
};


} // namespace Kapture

#endif // KPT_COMMON_H
