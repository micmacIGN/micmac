#include "global.h"
#include <QRegularExpression>

bool showDebug = false;


#if !defined(WIN32) && !defined(_WIN32)
QString quotedArg(const QString& aParam)
{
    constexpr const char* SingleQuoting="!$`\\";
    constexpr const char* DoubleQuoting=" \t~#*?()[]{}<>;&|\"";

    if (aParam.size() == 0)
        return "\"\"";
    if (aParam.toStdString().find_first_of(SingleQuoting) != std::string::npos) {
        QString result="'";
        for (const auto& c: aParam) {
            if (c=='\'')
                result += "'\\''";
            else
                result += c;
        }
        return result + "'";
    }
    if (aParam.toStdString().find_first_of(DoubleQuoting) != std::string::npos) {
        QString result="\"";
        for (const auto& c: aParam) {
            if (c=='"')
                result += "\\\"";
            else
                result += c;
        }
        return result + "\"";
    }
    return aParam;
}

#else
QString quotedArg(const QString& aParam)
{
    constexpr const char* DoubleQuoting=" \t*?&|()<>^\"";

    if (aParam.size() == 0)
        return "\"\"";
    if (aParam.toStdString().find_first_of(DoubleQuoting) != std::string::npos) {
        QString result="\"";
        for (const auto& c: aParam) {
            if (c=='"')
                result += "\\\"";
            else
                result += c;
        }
        return result + "\"";
    }
    return aParam;
}
#endif
