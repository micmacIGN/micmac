#ifndef GLOBAL_H
#define GLOBAL_H

#include <QString>

constexpr const char *APP_ORGANIZATION = "IGN-LASTIG";
constexpr const char *APP_NAME = "vMMVII";
constexpr const char *APP_VERSION = "1.0";

constexpr const char *MMVII_EXE_FILE = "MMVII";
constexpr const char *MMVII_LOG_FILE = "MMVII-LogFile.txt";

constexpr const char *MMVII_VMMVII_SPEC_ARG = "ExecFrom=vMMVII";


constexpr const char *OUTPUT_CONSOLE_INFO_COLOR = "blue";
constexpr const char *OUTPUT_CONSOLE_ERROR_COLOR = "red";

QString quotedArg(const QString& arg);

extern bool showDebug;

#endif // GLOBAL_H
