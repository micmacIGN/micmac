#ifndef MMVII_GENARGSSPEC_H
#define MMVII_GENARGSSPEC_H

#include "MMVII_enums.h"
#include <string>

namespace MMVII {

class cGenArgsSpecContext {
public:
    cGenArgsSpecContext(eTA2007 firstFileType, eTA2007 lastFileType, eTA2007 firstDirType, eTA2007 lastDirType)
    : firstFileType(firstFileType),lastFileType(lastFileType),firstDirType(firstDirType),lastDirType(lastDirType)
    {}

    std::string jsonSpec;
    std::string errors;
    eTA2007 firstFileType;
    eTA2007 lastFileType;
    eTA2007 firstDirType;
    eTA2007 lastDirType;
};

} // namespace MMVII
#endif // MMVII_GENARGSSPEC_H
