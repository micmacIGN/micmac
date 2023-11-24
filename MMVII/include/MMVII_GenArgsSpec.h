#ifndef MMVII_GENARGSSPEC_H
#define MMVII_GENARGSSPEC_H

#include "MMVII_enums.h"
#include <string>

namespace MMVII {

class cGenArgsSpecContext {
public:
    cGenArgsSpecContext(
        const std::vector<eTA2007> & prjSubDirList,
        const std::map<eTA2007,std::vector<std::string>> & fileTypes
    )
    : prjSubDirList(prjSubDirList),fileTypes(fileTypes)
    {}

    std::string jsonSpec;
    std::string errors;
    std::vector<eTA2007> prjSubDirList;
    std::map<eTA2007,std::vector<std::string>> fileTypes;
};

} // namespace MMVII
#endif // MMVII_GENARGSSPEC_H
