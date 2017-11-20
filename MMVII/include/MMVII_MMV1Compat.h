#ifndef  _MMVII_MMV1_Compat_H_
#define  _MMVII_MMV1_Compat_H_

namespace MMVII
{

/** \file MMVII_MMV1Compat.h
    \brief Function/class to make communication between MMVII and MMv1

*/

void  MMV1InitSet(std::vector<std::string> & aRes,const std::string & aName);
void  MMV1SaveSet(const std::vector<std::string> & aRes,const std::string & aName);


};

#endif  //  _MMVII_MMV1_Compat_H_
