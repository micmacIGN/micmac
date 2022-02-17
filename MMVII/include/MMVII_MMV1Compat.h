#ifndef  _MMVII_MMV1_Compat_H_
#define  _MMVII_MMV1_Compat_H_

namespace MMVII
{

/** \file MMVII_MMV1Compat.h
    \brief Function/class to make communication between MMVII and MMv1

*/

tNameSet  MMV1InitSet(const std::string & aName);
tNameRel  MMV1InitRel(const std::string & aName);


//  Defined in MMVII_Stringifier.h for Serialization
// template<class Type> void  MMv1_SaveInFile(const Type & aVal,const std::string & aName)
// template<> void  MMv1_SaveInFile(const tNameSet & aVal,const std::string & aName);




};

#endif  //  _MMVII_MMV1_Compat_H_
