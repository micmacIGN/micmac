#ifndef  _MMVII_TIEP_H_
#define  _MMVII_TIEP_H_

#include "MMVII_Ptxd.h"
#include "MMVII_util_tpl.h"

namespace MMVII
{

class c1ConfigLogMTP
{
     public :
         c1ConfigLogMTP();

         void SetIndIm(const  std::vector<int> & );
         void SetNbPts(size_t);
         void SetIdP0(size_t);
         size_t NbPts() const; /// Accessor

         void  AddData(const cAuxAr2007 & anAux);

         void NewIndIm(std::vector<int> & aNewInd,const std::vector<int> & aLut) const;


     private :
        std::vector<int>  mIndIm;
        size_t mIdP0;
        size_t mNbPts;
};

class cGlobConfLogMTP
{
      public :
          cGlobConfLogMTP(std::vector<std::string> &,size_t aNbConfig);
          cGlobConfLogMTP();

          void Resize();
          c1ConfigLogMTP & KthConf(size_t aK);
          const std::vector<c1ConfigLogMTP> & Configs() const; ///< Accessor

          void  AddData(const cAuxAr2007 & anAux);

          const std::vector<std::string> &  VNamesIm() const;
      private :
          std::vector<std::string>  mVNamesIm;
          std::vector<c1ConfigLogMTP>  mConfigs;
};

void AddData(const cAuxAr2007 & anAux,c1ConfigLogMTP & aConf);
void AddData(const cAuxAr2007 & anAux,cGlobConfLogMTP & aGlobConf);


// Not finish, theoretically adpated to  "big" data with optimization by pre-compiled index

class cReadMTP_Large
{
      public :
              cReadMTP_Large(const std::vector<std::string> aVNames,const std::string aNameConfig);
      public :
};



};


#endif //  _MMVII_TIEP_H_
