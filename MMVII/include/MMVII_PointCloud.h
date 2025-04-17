#ifndef  _MMVII_POINT_CLOUD_H_
#define  _MMVII_POINT_CLOUD_H_

// #include "MMVII_AllClassDeclare.h"
#include "MMVII_Ptxd.h"



namespace MMVII
{

class cAppli_ImportTxtCloud;
class cPointCloud
{
    public :
        cPointCloud(bool isMode8=true);

        int NbPts() const
        {
            return + (mMode8 ?  mPtsR.size() : mPtsF.size()) ;
        }
        void AddPt(const cPt3dr& aPt)
        {
             if (mMode8)
                mPtsR.push_back(aPt- mOffset);
             else
                mPtsF.push_back(cPt3df::FromPtR(aPt - mOffset));
        }


        cPt3dr  KthPt(int aK) const
        {
            return  mOffset + ( mMode8 ? 
                                mPtsR.at(aK) : 
                                cPt3dr(mPtsF.at(aK).x(),mPtsF.at(aK).y(),mPtsF.at(aK).z()));
        }
        void AddData(const  cAuxAr2007 & anAux) ;
        cBox3dr  Box() const;
        void Clip(cPointCloud&,const cBox2dr & aBox) const;
	void SetOffset(const cPt3dr & anOffset);

    private :
        std::vector<std::vector<std::string>>  mParams;   /// for further data
        cPt3dr                 mOffset;
        bool                   mMode8;
        std::vector<cPt3dr>    mPtsR;
        std::vector<cPt3df>    mPtsF;
};
void AddData(const  cAuxAr2007 & anAux,cPointCloud & aPC);


};

#endif  //  _MMVII_POINT_CLOUD_H_
