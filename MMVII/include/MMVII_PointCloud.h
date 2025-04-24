#ifndef  _MMVII_POINT_CLOUD_H_
#define  _MMVII_POINT_CLOUD_H_

// #include "MMVII_AllClassDeclare.h"
#include "MMVII_Ptxd.h"



namespace MMVII
{

class cAppli_ImportTxtCloud;
class cProjPointCloud;

class cPointCloud
{
    public :
        friend cAppli_ImportTxtCloud;
        friend cProjPointCloud;

        cPointCloud(bool isMode8=true);

        size_t NbPts() const
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

        cPt3dr  KthPtWoOffs(int aK) const
        {
            return  ( mMode8 ?  mPtsR.at(aK) : cPt3dr(mPtsF.at(aK).x(),mPtsF.at(aK).y(),mPtsF.at(aK).z()));
        }

        cPt3dr  KthPt(int aK) const { return  mOffset + KthPtWoOffs(aK); }

        void AddData(const  cAuxAr2007 & anAux) ;
        cBox3dr  Box() const;
        void Clip(cPointCloud&,const cBox2dr & aBox) const;
	void SetOffset(const cPt3dr & anOffset);

        void ToPly(const std::string & aName,bool WithOffset=false) const;

        tREAL8   Density() const;    ///< Accessor
        tREAL8   CurDensity() const; ///< density from current data, may differ from import
        cBox2dr  Box2d() const;      ///< Accessor

        void SetNbColours(int );
        int  GetNbColours() const;
        std::vector<tU_INT1> & GrayColors();

        void SetLeavesUnit(tREAL8 aPropAvgD,bool SVP);  ///< fix LeavesUnit && init mSzLeaves
        void SetSzLeaves(int aK,tREAL8 );
        tREAL8 GetSzLeave(int aK) const;
        tU_INT1 GetIntSzLeave(int aK) const;
        tREAL8  ConvertInt2SzLeave(int aK) const;
        bool  SzLeavesIsInit() const;


        void SetMulDegVis(tREAL8 );
        void SetDegVis(int aK,tREAL8 );
        tREAL8 GetDegVis(int aK) const;


    private :
        std::vector<std::vector<std::string>>  mParams;   ///< for further data
        cPt3dr                                 mOffset;   ///< to have more accuracy with 4-Bytes/Big coord
        bool                                   mMode8;    ///<  8-Bytes/4 Bytes 
        std::vector<cPt3dr>                    mPtsR;     ///<  8-Bytes points
        std::vector<cPt3df>                    mPtsF;     ///<  4 Bytes points
        std::vector<std::vector<tU_INT1>>      mColors;   ///<  Colors associated

        tREAL8                                 mMulDegVis; ///< convert int to [0 1]
        std::vector<tU_INT2>                   mDegVis;    ///<  As computed by Colorate

        tREAL8                                 mDensity;    ///<     Number of point / m2 (or whatever unit), computed at import
        cBox2dr                                mBox2d;      ///<
        tREAL8                                 mLeavesUnit; ///<
        std::vector<tU_INT1>                   mSzLeaves;   ///<  Size of each leaves
};
void AddData(const  cAuxAr2007 & anAux,cPointCloud & aPC);


};

#endif  //  _MMVII_POINT_CLOUD_H_
