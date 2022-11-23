#ifndef  _MMVII_ZBUFFER_H_
#define  _MMVII_ZBUFFER_H_

#include "MMVII_AllClassDeclare.h"



namespace MMVII
{

/**   This abstract class is used to decribe an object containing many triangles.
 *
 * In Z-Buffer, we can use an explicit mesh, but also an implicit one if we parse an image
 * where each pixel is made from two triangle. This implicit class allow to maipulate the two
 * object in the same interface (an avoid converting the pixels in hundred million of triangles ...)
 */

class  cTri3DIterator
{
     public :
        virtual bool GetNextTri(tTri3dr &) = 0;
        virtual bool GetNextPoint(cPt3dr &) = 0;
        virtual void ResetTri()  = 0;
        virtual void ResetPts()  = 0;

        ///  Down cast
        virtual cCountTri3DIterator * CastCount();

        void ResetAll() ;
     private :
};

/** in many case, the implementation can be done by counters */

class cCountTri3DIterator : public cTri3DIterator
{
     public :
        cCountTri3DIterator(size_t aNbP,size_t aNbF);

        virtual cPt3dr  KthP(int aKP) const = 0;
        virtual tTri3dr KthF(int aKF) const = 0;

        bool GetNextTri(tTri3dr &) override;
        bool GetNextPoint(cPt3dr &) override;
        void ResetTri()  override;
        void ResetPts()  override;
        cCountTri3DIterator * CastCount() override;

     private :
        size_t  mNbP;
        size_t  mNbF;
        size_t  mIndexF;
        size_t  mIndexP;
};

/** A mesh as counter triangle iterator */

class cMeshTri3DIterator : public cCountTri3DIterator
{
     public :
        cMeshTri3DIterator(cTriangulation3D<tREAL8> *);

        cPt3dr  KthP(int aKP) const override;
        tTri3dr KthF(int aKF) const override;
     private :
        cTriangulation3D<tREAL8> *  mTri;
};

/**  Enum for stories the state of a triangle after zbuffer computation */

enum class eZBufRes
           {
              Undefined,      ///< to have some value to return when nothing is computed
              UnRegIn,        ///< Un-Regular since input
              OutIn,          ///< Out domain  since input (current ?)
              UnRegOut,       ///< Un-Regular since output
              OutOut,         ///< Out domain  since output (never )
              BadOriented,    ///< Badly oriented
              Hidden,         ///< Hidden by other
              NoPix,          ///< When there is no pixel in triangle, decision har to do
              LikelyVisible,  ///< Probably visible -> No Pixel but connected to visible
              Visible         ///< Visible
           };

bool  ZBufLabIsOk(eZBufRes aLab); // return if visible or likely

/**  Enum for stories the kind of computation to do inside a zbuffer */

enum class eZBufModeIter
           {
               ProjInit,   ///< initial projection (compute Z Buffer itself)
               SurfDevlpt  ///< computed visibility + resolution of visible tri
           };

/** Result for each triangle in surface devlpt */

struct cResModeSurfD
{
    public :
        eZBufRes mResult;
        double   mResol   ;
};
void  AddData(const cAuxAr2007  &anAux,cResModeSurfD& aRMS ); ///< serialization


class  cZBuffer
{
      public :

          typedef tREAL4                            tElem;
          typedef cDataIm2D<tElem>                  tDIm;
          typedef cIm2D<tElem>                      tIm;
          typedef cIm2D<tU_INT1>                    tImSign;
          typedef cDataIm2D<tU_INT1>                tDImSign;

          typedef cDataInvertibleMapping<tREAL8,3>  tMap;
          typedef cDataBoundedSet<tREAL8,3>         tSet;

          static constexpr tElem mInfty =  -1e10;

	  ///  constructor
          cZBuffer(cTri3DIterator & aMesh,const tSet & aSetIn,const tMap & aMap,const tSet & aSetOut,double aResolOut);


	  /// 
          void MakeZBuf(eZBufModeIter aMode);


          const cPt2di  SzPix() ; ///< Accessor
          tIm   ZBufIm() const; ///< Accessor
          cResModeSurfD&  ResSurfD(size_t) ;   ///< accessor to VecResSurfD
          std::vector<cResModeSurfD> & VecResSurfD() ;   ///< accessor
          double  MaxRSD() const;                        ///< accessor
          bool  IsOk() const;                            ///< accessor

      private :
          cZBuffer(const cZBuffer & ) = delete;
          void AssertIsOk() const;
          cPt2dr  ToPix(const cPt3dr&) const;  /// Output coord-> pix coord
          /// compute resolution between  resol (in worst dir) between a  3D tri in and a 2D tri Out (in facts proj of the 3D out)
          double ComputeResol(const tTri3dr & aTriIn ,const tTri3dr & aTriOut) const;
	  ///  make the job for one triangle, different computation possible depending on aMode
          eZBufRes MakeOneTri(const tTri3dr & aTriIn,const tTri3dr & aTriOut,eZBufModeIter aMode);

          bool                  mIsOk;       ///< Many things can happen bad ...
          bool                  mZF_SameOri; ///< Axe of Z (in out coord) and oriented surface have same orientation
          int                   mMultZ;      ///< multiplier associated to SameOri
          cTri3DIterator &      mMesh;       ///<  The mesh described as an iterator on triangles
          cCountTri3DIterator * mCountMesh;  ///< posible cast on a counting-iteraror (unused 4 now)
          const tMap &          mMapI2O;     ///<  Map Input -> Output 
          const tSet &          mSetIn;      ///< Set where input triangles are defined
          const tSet &          mSetOut;     ///< Set where output triangles are defined
          double                mResolOut;   ///<  Resolution for computing Z Buf (ratio  )

          cBox3dr          mBoxIn;     ///< Box in input space, not sure usefull, but ....
          cBox3dr          mBoxOut;    ///< Box in output space, usefull for xy, not sure for z , but ...
          cHomot2D<tREAL8> mROut2Pix;  ///<  Mapping Out Coord -> Pix Coord
          tIm              mZBufIm;    ///<  Image storing the Z buffer its (max or min depending on mZF_SameOri)
          tImSign          mImSign;   ///< sign of normal  1 or -1 , 0 if uninit
          cPt2di           mSzPix;      ///<  sz of pixels images (sign, zbuf)

          double           mLastResSurfDev;  ///< store the last resolution computed (facility 4 multiple results)
          double           mMaxRSD;          ///< max resolution reacged

          std::vector<cResModeSurfD>  mResSurfD;
};





};

#endif  //  _MMVII_ZBUFFER_H_
