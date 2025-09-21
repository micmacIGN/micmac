#ifndef __MMVII_MANIFOLDS_H_
#define __MMVII_MANIFOLDS_H_

#include "MMVII_Mappings.h"
#include "MMVII_Geom2D.h"


namespace MMVII
{

/**   
    DimM = Dim of Manifold, for ex 1 for curve
    DimE = Dim of embedding space, for ex 2 for a curve in plane
*/
template <const int DimM,const int DimE> class cManifold
{
     public :
        static const int DimC = DimE - DimM;

        typedef cPtxd<tREAL8,DimM>    tPtM;
        typedef cPtxd<tREAL8,DimE>    tPtE;
        typedef std::vector<tPtE>     tTgSp;
        typedef  std::pair<int,tPtM>                tResPOP; // Type result Param of Pt

        cManifold(int aNbMap=1,const tREAL8 aEpsDeriv = 1e-4);
        int NbMap() const; //< Accessor

        /// return the num map of a point, defaults assume mNbMap==1
        virtual int  GetNumMap(const tPtE &) const ;
        ///  Retun a point of manifold in embedding space kowning a parameter, default error
        virtual  tPtE   PtOfParam(const tResPOP &) const ;
        ///  "Inverse" of PtOfParam, Assuming tPtE is on manifold
        virtual  tResPOP   ParamOfPt(const tPtE &) const ;

        /// return TgSpace, assume aPtE is on the manifold
        virtual tTgSp  TgSpace(const tPtE & aPtE) const ;
        /// Given any point, gives an approximate value of the projection
        virtual tPtE  ApproxProj(const tPtE &) const  = 0 ;
        ///  Return the projection on the manifold, default iterative method
        virtual tPtE  Proj(const tPtE &) const  ;
        ///  Return the projection on the tangent space to a Manifolfd
     private :
        tPtE  OneIterProjByTgt(const tPtE & aP0,const tPtE & aP2Proj) const  ;

         int     mNbMap;
         tREAL8  mEpsDeriv;
};

template<int DimE> class cLineManifold : public cManifold<1,DimE>
{
     public :
        // virtual  tPtE   PtOfParam(const tPtM &,int aNumMap) const ;
        static const int DimM = 1;
        static const int DimC = DimE - DimM;

        typedef cPtxd<tREAL8,DimM>                  tPtM;
        typedef cPtxd<tREAL8,DimE>                  tPtE;
        typedef cSegmentCompiled<tREAL8,DimE>       tSeg;
        typedef  std::pair<int,tPtM>                tResPOP; // Type result Param of Pt

        tPtE   PtOfParam(const tResPOP &) const override;
        tResPOP   ParamOfPt(const tPtE &) const override;
        tPtE  ApproxProj(const tPtE &) const  override ;

   private :
        tSeg   mSeg;
};


template <const int DimM,const int DimE> class cManifoldFromMapping : public  cManifold<DimM,DimE>
{
     public :
        typedef cDataInvertibleMapping<tREAL8,DimE> tMap;
        typedef cManifold<DimM,DimE>                tMan;
        // virtual  tPtE   PtOfParam(const tPtM &,int aNumMap) const ;
        static const int DimC = DimE - DimM;

        typedef cPtxd<tREAL8,DimM>                  tPtM;
        typedef cPtxd<tREAL8,DimE>                  tPtE;
        typedef cSegmentCompiled<tREAL8,DimE>       tSeg;
        typedef  std::pair<int,tPtM>                tResPOP; // Type result Param of Pt

        cManifoldFromMapping(tMan*,tMap*);

        tPtE   PtOfParam(const tResPOP&) const override;
        tResPOP   ParamOfPt(const tPtE &) const override;
        tPtE  ApproxProj(const tPtE &) const  override ;



   private :
        tMan   *mMan;
        tMap   *mMap;

};

};

#endif //  __MMVII_MANIFOLDS_H_
