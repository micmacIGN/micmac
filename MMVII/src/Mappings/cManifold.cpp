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
        static const int DimC = DimE - DimM,

        typedef cPtxd<tREAL8,DimM>    tPtM;
        typedef cPtxd<tREAL8,DimE>    tPtE;
        typedef std::array<tPtE,DimC> tTgSp;

        ///  Retun a point of manifold in embedding space kowning a parameter, default error
        virtual  tPtE   PtOfParam(const tPtM &) const ;
        ///  "Inverse" of PtOfParam, Assuming tPtE is on manifold
        virtual  tPtM   ParamOfPt(const tPtE &) const ;

        /// return TgSpace
        virtual tTgSp  TgSpace(const tPtE &) const;
     private :
};



template class cManifold<1,2>; // curve 2d
template class cManifold<1,3>; // curve 3d



};

