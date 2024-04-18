#ifndef  _MMVII_Tiling_H_
#define  _MMVII_Tiling_H_

#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"

namespace MMVII
{

template <const int Dim> class cTilingIndex;
template <class Type>  class  cTiling ;

/* **************************************** */
/*                                          */
/*              cTilingIndex                */
/*                                          */
/* **************************************** */

/**  Mother class of tilings, dont contain object, depend only of dimension
 *
 *   Allow the computation 
 *       Reaal Coordinate  in R^n -> index of tile in N^n -> linear index in N
 */

template <const int Dim> class cTilingIndex : cMemCheck
{
       public :
           typedef cPtxd<int,Dim>        tIPt;
           typedef cPtxd<tREAL8,Dim>     tRPt;
           typedef cPixBox<Dim>          tIBox;
           typedef cTplBox<tREAL8,Dim>   tRBox;
           typedef cSegment<tREAL8,Dim>  tSeg;

           typedef std::list<int>        tLIInd;

	   /// Constructor from Bounding Box and target number of case, WihBoxOut indicate if we allow to put point outside box
	   cTilingIndex(const tRBox &,bool WithBoxOut, int aNbCase); // NCase= targeted total, not by dimension

	   /// Number of tiles
	   size_t  NbElem() const;
           bool  OkOut() const;  ///<  Accessor 
           const tRBox & Box() const;

	   /// Convert  R^n -> N^n
	   tIPt  PtIndex(const tRPt &) const;

	   const tIBox &  IBoxIn() const {return mIBoxIn;} ///< accessor
        protected :
	   /// Convert  R^n -> N
	   int   IIndex(const tRPt &) const;

	   /// return the list of index of tiles overlaping the box
	   tLIInd  GetCrossingIndexes(const tRBox &) const;  // 2 define
	   /// return the index of the box corresponding to 1 point, it's a list to have same interface as with box
	   tLIInd  GetCrossingIndexes(const tRPt &)  const;

	   void AssertInside(const tRPt &) const;

        private :

	   cTilingIndex(const cTilingIndex<Dim> &) = delete;
	   void operator = (const cTilingIndex<Dim> &) = delete;
	   /// Helper in constructor
	   static tREAL8  ComputeStep(const tRBox &, int aNbCase);
	  
	   // tIPt  Index(const tRPt &) const;
	   tRBox    mRBoxIn;  ///< copy of in box
	   bool     mOkOut;   ///< is it ok if object are out
	   int      mNbCase;  ///< targeted total number of case/small boxx

	   tREAL8   mStep;  ///< computed step 
	   tIPt     mSzI;   ///< number of case in each dim
	   tIBox    mIBoxIn; ///< box number + add a margin for object outside
};

/*  For fast retrieving of object in tiling at given point position we test equality with a
 *  point;  but this equality can be called between, for ex, point and segemnt, so we define this special
 *  equality function that behave as an equality for 2 points, and generate error else (because exact
 *  retrieving from a point cannot be uses)
 */
template <class Type,const int Dim>  bool EqualPt(const Type &,const cPtxd<tREAL8,Dim> & aPt)
{
  MMVII_INTERNAL_ERROR("Called EqualPt with bad primitive");

  return false;
}

template <const int Dim>  bool EqualPt(const cPtxd<tREAL8,Dim> & aP1,const cPtxd<tREAL8,Dim> & aP2)
{
      return (aP1==aP2);
}

/* **************************************** */
/*                                          */
/*              cTilingIndex                */
/*                                          */
/* **************************************** */

/**  Tiling for geometric indexing
 *
 *    Herit from cTilingIndex + contains the object themself. Main services :
 *
 *        - Add an object
 *        - list of object in a certain geometric region  (as GetObjAtDist)
 *        - object if any at an exact position (GetObjAtPos)
 */

template <class Type>  class  cTiling : public cTilingIndex<Type::Dim>
{
     public :
           typedef cTilingIndex<Type::Dim>  tTI;
           typedef typename tTI::tRBox      tRBox;
           typedef typename tTI::tRPt       tRPt;

           typedef typename Type::tPrimGeom tPrimGeom;
           typedef typename Type::tArgPG    tArgPG;
	   static constexpr int Dim = Type::Dim;

           typedef std::list<Type>          tCont1Tile;
           typedef std::vector<tCont1Tile>  tVectTiles;

	   cTiling(const tRBox & aBox,bool WithBoxOut, int aNbCase,const tArgPG & anArg):
	       tTI     (aBox,WithBoxOut,aNbCase),
	       mVTiles (this->NbElem()),
	       mArgPG  (anArg)
           {
           }
	   void Add(const Type & anObj)
	   {
               // Check eventually that object is include in box-define
               if (! this->OkOut())
               {
                  tRBox aBox = anObj.GetPrimGeom(mArgPG).GetBoxEnglob();
	          this->AssertInside(aBox.P0());
	          this->AssertInside(aBox.P1());
               }

               //  Put object in all  box that it crosses
               for (const auto &  aInd :  tTI::GetCrossingIndexes(anObj.GetPrimGeom(mArgPG))  )
               {
                   mVTiles.at(aInd).push_back(anObj);
               }
	   }

	   /// this method can be used only when tPrimGeom is a point
           void GetObjAtPos(std::list<Type*>& aRes,const tRPt & aPt)
           {
                aRes.clear();

                for (auto & anObj : mVTiles.at(this->IIndex(aPt)) )
                {
                    if (EqualPt(anObj.GetPrimGeom(mArgPG),aPt))
                       aRes.push_back(&anObj);
                }
           }
           /// Same method, when there is 0 or 1 object
           Type* GetObjAtPos(const tRPt & aPt)
           {
               std::list<Type*> aL;
               GetObjAtPos(aL,aPt);

               if (aL.empty()) return nullptr;
               MMVII_INTERNAL_ASSERT_tiny(aL.size()==1,"GetObjAtPos : multiple object");
               return  *(aL.begin());
           }

	   /// return list of object at given dist
	   template <class tPrimG2> std::list<Type*> GetObjAtDist(const tPrimG2 &aPrimG2,tREAL8 aDist)
	   {
                 std::list<Type*> aRes;
		 tRBox  aBox = aPrimG2.GetBoxEnglob().Dilate(aDist);  // Get indices of boxes that crosse englobing box
		 for (const auto & anInd : this->GetCrossingIndexes(aBox))
		 {
                      for (auto & anObj : mVTiles.at(anInd)) // Parse all obj of each tile
		      {
                           if (aPrimG2.InfEqDist(anObj.GetPrimGeom(mArgPG),aDist))
                              aRes.push_back(&anObj);
		      }
		 }
                 return aRes;
	   }
     private :
	   tVectTiles  mVTiles;
	   tArgPG      mArgPG;
};

/**  Sometime we only need to put points in a spatial index, this
 * class make the interface allowing to use simple point in spatial indexing
 */

template <const int TheDim> class cPointSpInd
{
    public :
        static constexpr int Dim = TheDim;
        typedef cPtxd<tREAL8,TheDim>  tPrimGeom;
        typedef int     tArgPG;  /// unused here

        const tPrimGeom & GetPrimGeom(int Arg=-1) const {return mPt;}

        cPointSpInd(const tPrimGeom & aPt) :
           mPt (aPt)
        {
        }

    private :
         tPrimGeom  mPt;
};


/** Class for generating point such that all pairs are at distance > given value */
template <const int TheDim> class cGeneratePointDiff
{
     public :
           static constexpr int Dim = TheDim;

           typedef cPtxd<tREAL8,Dim>     tRPt;
           typedef cTplBox<tREAL8,Dim>   tRBox;
           typedef cPointSpInd<Dim>      tPSI;
           typedef cTiling<tPSI>         tTiling;

           cGeneratePointDiff(const tRBox & aBox,tREAL8 aDistMin,int aNbMax=1000) ;
	   ///  generate a new point
           tRPt GetNewPoint(int aNbTest=1000);

     private :
           int      mNbMax;
           tTiling  mTiling;
           tREAL8   mDistMin;
};



};
#endif //   _MMVII_Tiling_H_
