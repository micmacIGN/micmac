#include "include/MMVII_all.h"
#include "MatchTieP.h"

namespace MMVII
{

class cCdtBasic;            // Store a Aime PC + other stuff necessary for match
class cHypMatchCdtBasic;    // An hypothesis of match between 2 candidate
class cSetCdtBasic;         // All the Cdt of one type in on image
class cCpleSetCdtBasic ;    //  The two corresponding sets in two images
class cAppliMatchTiePBasic; // The application

/**  A class to store one hypothesi of match , contain 2 candidate and a cost,
     The two candidate will share the same cHypMatchCdtBasic
*/


class cHypMatchCdtBasic : public cMemCheck
{
     public :
        const double & Cost() const;  ///< Accessor to cost
        /** We need to know for a given Hyp who is the other candidate */
        cCdtBasic & Other(const cCdtBasic & MySelf);
        cHypMatchCdtBasic(double aCost,cCdtBasic & aCd1,cCdtBasic & aCd2,int aIndexOr);
        /// Std Comparison, the best hyp is the lowest cost
        bool operator < (const cHypMatchCdtBasic & aH2) const {return mCost<aH2.mCost;}
     private :
        double      mCost;     /// Cost/Distance between two candidates
        int         mIndexOr;  /// Index of the method used to compute orientation
        cCdtBasic * mCd1;      ///  Cd1
        cCdtBasic * mCd2;      ///  Cd2
};
typedef std::shared_ptr<cHypMatchCdtBasic>  tSPHypM;

/**  Descripor (Aime) + Hypothesis of match
*/
class cCdtBasic  : public cMemCheck
{
    public :
       cCdtBasic(cAimePCar *);
       /// Make a firt level of match by computing the NbSel best match
       void MatchInit(cSetCdtBasic&,int NbSel);
       cAimePCar &  PC(); // Accessor
    private :
       cAimePCar *             mPC;        ///  Aime descriptor
       std::vector<tSPHypM>    mHypMatch;  ///  Hypothesis of match
};

class cSetCdtBasic : public cMemCheck
{
    public :
        cSetCdtBasic(cSetAimePCAR *);
        cSetAimePCAR &           SetPC();  // Accessor
        std::vector<cCdtBasic>&  VCdt();   // Accessor
    private :
        cSetAimePCAR *          mSetPC;  // The initial set of Aime Points
        std::vector<cCdtBasic>  mVCdt;   // The structure that containt all the candidtes
};

class cCpleSetCdtBasic : public cMemCheck
{
    public :
        cCpleSetCdtBasic(cSetAimePCAR * aS1,cSetAimePCAR * aS2);
        void MakeCost();
    private :
        cSetCdtBasic  mS1;
        cSetCdtBasic  mS2;
};

/*  ******************************************************** */
/*                                                           */
/*              cCdtBasic                                    */
/*                                                           */
/*  ******************************************************** */

cCdtBasic:: cCdtBasic(cAimePCar * aPC) :
   mPC (aPC)
{
}

void cCdtBasic::MatchInit(cSetCdtBasic& aSet,int NbSel)
{
    static int aCpt=0;
    aCpt++;

    std::vector<cHypMatchCdtBasic> aVH;
    for (auto & aCd2 : aSet.VCdt())   
    {
        cWhitchMin<int,double> aWM=  mPC->Desc().DistanceFromBestPeek(aCd2.mPC->Desc(),aSet.SetPC());
        aVH.push_back(cHypMatchCdtBasic(aWM.Val(),*this,aCd2,aWM.Index()));
    }
    std::sort(aVH.begin(),aVH.end());
    StdOut() << "BEST COST  " << aVH[0].Cost() << " " << aVH.back().Cost() << " Cpt " << aCpt << "\n";
    
}


/*  ******************************************************** */
/*                                                           */
/*              cHypMatchCdtBasic                            */
/*                                                           */
/*  ******************************************************** */

cHypMatchCdtBasic::cHypMatchCdtBasic(double aCost,cCdtBasic & aCd1,cCdtBasic & aCd2,int aIndexOr):
   mCost    (aCost),
   mIndexOr (aIndexOr),
   mCd1     (&aCd1),
   mCd2     (&aCd2)
{
}

const double & cHypMatchCdtBasic::Cost() const {return mCost;}

/*  ******************************************************** */
/*                                                           */
/*              cSetCdtBasic                                 */
/*                                                           */
/*  ******************************************************** */

cSetCdtBasic::cSetCdtBasic(cSetAimePCAR * aSet) :
     mSetPC (aSet)
{
    for (auto & aPC : mSetPC->VPC())
       mVCdt.push_back(cCdtBasic(&aPC));
}

std::vector<cCdtBasic>&  cSetCdtBasic::VCdt()  {return mVCdt;}
cSetAimePCAR &           cSetCdtBasic::SetPC() {return *mSetPC;}

/*  ******************************************************** */
/*                                                           */
/*              cCpleSetCdtBasic                             */
/*                                                           */
/*  ******************************************************** */

cCpleSetCdtBasic::cCpleSetCdtBasic(cSetAimePCAR * aS1,cSetAimePCAR * aS2) :
     mS1 (aS1),
     mS2 (aS2)
{
}

void cCpleSetCdtBasic::MakeCost()
{
    for (auto & aCd1 : mS1.VCdt())
    {
        aCd1.MatchInit(mS2,20);
    }
}


/**  
      This class implement basic match
*/
class cAppliMatchTiePBasic : public cBaseMatchTieP
{
     public :

        cAppliMatchTiePBasic(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
        std::string  mTestObl;            ///< temporary
        std::string  mTestOPt;            ///< temporary 
        std::vector<cCpleSetCdtBasic>  mVSCB; // Pair of set of Aime/cdt
};

/* =============================================== */
/*                                                 */
/*             cAppliMatchTiePBasic                */
/*                                                 */
/* =============================================== */

cAppliMatchTiePBasic::cAppliMatchTiePBasic(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cBaseMatchTieP(aVArgs,aSpec)
{
}

cCollecSpecArg2007 & cAppliMatchTiePBasic::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
     cBaseMatchTieP::ArgObl(anArgObl);

     return anArgObl 
               <<   Arg2007(mTestObl,"Test Obl");

}

cCollecSpecArg2007 & cAppliMatchTiePBasic::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   cBaseMatchTieP::ArgOpt(anArgOpt);
   return anArgOpt
          << AOpt2007(mTestOPt,"TOpt","Test Optionnal local",{eTA2007::HDV})
   ;
}

int cAppliMatchTiePBasic::Exe() 
{
   PostInit();
   for (int aKSet=0 ; aKSet<int(mVSAPc1.size()) ; aKSet++)
   {
        mVSCB.push_back(cCpleSetCdtBasic(&mVSAPc1.at(aKSet),&mVSAPc2.at(aKSet)));
        mVSCB.back().MakeCost();
        StdOut() << "=================================\n";
   }
   return EXIT_SUCCESS;
}

/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_MatchTieP(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliMatchTiePBasic(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecMatchTieP
(
     "TieP-AimeBasicMatch",
      Alloc_MatchTieP,
      "Match caracteristic points and descriptors computed with Aime method",
      {eApF::TieP},
      {eApDT::PCar},
      {eApDT::TieP},
      __FILE__
);




};
