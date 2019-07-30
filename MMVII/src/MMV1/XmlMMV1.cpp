#include "include/V1VII.h"


#include "src/uti_image/NewRechPH/cParamNewRechPH.h"
#include "../CalcDescriptPCar/AimeTieP.h"





namespace MMVII
{


//=============  tNameRel ====================

void TestTimeV1V2()
{
    for (int aK= 1 ; aK<1000000 ; aK++)
    {
         int aSom=0;
         ElTimer aChronoV1;
         double aT1 = cMMVII_Appli::CurrentAppli().SecFromT0();
         for (int aI=0 ; aI<10000; aI++)
         {
              for (int aJ=0 ; aJ<10000; aJ++)
              {
                  aSom += 1/(1+aI+aJ);
              }
         }
         if (aSom==-1)
            return;
         double aT2 = cMMVII_Appli::CurrentAppli().SecFromT0();

         double aDV1 = aChronoV1.uval();
         double aDV2 = aT2-aT1;

         StdOut()  << "Ratio " << aDV1 / aDV2  << " TimeV1: " << aDV1 << "\n";
    }
}

//=============  tNameRel ====================

tNameRel  MMV1InitRel(const std::string & aName)
{
   tNameRel aRes;
   cSauvegardeNamedRel aSNR = StdGetFromPCP(aName,SauvegardeNamedRel);
   for (const auto & aCpl : aSNR.Cple())
   {
       aRes.Add(tNamePair(aCpl.N1(),aCpl.N2()));
   }

   return aRes;
}

/// Write a rel in MMV1 format
template<> void  MMv1_SaveInFile(const tNameRel & aSet,const std::string & aName)
{
   std::vector<const tNamePair *> aV;
   aSet.PutInVect(aV,true);

   cSauvegardeNamedRel aSNR;
   for (const auto & aPair : aV)
   {
      aSNR.Cple().push_back(cCpleString(aPair->V1(),aPair->V2()));
   }
   MakeFileXML(aSNR,aName);
}

//=============  tNameSet ====================

/// Read MMV1 Set
tNameSet  MMV1InitSet(const std::string & aName)
{
   tNameSet aRes ;
   cListOfName aLON = StdGetFromPCP(aName,ListOfName);
   for (const auto & el : aLON.Name())
       aRes.Add(el);
   return aRes;
}

/// Write a set in MMV1 format
template<> void  MMv1_SaveInFile(const tNameSet & aSet,const std::string & aName)
{
    std::vector<const std::string *> aV;
    aSet.PutInVect(aV,true);

    cListOfName aLON;
    for (const auto & el : aV)
        aLON.Name().push_back(*el);
    MakeFileXML(aLON,aName);
}

/********************************************************/


/// Class implementing services promized by cInterf_ExportAimeTiep

/**  This class use MMV1 libray to implement service described in cInterf_ExportAimeTiep
*/
class cImplem_ExportAimeTiep : public cInterf_ExportAimeTiep
{
     public :
         cImplem_ExportAimeTiep(bool IsMin,int ATypePt,const std::string & aName);
         virtual ~cImplem_ExportAimeTiep();

         void AddAimeTieP(const cProtoAimeTieP & aPATP ) override;
         void Export(const std::string &) override;
     private :
          cXml2007FilePt  mPtsXml;
};
/* ================================= */
/*    cInterf_ExportAimeTiep         */
/* ================================= */

cInterf_ExportAimeTiep::~cInterf_ExportAimeTiep()
{
}

cInterf_ExportAimeTiep * cInterf_ExportAimeTiep::Alloc(bool IsMin,int ATypePt,const std::string & aName)
{
    return new cImplem_ExportAimeTiep(IsMin,ATypePt,aName);
}



/* ================================= */
/*    cImplem_ExportAimeTiep         */
/* ================================= */

cImplem_ExportAimeTiep::cImplem_ExportAimeTiep(bool IsMin,int ATypePt,const std::string & aNameType)
{
    mPtsXml.IsMin() = IsMin;
    mPtsXml.TypePt() = IsMin;
    mPtsXml.NameTypePt() = aNameType;
    
}
cImplem_ExportAimeTiep::~cImplem_ExportAimeTiep()
{
}

void cImplem_ExportAimeTiep::AddAimeTieP(const cProtoAimeTieP & aPATP ) 
{
    cXml2007Pt aPXml;

    aPXml.Pt() = ToMMV1(aPATP.Pt());
    aPXml.NumOct() = aPATP.NumOct();
    aPXml.NumIm() = aPATP.NumIm();
    aPXml.ScaleInO() = aPATP.ScaleInO();
    aPXml.ScaleAbs() = aPATP.ScaleAbs();

    mPtsXml.Pts().push_back(aPXml);
}
void cImplem_ExportAimeTiep::Export(const std::string & aName)
{
     MakeFileXML(mPtsXml,aName);
}



};
