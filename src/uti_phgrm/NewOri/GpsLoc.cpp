/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/

#include "NewOri.h"

class cGpsLoc_Som;
class cGpsLoc_Triplet;
typedef cGpsLoc_Som *  tGLS_Ptr;

///  Classe pour definir un repere image avec juste 1 Point et 3 vecteur  
class cGpsLoc_Rep;

class cAppliGpsLoc;


/** 
   cXml_Rotation
ElRotation3D Xml2El(const cXml_Rotation & aXml)
cXml_Rotation El2Xml(const ElRotation3D & aRot);

*/
 
/* ========================================= */
/*                                           */
/*         cGpsLoc_Rep                       */
/*                                           */
/* ========================================= */



class cGpsLoc_Rep
{
     public :
        cGpsLoc_Rep(); // Pour rotation identitie
        cGpsLoc_Rep(const ElRotation3D & aPose);
        ElMatrix<double>  MatRot() const;
        const Pt3dr & Ori() const {return mOri;}
     private :
         Pt3dr mOri;
         Pt3dr mI;
         Pt3dr mJ;
         Pt3dr mK;
     
};

cGpsLoc_Rep::cGpsLoc_Rep () :
    mOri (0,0,0),
    mI   (1,0,0),
    mJ   (0,1,0),
    mK   (0,0,1)
{
}

cGpsLoc_Rep::cGpsLoc_Rep (const ElRotation3D & aPose) :
    mOri (aPose.ImAff(Pt3dr(0,0,0))),
    mI   (aPose.ImVect(Pt3dr(1,0,0))),
    mJ   (aPose.ImVect(Pt3dr(0,1,0))),
    mK   (aPose.ImVect(Pt3dr(0,0,1)))
{
}


ElMatrix<double> cGpsLoc_Rep::MatRot() const
{
   return MatFromCol(mI,mJ,mK);
}

/* ========================================= */
/*                                           */
/*           cGpsLoc_Som                     */
/*                                           */
/* ========================================= */

class cGpsLoc_Som
{
    public :
        cGpsLoc_Som(const std::string & aName);
        cGpsLoc_Rep  & Sol();
        void Save(cNewO_NameManager *);
    private :
        std::string mName;
        cGpsLoc_Rep mSol;
};

cGpsLoc_Som::cGpsLoc_Som(const std::string & aName)  :
    mName (aName)
{
}

cGpsLoc_Rep & cGpsLoc_Som::Sol() {return mSol;}

void cGpsLoc_Som::Save(cNewO_NameManager * aNM)
{
   CamStenope * aCS = aNM->CalibrationCamera(mName);

   ElRotation3D aR (mSol.Ori(),mSol.MatRot(),true);
   aCS->SetOrientation(aR.inv());

   cOrientationConique anOC =  aCS->StdExportCalibGlob();

   std::string aNameOri = aNM->NameOriOut(mName);

   MakeFileXML(anOC,aNameOri);
   std::cout << mName << " " << aCS << "\n";
}

/* ========================================= */
/*                                           */
/*         cGpsLoc_Triplet                   */
/*                                           */
/* ========================================= */

class cGpsLoc_Triplet
{
    public :
        cGpsLoc_Triplet(tGLS_Ptr  aS1,tGLS_Ptr aS2,tGLS_Ptr aS3,const cXml_Ori3ImInit & aXmlOri);
        void InitSomTrivial();
    private :
       tGLS_Ptr     mSoms[3];
       cGpsLoc_Rep  mReps[3];
       Pt3dr        mPMed;
};

cGpsLoc_Triplet::cGpsLoc_Triplet(tGLS_Ptr  aS1,tGLS_Ptr aS2,tGLS_Ptr aS3,const cXml_Ori3ImInit & aXmlOri) :
    mSoms {aS1,aS2,aS3}
{
    mReps[0] = cGpsLoc_Rep();
    mReps[1] = cGpsLoc_Rep(Xml2El(aXmlOri.Ori2On1()));
    mReps[2] = cGpsLoc_Rep(Xml2El(aXmlOri.Ori3On1()));
    mPMed = aXmlOri.PMed();
}


void cGpsLoc_Triplet::InitSomTrivial()
{
    for (auto aK : {0,1,2})
    {
         mSoms[aK]->Sol() = mReps[aK];
    }
}




/* ========================================= */
/*                                           */
/*         cGpsLoc_Triplet                   */
/*                                           */
/* ========================================= */

class cAppliGpsLoc : public cCommonMartiniAppli
{
     public :
          cAppliGpsLoc(int argc,char ** argv);
     private :
          std::string                          mDir;
          std::map<std::string,cGpsLoc_Som *>  mMapS;
          std::vector<cGpsLoc_Triplet>         mV3;
          int                                  mNbSom;
};


cAppliGpsLoc::cAppliGpsLoc(int argc,char ** argv) :
    cCommonMartiniAppli (),
    mDir                ("./")
{
   int toto, tata;
   std::string aPat;

   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(aPat,"GGGGGGpsss Name First Image", eSAM_IsExistFile) ,
        LArgMain() << EAM(tata,"GenOri",true,"Generate Ori, Def=true, false for quick process to RedTieP")
                   << ArgCMA()
   );

   cElemAppliSetFile anEASF(aPat);
   const std::vector<std::string> * aSetName =   anEASF.SetIm();
   for (auto aName : *aSetName)
   {
       mMapS[aName] = new cGpsLoc_Som(aName);
   }


   cNewO_NameManager *  aNM =NM(mDir);
   std::string aNameLTriplets = aNM->NameTopoTriplet(true);
   cXml_TopoTriplet  aLT = StdGetFromSI(aNameLTriplets,Xml_TopoTriplet);

   for (auto a3 : aLT.Triplets())
   {
       if (DicBoolFind(mMapS,a3.Name1()) && DicBoolFind(mMapS,a3.Name2()) && DicBoolFind(mMapS,a3.Name3()))
       {
          cGpsLoc_Som * aS1 = mMapS[a3.Name1()];
          cGpsLoc_Som * aS2 = mMapS[a3.Name2()];
          cGpsLoc_Som * aS3 = mMapS[a3.Name3()];

          std::string  aName3R = aNM->NameOriOptimTriplet(true,a3.Name1(),a3.Name2(),a3.Name3());
          cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);
          mV3.push_back(cGpsLoc_Triplet(aS1,aS2,aS3,aXml3Ori));
       }
   }
   mNbSom = mMapS.size();


   // Cas particulier ou il n'y a que 3 sommets, pour tester on initialise avec les
   // orientation du triplet
   if (mNbSom==3)
   {
      ELISE_ASSERT(mV3.size()==1,"Incoherent size with on triplet");
      mV3[0].InitSomTrivial();
      for (auto aPair : mMapS)
           aPair.second->Save(aNM);
   }
}


int CPP_NOGpsLoc(int argc,char ** argv)
{
   cAppliGpsLoc anAppli(argc,argv);
   return EXIT_SUCCESS;
}





